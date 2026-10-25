"""SQLite-backed tenant and API-key storage."""

from contextlib import contextmanager
import base64
from datetime import datetime, timezone
import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
import sqlite3
import time
import uuid


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


class LedgerError(Exception):
    def __init__(self, status: int, code: str, message: str | None = None):
        super().__init__(message or code.replace("_", " "))
        self.status = status
        self.code = code
        self.message = message or code.replace("_", " ")


class Store:
    """Own the durable identity state for one gateway instance."""

    def __init__(self, directory: str, pepper: str, bootstrap_key: str, model: str = "local-model"):
        if len(pepper.encode()) < 32:
            raise ValueError("KEY_HASH_SECRET must contain at least 32 bytes")

        root = Path(__file__).resolve().parents[1]
        data_dir = Path(directory).expanduser().resolve()
        if data_dir in {Path("/"), Path.home(), root}:
            raise ValueError("DATA_DIR must be a dedicated private subdirectory")

        data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(data_dir, 0o700)
        self.path = data_dir / "ledger.sqlite3"
        self._pepper = pepper.encode()

        migration = root / "migrations" / "001_control_plane.sql"
        pricing_config = json.loads((root / "configs" / "pricing.json").read_text(encoding="utf-8"))
        input_rate = pricing_config.get("input_micro_usd_per_token")
        output_rate = pricing_config.get("output_micro_usd_per_token")
        if (
            pricing_config.get("status") != "experimental"
            or isinstance(input_rate, bool)
            or not isinstance(input_rate, int)
            or input_rate < 0
            or isinstance(output_rate, bool)
            or not isinstance(output_rate, int)
            or output_rate < 0
        ):
            raise ValueError("configs/pricing.json must contain nonnegative experimental token rates")
        with self.connection() as database:
            database.executescript(migration.read_text())
        os.chmod(self.path, 0o600)

        with self.transaction() as database:
            fingerprint = self.key_hash("credential-service/1 pepper identity")
            existing = database.execute(
                "SELECT value FROM settings WHERE name='pepper_fingerprint'"
            ).fetchone()
            if existing and not hmac.compare_digest(existing["value"], fingerprint):
                raise ValueError("KEY_HASH_SECRET differs from the persistent ledger")
            database.execute(
                "INSERT OR IGNORE INTO settings(name,value) VALUES(?,?)",
                ("pepper_fingerprint", fingerprint),
            )
            database.execute(
                """
                INSERT OR IGNORE INTO prices(
                    pricing_version,model,input_micro_usd_per_token,
                    output_micro_usd_per_token,created_at
                ) VALUES(?,?,?,?,?)
                """,
                (f"pricing-{model}-1", model, input_rate, output_rate, utc_now()),
            )

            existing_key = database.execute(
                "SELECT key_id FROM api_keys WHERE key_hash=?",
                (self.key_hash(bootstrap_key),),
            ).fetchone()
            if not existing_key:
                tenant_id = "tenant_benchmark"
                database.execute(
                    """
                    INSERT OR IGNORE INTO tenants(
                        tenant_id,name,status,balance_micro_usd,token_limit,
                        rpm_limit,concurrency_limit,created_at
                    ) VALUES(?,?,?,?,?,?,?,?)
                    """,
                    (
                        tenant_id,
                        "Benchmark",
                        "active",
                        10**15,
                        10**12,
                        100000,
                        120,
                        utc_now(),
                    ),
                )
                self._insert_key(
                    database,
                    tenant_id=tenant_id,
                    name="Bootstrap benchmark",
                    raw_key=bootstrap_key,
                )
        self.recover_incomplete_requests(force=True)

    @staticmethod
    def _new_api_key() -> str:
        return "llm_" + secrets.token_urlsafe(32)

    @staticmethod
    def _validate_limit(name: str, value: object, *, minimum: int = 0) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise LedgerError(400, "invalid_request", f"{name} must be an integer >= {minimum}")
        return value

    def create_tenant(
        self,
        *,
        name: str,
        balance_micro_usd: int,
        token_limit: int,
        rpm_limit: int,
        concurrency_limit: int,
    ) -> dict[str, object]:
        if not isinstance(name, str) or not name.strip():
            raise LedgerError(400, "invalid_request", "name is required")
        balance_micro_usd = self._validate_limit("balance_micro_usd", balance_micro_usd)
        token_limit = self._validate_limit("token_limit", token_limit)
        rpm_limit = self._validate_limit("rpm_limit", rpm_limit)
        concurrency_limit = self._validate_limit("concurrency_limit", concurrency_limit, minimum=1)
        tenant_id = "tenant_" + uuid.uuid4().hex
        created_at = utc_now()
        with self.transaction() as database:
            database.execute(
                """INSERT INTO tenants(
                    tenant_id,name,status,balance_micro_usd,token_limit,
                    rpm_limit,concurrency_limit,created_at
                ) VALUES(?,?,'active',?,?,?,?,?)""",
                (tenant_id, name.strip(), balance_micro_usd, token_limit, rpm_limit, concurrency_limit, created_at),
            )
            database.execute(
                "INSERT INTO audit(event,tenant_id,detail,created_at) VALUES(?,?,?,?)",
                ("tenant_created", tenant_id, json.dumps({"name": name.strip()}), created_at),
            )
        return {
            "tenant_id": tenant_id,
            "name": name.strip(),
            "status": "active",
            "balance_micro_usd": balance_micro_usd,
            "token_limit": token_limit,
            "rpm_limit": rpm_limit,
            "concurrency_limit": concurrency_limit,
        }

    def list_tenants(self) -> list[dict[str, object]]:
        with self.connection() as database:
            rows = database.execute(
                """SELECT t.tenant_id,t.name,t.status,t.balance_micro_usd,
                          t.token_limit,t.rpm_limit,t.concurrency_limit,
                          COALESCE(SUM(CASE WHEN r.status='settled'
                            THEN COALESCE(r.input_tokens,0)+COALESCE(r.output_tokens,0)
                            ELSE 0 END),0) AS used_tokens,
                          COALESCE(SUM(CASE WHEN r.status IN ('reserved','running','pending_reconciliation','cancelled_unknown')
                            THEN r.reserved_input+r.reserved_output ELSE 0 END),0) AS reserved_tokens
                   FROM tenants t LEFT JOIN requests r USING(tenant_id)
                   GROUP BY t.tenant_id ORDER BY t.created_at,t.tenant_id"""
            ).fetchall()
        return [dict(row) for row in rows]

    def update_tenant(self, tenant_id: str, changes: dict[str, object]) -> dict[str, object]:
        allowed = {"status", "token_limit", "rpm_limit", "concurrency_limit"}
        if not changes or set(changes) - allowed:
            raise LedgerError(400, "invalid_request", "only status and quota limits may be changed")
        if "status" in changes and changes["status"] not in {"active", "disabled"}:
            raise LedgerError(400, "invalid_request", "status must be active or disabled")
        validated = dict(changes)
        for name in {"token_limit", "rpm_limit", "concurrency_limit"} & set(changes):
            validated[name] = self._validate_limit(
                name, changes[name], minimum=1 if name == "concurrency_limit" else 0
            )
        assignments = ",".join(f"{field}=?" for field in validated)
        with self.transaction() as database:
            cursor = database.execute(
                f"UPDATE tenants SET {assignments} WHERE tenant_id=?",
                (*validated.values(), tenant_id),
            )
            if cursor.rowcount != 1:
                raise LedgerError(404, "tenant_not_found")
            database.execute(
                "INSERT INTO audit(event,tenant_id,detail,created_at) VALUES(?,?,?,?)",
                ("tenant_updated", tenant_id, json.dumps(validated, sort_keys=True), utc_now()),
            )
            row = database.execute("SELECT * FROM tenants WHERE tenant_id=?", (tenant_id,)).fetchone()
        return dict(row)

    def adjust_credits(
        self, tenant_id: str, *, delta_micro_usd: int, reason: str, idempotency_key: str
    ) -> dict[str, object]:
        if isinstance(delta_micro_usd, bool) or not isinstance(delta_micro_usd, int):
            raise LedgerError(400, "invalid_request", "delta_micro_usd must be an integer")
        if not isinstance(reason, str) or not reason.strip() or not idempotency_key:
            raise LedgerError(400, "invalid_request", "reason and Idempotency-Key are required")
        body_hash = hashlib.sha256(
            json.dumps({"delta_micro_usd": delta_micro_usd, "reason": reason.strip()}, sort_keys=True).encode()
        ).hexdigest()
        with self.transaction() as database:
            previous = database.execute(
                "SELECT body_hash,response_json FROM adjustments WHERE tenant_id=? AND idempotency_key=?",
                (tenant_id, idempotency_key),
            ).fetchone()
            if previous:
                if not hmac.compare_digest(previous["body_hash"], body_hash):
                    raise LedgerError(409, "idempotency_conflict")
                return json.loads(previous["response_json"])
            tenant = database.execute(
                "SELECT balance_micro_usd FROM tenants WHERE tenant_id=?", (tenant_id,)
            ).fetchone()
            if not tenant:
                raise LedgerError(404, "tenant_not_found")
            balance = tenant["balance_micro_usd"] + delta_micro_usd
            if balance < 0:
                raise LedgerError(402, "insufficient_quota")
            adjustment_id = "adj_" + uuid.uuid4().hex
            created_at = utc_now()
            response = {"adjustment_id": adjustment_id, "tenant_id": tenant_id, "balance_micro_usd": balance}
            database.execute(
                "UPDATE tenants SET balance_micro_usd=? WHERE tenant_id=?", (balance, tenant_id)
            )
            database.execute(
                """INSERT INTO adjustments(
                    adjustment_id,tenant_id,delta_micro_usd,reason,idempotency_key,
                    body_hash,response_json,created_at
                ) VALUES(?,?,?,?,?,?,?,?)""",
                (adjustment_id, tenant_id, delta_micro_usd, reason.strip(), idempotency_key, body_hash,
                 json.dumps(response, sort_keys=True), created_at),
            )
            database.execute(
                "INSERT INTO audit(event,tenant_id,detail,created_at) VALUES(?,?,?,?)",
                ("credit_adjusted", tenant_id, json.dumps({"adjustment_id": adjustment_id, "reason": reason.strip()}), created_at),
            )
        return response

    def create_key(self, tenant_id: str, *, name: str, expires_at: str | None = None) -> dict[str, object]:
        if not isinstance(name, str) or not name.strip():
            raise LedgerError(400, "invalid_request", "name is required")
        expires_at = self._normalize_expiration(expires_at)
        raw_key = self._new_api_key()
        created_at = utc_now()
        with self.transaction() as database:
            if not database.execute("SELECT 1 FROM tenants WHERE tenant_id=?", (tenant_id,)).fetchone():
                raise LedgerError(404, "tenant_not_found")
            key_id = self._insert_key(
                database, tenant_id=tenant_id, name=name.strip(), raw_key=raw_key, expires_at=expires_at
            )
            database.execute(
                "INSERT INTO audit(event,tenant_id,detail,created_at) VALUES(?,?,?,?)",
                ("key_created", tenant_id, json.dumps({"key_id": key_id}), created_at),
            )
        return {"key_id": key_id, "api_key": raw_key, "status": "active", "expires_at": expires_at}

    def list_keys(self, tenant_id: str) -> list[dict[str, object]]:
        with self.connection() as database:
            if not database.execute("SELECT 1 FROM tenants WHERE tenant_id=?", (tenant_id,)).fetchone():
                raise LedgerError(404, "tenant_not_found")
            rows = database.execute(
                """SELECT key_id,name,masked_key,status,created_at,expires_at
                   FROM api_keys WHERE tenant_id=? ORDER BY created_at,key_id""",
                (tenant_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def _normalize_expiration(expires_at: str | None) -> str | None:
        if expires_at is None:
            return None
        try:
            parsed = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
        except (AttributeError, ValueError) as error:
            raise LedgerError(400, "invalid_request", "expires_at must be an ISO-8601 timestamp") from error
        if parsed.tzinfo is None:
            raise LedgerError(400, "invalid_request", "expires_at must include a timezone")
        if parsed.astimezone(timezone.utc) <= datetime.now(timezone.utc):
            raise LedgerError(400, "invalid_request", "expires_at must be in the future")
        return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    def rotate_key(self, key_id: str, *, expires_at: str | None = None) -> dict[str, object]:
        expires_at = self._normalize_expiration(expires_at)
        raw_key = self._new_api_key()
        with self.transaction() as database:
            old = database.execute(
                "SELECT tenant_id,name,status FROM api_keys WHERE key_id=?", (key_id,)
            ).fetchone()
            if not old:
                raise LedgerError(404, "key_not_found")
            if old["status"] != "active":
                raise LedgerError(409, "key_not_active")
            database.execute("UPDATE api_keys SET status='revoked' WHERE key_id=?", (key_id,))
            new_key_id = self._insert_key(
                database, tenant_id=old["tenant_id"], name=old["name"], raw_key=raw_key,
                expires_at=expires_at,
            )
            database.execute(
                "INSERT INTO audit(event,tenant_id,detail,created_at) VALUES(?,?,?,?)",
                ("key_rotated", old["tenant_id"], json.dumps({"key_id": key_id, "new_key_id": new_key_id}), utc_now()),
            )
        return {
            "key_id": new_key_id,
            "api_key": raw_key,
            "replaced_key_id": key_id,
            "status": "active",
            "replaced_status": "revoked",
            "expires_at": expires_at,
        }

    def revoke_key(self, key_id: str) -> dict[str, str]:
        with self.transaction() as database:
            row = database.execute("SELECT tenant_id FROM api_keys WHERE key_id=?", (key_id,)).fetchone()
            if not row:
                raise LedgerError(404, "key_not_found")
            database.execute("UPDATE api_keys SET status='revoked' WHERE key_id=?", (key_id,))
            database.execute(
                "INSERT INTO audit(event,tenant_id,detail,created_at) VALUES(?,?,?,?)",
                ("key_revoked", row["tenant_id"], json.dumps({"key_id": key_id}), utc_now()),
            )
        return {"key_id": key_id, "status": "revoked"}

    def get_pricing(self, model: str) -> dict[str, object]:
        with self.connection() as database:
            row = database.execute(
                "SELECT * FROM prices WHERE model=? ORDER BY created_at DESC,pricing_version DESC LIMIT 1",
                (model,),
            ).fetchone()
        if not row:
            raise LedgerError(404, "pricing_not_found")
        return dict(row)

    def set_pricing(self, model: str, *, input_price: int, output_price: int) -> dict[str, object]:
        if not isinstance(model, str) or not model.strip():
            raise LedgerError(400, "invalid_request", "model is required")
        input_price = self._validate_limit("input_micro_usd_per_token", input_price)
        output_price = self._validate_limit("output_micro_usd_per_token", output_price)
        version = "pricing_" + uuid.uuid4().hex
        created_at = utc_now()
        with self.transaction() as database:
            database.execute(
                """INSERT INTO prices(
                    pricing_version,model,input_micro_usd_per_token,
                    output_micro_usd_per_token,created_at
                ) VALUES(?,?,?,?,?)""",
                (version, model.strip(), input_price, output_price, created_at),
            )
            database.execute(
                "INSERT INTO audit(event,detail,created_at) VALUES(?,?,?)",
                ("pricing_updated", json.dumps({"model": model.strip(), "pricing_version": version}), created_at),
            )
        return {
            "pricing_version": version,
            "model": model.strip(),
            "input_micro_usd_per_token": input_price,
            "output_micro_usd_per_token": output_price,
            "created_at": created_at,
        }

    def reserve_request(
        self,
        *,
        auth: dict[str, str],
        route: str,
        model: str,
        body: dict[str, object],
        input_tokens: int,
        max_output_tokens: int,
        idempotency_key: str | None = None,
        parent_request_id: str | None = None,
    ) -> dict[str, object]:
        input_tokens = self._validate_limit("input_tokens", input_tokens)
        max_output_tokens = self._validate_limit("max_output_tokens", max_output_tokens)
        tenant_id = auth["tenant_id"]
        key_id = auth["key_id"]
        body_hash = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        now = time.time()
        request_id = "req_" + uuid.uuid4().hex
        with self.transaction() as database:
            if idempotency_key:
                previous = database.execute(
                    """SELECT request_id,body_hash,status FROM requests
                       WHERE tenant_id=? AND route=? AND idempotency_key=?""",
                    (tenant_id, route, idempotency_key),
                ).fetchone()
                if previous:
                    if previous["body_hash"] != body_hash:
                        code = "idempotency_conflict"
                    elif previous["status"] in {"reserved", "running", "pending_reconciliation"}:
                        code = "request_in_progress"
                    else:
                        code = "request_already_completed"
                    raise LedgerError(409, code, f"Request already exists as {previous['request_id']}")
            tenant = database.execute(
                "SELECT status,balance_micro_usd,token_limit,rpm_limit,concurrency_limit FROM tenants WHERE tenant_id=?",
                (tenant_id,),
            ).fetchone()
            key = database.execute(
                "SELECT status,expires_at FROM api_keys WHERE key_id=? AND tenant_id=?",
                (key_id, tenant_id),
            ).fetchone()
            if not tenant or tenant["status"] != "active":
                raise LedgerError(403, "tenant_disabled")
            if not key or key["status"] != "active" or (key["expires_at"] and key["expires_at"] <= utc_now()):
                raise LedgerError(401, "invalid_api_key")
            price = database.execute(
                "SELECT pricing_version,input_micro_usd_per_token,output_micro_usd_per_token "
                "FROM prices WHERE model=? ORDER BY created_at DESC,pricing_version DESC LIMIT 1",
                (model,),
            ).fetchone()
            if not price:
                raise LedgerError(404, "pricing_not_found")
            reserved_cost = (
                input_tokens * price["input_micro_usd_per_token"]
                + max_output_tokens * price["output_micro_usd_per_token"]
            )
            current = database.execute(
                """SELECT COALESCE(SUM(input_tokens+output_tokens),0) AS used_tokens,
                          COALESCE(SUM(CASE WHEN status IN ('reserved','running','pending_reconciliation','cancelled_unknown')
                            THEN reserved_input+reserved_output ELSE 0 END),0) AS reserved_tokens,
                          COALESCE(SUM(CASE WHEN status IN ('reserved','running','pending_reconciliation','cancelled_unknown')
                            THEN reserved_micro_usd ELSE 0 END),0) AS reserved_cost,
                          COALESCE(SUM(CASE WHEN status IN ('reserved','running','pending_reconciliation') THEN 1 ELSE 0 END),0) AS inflight,
                          COALESCE(SUM(CASE WHEN created_epoch>=? THEN 1 ELSE 0 END),0) AS recent_requests
                   FROM requests WHERE tenant_id=?""",
                (now - 60, tenant_id),
            ).fetchone()
            if current["inflight"] >= tenant["concurrency_limit"]:
                raise LedgerError(429, "concurrency_limit")
            if current["recent_requests"] >= tenant["rpm_limit"]:
                raise LedgerError(429, "rpm_limit")
            if current["used_tokens"] + current["reserved_tokens"] + input_tokens + max_output_tokens > tenant["token_limit"]:
                raise LedgerError(402, "insufficient_quota")
            if tenant["balance_micro_usd"] - current["reserved_cost"] < reserved_cost:
                raise LedgerError(402, "insufficient_quota")
            database.execute(
                """INSERT INTO requests(
                    request_id,tenant_id,key_id,parent_request_id,route,model,body_hash,
                    idempotency_key,status,pricing_version,reserved_input,reserved_output,
                    reserved_micro_usd,engine_started,created_at,created_epoch
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,0,?,?)""",
                (request_id, tenant_id, key_id, parent_request_id, route, model, body_hash,
                 idempotency_key, "reserved", price["pricing_version"], input_tokens,
                 max_output_tokens, reserved_cost, utc_now(), now),
            )
        return {
            "request_id": request_id,
            "pricing_version": price["pricing_version"],
            "input_price": price["input_micro_usd_per_token"],
            "output_price": price["output_micro_usd_per_token"],
            "reserved_micro_usd": reserved_cost,
            "reserved_input": input_tokens,
            "reserved_output": max_output_tokens,
        }

    def mark_engine_started(self, request_id: str) -> None:
        with self.transaction() as database:
            database.execute(
                "UPDATE requests SET status='running',engine_started=1 WHERE request_id=? AND status='reserved'",
                (request_id,),
            )

    def mark_pending_reconciliation(self, request_id: str, error_code: str) -> None:
        with self.transaction() as database:
            database.execute(
                """UPDATE requests SET status='pending_reconciliation',error_code=?
                   WHERE request_id=? AND status IN ('reserved','running')""",
                (error_code, request_id),
            )

    def mark_cancelled_unknown(self, request_id: str, error_code: str) -> None:
        """Keep unknown usage auditable while freeing the tenant concurrency slot."""

        with self.transaction() as database:
            database.execute(
                """UPDATE requests SET status='cancelled_unknown',error_code=?
                   WHERE request_id=? AND status IN ('reserved','running')""",
                (error_code, request_id),
            )

    def release_request(self, request_id: str, error_code: str) -> None:
        """Release a reservation only when the engine confirms it did no generation."""

        with self.transaction() as database:
            database.execute(
                """UPDATE requests SET status='failed',error_code=?,reserved_input=0,
                          reserved_output=0,reserved_micro_usd=0,finished_at=?
                   WHERE request_id=? AND status IN ('reserved','running') AND usage_evidence IS NULL""",
                (error_code, utc_now(), request_id),
            )

    def journal_usage(self, request_id: str, input_tokens: int, output_tokens: int, cached_tokens: int = 0) -> None:
        input_tokens = self._validate_limit("input_tokens", input_tokens)
        output_tokens = self._validate_limit("output_tokens", output_tokens)
        cached_tokens = self._validate_limit("cached_tokens", cached_tokens)
        evidence = json.dumps(
            {"input_tokens": input_tokens, "output_tokens": output_tokens, "cached_tokens": cached_tokens},
            sort_keys=True,
        )
        with self.transaction() as database:
            row = database.execute("SELECT status FROM requests WHERE request_id=?", (request_id,)).fetchone()
            if not row:
                raise LedgerError(404, "request_not_found")
            if row["status"] == "settled":
                return
            database.execute(
                """UPDATE requests SET usage_evidence=?,input_tokens=?,output_tokens=?,cached_tokens=?
                   WHERE request_id=?""",
                (evidence, input_tokens, output_tokens, cached_tokens, request_id),
            )

    def settle_request(self, request_id: str) -> dict[str, object]:
        with self.transaction() as database:
            request = database.execute(
                """SELECT tenant_id,status,pricing_version,reserved_micro_usd,
                          input_tokens,output_tokens,cached_tokens
                   FROM requests WHERE request_id=?""",
                (request_id,),
            ).fetchone()
            if not request:
                raise LedgerError(404, "request_not_found")
            if request["status"] == "settled":
                return {"request_id": request_id, "status": "settled", "cost_micro_usd": None}
            if request["input_tokens"] is None or request["output_tokens"] is None:
                raise LedgerError(409, "usage_not_available")
            price = database.execute(
                "SELECT input_micro_usd_per_token,output_micro_usd_per_token FROM prices WHERE pricing_version=?",
                (request["pricing_version"],),
            ).fetchone()
            cost = request["input_tokens"] * price["input_micro_usd_per_token"] + request["output_tokens"] * price[
                "output_micro_usd_per_token"
            ]
            if cost > request["reserved_micro_usd"]:
                database.execute(
                    "UPDATE requests SET status='pending_reconciliation',error_code='usage_exceeds_reservation' "
                    "WHERE request_id=?",
                    (request_id,),
                )
                return {
                    "request_id": request_id,
                    "status": "pending_reconciliation",
                    "error_code": "usage_exceeds_reservation",
                }
            database.execute(
                "UPDATE tenants SET balance_micro_usd=balance_micro_usd-? WHERE tenant_id=?",
                (cost, request["tenant_id"]),
            )
            database.execute(
                """UPDATE requests SET status='settled',cost_micro_usd=?,
                          reserved_input=0,reserved_output=0,reserved_micro_usd=0,finished_at=?
                   WHERE request_id=?""",
                (cost, utc_now(), request_id),
            )
            database.execute(
                "INSERT INTO audit(event,tenant_id,request_id,detail,created_at) VALUES(?,?,?,?,?)",
                ("request_settled", request["tenant_id"], request_id, json.dumps({"cost_micro_usd": cost}), utc_now()),
            )
        return {"request_id": request_id, "status": "settled", "cost_micro_usd": cost}

    def reconcile_unknown(self, request_id: str, *, force: bool = False) -> bool:
        with self.transaction() as database:
            row = database.execute(
                """SELECT tenant_id,status,reserved_micro_usd,created_epoch,usage_evidence
                   FROM requests WHERE request_id=?""",
                (request_id,),
            ).fetchone()
            if not row or row["status"] not in {"reserved", "running", "pending_reconciliation", "cancelled_unknown"}:
                return False
            if row["usage_evidence"] is not None:
                return False
            if not force and time.time() - row["created_epoch"] < 60:
                return False
            released = row["reserved_micro_usd"]
            database.execute(
                """UPDATE requests SET status='reconciled_unknown',reserved_input=0,
                          reserved_output=0,reserved_micro_usd=0,error_code='usage_unknown',finished_at=?
                   WHERE request_id=?""",
                (utc_now(), request_id),
            )
            adjustment_id = "adj_" + uuid.uuid4().hex
            idem = f"recovery:{request_id}"
            detail = {"request_id": request_id, "usage": "unknown", "released_micro_usd": released}
            database.execute(
                """INSERT INTO adjustments(
                    adjustment_id,tenant_id,delta_micro_usd,reason,idempotency_key,
                    body_hash,response_json,created_at
                ) VALUES(?,?,?,?,?,?,?,?)""",
                (adjustment_id, row["tenant_id"], 0, "recovery_release_unknown_usage", idem,
                 hashlib.sha256(json.dumps(detail, sort_keys=True).encode()).hexdigest(),
                 json.dumps(detail, sort_keys=True), utc_now()),
            )
            database.execute(
                "INSERT INTO audit(event,tenant_id,request_id,detail,created_at) VALUES(?,?,?,?,?)",
                ("recovery_unknown_usage", row["tenant_id"], request_id, json.dumps(detail), utc_now()),
            )
        return True

    def recover_incomplete_requests(self, *, force: bool = False) -> None:
        with self.connection() as database:
            rows = database.execute(
                "SELECT request_id,usage_evidence FROM requests WHERE status IN ('reserved','running','pending_reconciliation','cancelled_unknown')"
            ).fetchall()
        for row in rows:
            if row["usage_evidence"]:
                self.settle_request(row["request_id"])
            else:
                self.reconcile_unknown(row["request_id"], force=force)

    @staticmethod
    def _epoch_bound(value: str | None, label: str) -> float | None:
        if value is None:
            return None
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except (AttributeError, ValueError) as error:
            raise LedgerError(400, "invalid_request", f"{label} must be ISO-8601") from error
        if parsed.tzinfo is None:
            raise LedgerError(400, "invalid_request", f"{label} must include a timezone")
        return parsed.timestamp()

    @staticmethod
    def _encode_cursor(created_epoch: float, request_id: str) -> str:
        raw = json.dumps([created_epoch, request_id], separators=(",", ":")).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    @staticmethod
    def _decode_cursor(cursor: str | None) -> tuple[float, str] | None:
        if cursor is None:
            return None
        try:
            padded = cursor + "=" * (-len(cursor) % 4)
            value = json.loads(base64.urlsafe_b64decode(padded).decode())
            if (
                not isinstance(value, list)
                or len(value) != 2
                or isinstance(value[0], bool)
                or not isinstance(value[0], (int, float))
                or not isinstance(value[1], str)
            ):
                raise ValueError
            return float(value[0]), value[1]
        except (ValueError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise LedgerError(400, "invalid_cursor") from error

    def usage(
        self,
        tenant_id: str | None = None,
        *,
        from_time: str | None = None,
        to_time: str | None = None,
        cursor: str | None = None,
        limit: int = 100,
    ) -> dict[str, object]:
        limit = self._validate_limit("limit", limit, minimum=1)
        if limit > 100:
            raise LedgerError(400, "invalid_request", "limit must be at most 100")
        lower = self._epoch_bound(from_time, "from")
        upper = self._epoch_bound(to_time, "to")
        if lower is not None and upper is not None and lower > upper:
            raise LedgerError(400, "invalid_request", "from must not be after to")
        after = self._decode_cursor(cursor)
        filters = []
        params: list[object] = []
        if tenant_id:
            filters.append("tenant_id=?")
            params.append(tenant_id)
        if lower is not None:
            filters.append("created_epoch>=?")
            params.append(lower)
        if upper is not None:
            filters.append("created_epoch<=?")
            params.append(upper)
        where = "WHERE " + " AND ".join(filters) if filters else ""
        with self.connection() as database:
            if tenant_id:
                tenant = database.execute(
                    "SELECT tenant_id,balance_micro_usd FROM tenants WHERE tenant_id=?", (tenant_id,)
                ).fetchone()
                if not tenant:
                    raise LedgerError(404, "tenant_not_found")
                tenant_id_value = tenant["tenant_id"]
                balance = tenant["balance_micro_usd"]
            else:
                tenant_id_value = None
                balance = None
            totals = database.execute(
                f"""SELECT COALESCE(SUM(input_tokens),0) AS input_tokens,
                           COALESCE(SUM(output_tokens),0) AS output_tokens,
                           COALESCE(SUM(cost_micro_usd),0) AS cost_micro_usd,
                           COALESCE(SUM(CASE WHEN status IN ('reserved','running','pending_reconciliation','cancelled_unknown')
                             THEN reserved_micro_usd ELSE 0 END),0) AS reserved_micro_usd
                    FROM requests {where}""",
                tuple(params),
            ).fetchone()
            row_filters = list(filters)
            row_params = list(params)
            if after is not None:
                row_filters.append("(created_epoch<? OR (created_epoch=? AND request_id<?))")
                row_params.extend([after[0], after[0], after[1]])
            row_where = "WHERE " + " AND ".join(row_filters) if row_filters else ""
            rows = database.execute(
                f"""SELECT request_id,route,parent_request_id,tenant_id,key_id,model,status,
                           input_tokens,output_tokens,cost_micro_usd,pricing_version,
                           created_at AS timestamp,created_epoch
                    FROM requests {row_where} ORDER BY created_epoch DESC,request_id DESC LIMIT ?""",
                (*row_params, limit + 1),
            ).fetchall()
        has_more = len(rows) > limit
        visible_rows = rows[:limit]
        next_cursor = (
            self._encode_cursor(visible_rows[-1]["created_epoch"], visible_rows[-1]["request_id"])
            if has_more and visible_rows
            else None
        )
        summary: dict[str, object] = dict(totals)
        summary["total_tokens"] = totals["input_tokens"] + totals["output_tokens"]
        if tenant_id_value is not None:
            summary["balance_micro_usd"] = balance
        items = [
            {key: value for key, value in dict(row).items() if key != "created_epoch"}
            for row in visible_rows
        ]
        for item in items:
            if item["status"] == "settled":
                item["status"] = "completed"
        return {
            "summary": summary,
            "items": items,
            "next_cursor": next_cursor,
        }

    def audit_events(self, tenant_id: str | None = None) -> list[dict[str, object]]:
        where = "WHERE tenant_id=?" if tenant_id else ""
        params = (tenant_id,) if tenant_id else ()
        with self.connection() as database:
            if tenant_id and not database.execute(
                "SELECT 1 FROM tenants WHERE tenant_id=?", params
            ).fetchone():
                raise LedgerError(404, "tenant_not_found")
            rows = database.execute(
                f"""SELECT audit_id,event,tenant_id,request_id,detail,created_at
                    FROM audit {where} ORDER BY audit_id DESC LIMIT 100""",
                params,
            ).fetchall()
        return [dict(row) for row in rows]

    @contextmanager
    def connection(self):
        database = sqlite3.connect(self.path, timeout=10)
        database.row_factory = sqlite3.Row
        database.execute("PRAGMA foreign_keys=ON")
        database.execute("PRAGMA busy_timeout=5000")
        try:
            yield database
        finally:
            database.close()

    @contextmanager
    def transaction(self):
        with self.connection() as database:
            database.execute("BEGIN IMMEDIATE")
            try:
                yield database
                database.commit()
            except BaseException:
                database.rollback()
                raise

    def key_hash(self, raw_key: str) -> str:
        return hmac.new(self._pepper, raw_key.encode(), hashlib.sha256).hexdigest()

    def _insert_key(
        self,
        database: sqlite3.Connection,
        *,
        tenant_id: str,
        name: str,
        raw_key: str,
        expires_at: str | None = None,
    ) -> str:
        key_id = "key_" + uuid.uuid4().hex
        prefix = raw_key[:4]
        suffix = raw_key[-4:] if len(raw_key) >= 4 else raw_key
        database.execute(
            """
            INSERT INTO api_keys(
                key_id,tenant_id,name,key_hash,masked_key,status,created_at,expires_at
            ) VALUES(?,?,?,?,?,?,?,?)
            """,
            (
                key_id,
                tenant_id,
                name,
                self.key_hash(raw_key),
                f"{prefix}...{suffix}",
                "active",
                utc_now(),
                expires_at,
            ),
        )
        return key_id

    def authenticate(self, raw_key: str) -> dict[str, str]:
        with self.connection() as database:
            row = database.execute(
                """
                SELECT k.key_id,k.tenant_id,k.status,k.expires_at,
                       t.status AS tenant_status
                FROM api_keys AS k
                JOIN tenants AS t USING(tenant_id)
                WHERE k.key_hash=?
                """,
                (self.key_hash(raw_key),),
            ).fetchone()

        if (
            not row
            or row["status"] != "active"
            or (row["expires_at"] and row["expires_at"] <= utc_now())
        ):
            raise LedgerError(401, "invalid_api_key")
        if row["tenant_status"] != "active":
            raise LedgerError(403, "tenant_disabled")
        return {"tenant_id": row["tenant_id"], "key_id": row["key_id"]}

    def set_tenant_status(self, tenant_id: str, status: str) -> None:
        if status not in {"active", "disabled"}:
            raise ValueError("tenant status must be active or disabled")
        with self.transaction() as database:
            database.execute(
                "UPDATE tenants SET status=? WHERE tenant_id=?", (status, tenant_id)
            )

    def journal_mode(self) -> str:
        with self.connection() as database:
            return str(database.execute("PRAGMA journal_mode").fetchone()[0])
