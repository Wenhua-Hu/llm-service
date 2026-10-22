"""SQLite-backed tenant and API-key storage."""

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import hmac
import os
from pathlib import Path
import sqlite3
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

    def __init__(self, directory: str, pepper: str, bootstrap_key: str):
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

    def revoke_key(self, key_id: str) -> None:
        with self.transaction() as database:
            database.execute(
                "UPDATE api_keys SET status='revoked' WHERE key_id=?", (key_id,)
            )

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
