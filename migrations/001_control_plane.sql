PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS settings (
    name TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tenants (
    tenant_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    balance_micro_usd INTEGER NOT NULL CHECK(balance_micro_usd >= 0),
    token_limit INTEGER NOT NULL,
    rpm_limit INTEGER NOT NULL,
    concurrency_limit INTEGER NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS api_keys (
    key_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL REFERENCES tenants(tenant_id),
    name TEXT NOT NULL,
    key_hash TEXT UNIQUE NOT NULL,
    masked_key TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT
);

CREATE TABLE IF NOT EXISTS prices (
    pricing_version TEXT PRIMARY KEY,
    model TEXT NOT NULL,
    input_micro_usd_per_token INTEGER NOT NULL,
    output_micro_usd_per_token INTEGER NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS requests (
    request_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL REFERENCES tenants(tenant_id),
    key_id TEXT NOT NULL,
    parent_request_id TEXT,
    route TEXT NOT NULL,
    model TEXT NOT NULL,
    body_hash TEXT NOT NULL,
    idempotency_key TEXT,
    status TEXT NOT NULL,
    pricing_version TEXT NOT NULL REFERENCES prices(pricing_version),
    reserved_input INTEGER NOT NULL,
    reserved_output INTEGER NOT NULL,
    reserved_micro_usd INTEGER NOT NULL,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cached_tokens INTEGER,
    cost_micro_usd INTEGER,
    engine_started INTEGER NOT NULL DEFAULT 0,
    usage_evidence TEXT,
    error_code TEXT,
    created_at TEXT NOT NULL,
    created_epoch REAL NOT NULL,
    finished_at TEXT,
    UNIQUE(tenant_id,route,idempotency_key)
);

CREATE INDEX IF NOT EXISTS requests_tenant_time
ON requests(tenant_id,created_epoch);

CREATE TABLE IF NOT EXISTS adjustments (
    adjustment_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    delta_micro_usd INTEGER NOT NULL,
    reason TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    body_hash TEXT NOT NULL,
    response_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(tenant_id,idempotency_key)
);

CREATE TABLE IF NOT EXISTS audit (
    audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
    event TEXT NOT NULL,
    tenant_id TEXT,
    request_id TEXT,
    detail TEXT NOT NULL,
    created_at TEXT NOT NULL
);
