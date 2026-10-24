"""Environment-backed service configuration."""

from dataclasses import dataclass
import os


@dataclass(frozen=True, slots=True)
class Settings:
    """Non-secret runtime settings used by the gateway."""

    service_name: str
    service_version: str
    model_id: str
    model_revision: str
    engine_base_url: str | None
    api_key: str
    key_hash_secret: str
    data_dir: str
    gateway_slots: int = 4
    gateway_queue: int = 16
    profile: str = "baseline"
    admin_api_key: str = ""

    @classmethod
    def from_env(cls) -> "Settings":
        engine_base_url = os.getenv("ENGINE_BASE_URL")
        api_key = os.getenv("API_KEY", "")
        if not api_key:
            raise ValueError("API_KEY required")
        key_hash_secret = os.getenv("KEY_HASH_SECRET", "")
        if len(key_hash_secret.encode()) < 32:
            raise ValueError("KEY_HASH_SECRET must contain at least 32 bytes")
        data_dir = os.getenv("DATA_DIR", "")
        if not data_dir:
            raise ValueError("DATA_DIR required")
        gateway_slots = int(os.getenv("GATEWAY_SLOTS", "4"))
        gateway_queue = int(os.getenv("GATEWAY_QUEUE", "16"))
        profile = os.getenv("BENCH_PROFILE", "baseline")
        if not 1 <= gateway_slots <= 64:
            raise ValueError("GATEWAY_SLOTS must be between 1 and 64")
        if not 0 <= gateway_queue <= 112 or gateway_slots + gateway_queue > 120:
            raise ValueError("GATEWAY_QUEUE must be 0..112 and total capacity at most 120")
        if profile not in {"baseline", "optimized"}:
            raise ValueError("BENCH_PROFILE must be baseline or optimized")
        admin_api_key = os.getenv("ADMIN_API_KEY", "")
        if admin_api_key and admin_api_key == api_key:
            raise ValueError("ADMIN_API_KEY must be distinct from API_KEY")
        return cls(
            service_name=os.getenv("SERVICE_NAME", "llm-service"),
            service_version=os.getenv("SERVICE_VERSION", "0.1.0"),
            model_id=os.getenv("MODEL_ID", "local-model"),
            model_revision=os.getenv("MODEL_REVISION", "development"),
            engine_base_url=engine_base_url.rstrip("/") if engine_base_url else None,
            api_key=api_key,
            key_hash_secret=key_hash_secret,
            data_dir=data_dir,
            gateway_slots=gateway_slots,
            gateway_queue=gateway_queue,
            profile=profile,
            admin_api_key=admin_api_key,
        )
