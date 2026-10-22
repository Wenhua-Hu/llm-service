import os
import tempfile


os.environ.setdefault("API_KEY", "test-api-key")
os.environ.setdefault("MODEL_ID", "test-model")
os.environ.setdefault("MODEL_REVISION", "test-revision")
os.environ.setdefault("KEY_HASH_SECRET", "test-key-hash-secret-at-least-32-bytes")
os.environ.setdefault("DATA_DIR", tempfile.mkdtemp(prefix="llm-service-tests-"))
