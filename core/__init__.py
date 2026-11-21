import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT_DIR, "models")


def configure_runtime():
    """Load runtime overrides before loading any model libraries."""
    from dotenv import load_dotenv

    load_dotenv(os.path.join(ROOT_DIR, ".env"), encoding="utf-8-sig")
    os.environ.setdefault("HF_HOME", MODELS_DIR)
    os.environ.setdefault("PADDLE_PDX_CACHE_HOME", os.path.join(MODELS_DIR, "paddlex"))
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
