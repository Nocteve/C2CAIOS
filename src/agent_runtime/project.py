"""Project-local paths and isolated cache settings."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def configure_environment():
    # Only this Python process and its children see these overrides.
    os.environ["HF_HOME"] = str(ROOT / ".cache" / "huggingface")
    os.environ["HF_HUB_CACHE"] = str(ROOT / ".cache" / "huggingface" / "hub")
    os.environ["TORCH_HOME"] = str(ROOT / ".cache" / "torch")
    os.environ["XDG_CACHE_HOME"] = str(ROOT / ".cache")
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
