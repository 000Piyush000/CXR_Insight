"""
Central configuration for CXR Insight.

Everything here is read from environment variables (loaded from a local
.env file if present) so the same code runs in demo mode on a laptop with
no GPU, or fully loaded with real Swin-T5 / LLaMA-3.1 weights on a
workstation with CUDA.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def _bool_env(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


class Config:
    # --- Paths -------------------------------------------------------
    BASE_DIR = BASE_DIR
    UPLOAD_FOLDER = BASE_DIR / "uploads"
    REPORTS_FOLDER = BASE_DIR / "reports"
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "dcm"}
    MAX_CONTENT_LENGTH = 25 * 1024 * 1024  # 25 MB uploads

    # --- Flask ---------------------------------------------------------
    SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "change-me")
    PORT = int(os.getenv("FLASK_PORT", "5000"))
    DEBUG = _bool_env("FLASK_DEBUG", True)

    # --- Models ----------------------------------------------------------
    HF_TOKEN = os.getenv("HF_TOKEN") or None
    SWIN_T5_CHECKPOINT = os.getenv("SWIN_T5_CHECKPOINT") or None
    SWIN_ENCODER_ID = os.getenv(
        "SWIN_ENCODER_ID", "microsoft/swin-base-patch4-window7-224-in22k"
    )
    T5_DECODER_ID = os.getenv("T5_DECODER_ID", "t5-base")
    LLM_MODEL_ID = os.getenv("LLM_MODEL_ID", "meta-llama/Meta-Llama-3.1-8B-Instruct")
    DEVICE = os.getenv("DEVICE") or None  # None => auto-detect in model wrappers
    FORCE_DEMO_MODE = _bool_env("FORCE_DEMO_MODE", False)

    # Generation params
    MAX_REPORT_TOKENS = int(os.getenv("MAX_REPORT_TOKENS", "220"))
    MAX_CHAT_TOKENS = int(os.getenv("MAX_CHAT_TOKENS", "300"))


Config.UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
Config.REPORTS_FOLDER.mkdir(parents=True, exist_ok=True)
