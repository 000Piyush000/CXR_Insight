"""Small helpers for validating and preparing uploaded X-ray images."""

from __future__ import annotations

import uuid
from pathlib import Path

from PIL import Image, ImageOps
from werkzeug.utils import secure_filename

from config import Config


def allowed_file(filename: str) -> bool:
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in Config.ALLOWED_EXTENSIONS
    )


def save_upload(file_storage) -> Path:
    """Persist an uploaded werkzeug FileStorage to disk with a safe, unique name."""
    original_name = secure_filename(file_storage.filename)
    unique_prefix = uuid.uuid4().hex[:8]
    safe_name = f"{unique_prefix}_{original_name}"
    dest = Config.UPLOAD_FOLDER / safe_name
    file_storage.save(dest)
    return dest


def load_rgb_image(path: Path) -> Image.Image:
    """Load an image file as a normalized RGB PIL Image, correcting EXIF orientation."""
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")
