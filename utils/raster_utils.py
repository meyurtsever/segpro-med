"""Utilities for loading conventional 2D image files as one-slice volumes."""

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image, ImageOps


RASTER_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".jfif"}


def find_single_raster_image(directory_path: str) -> Path | None:
    """Return the sole raster file in a folder, or None for other layouts."""
    directory = Path(directory_path)
    if not directory.is_dir():
        return None

    files = sorted(
        (path for path in directory.iterdir() if path.is_file()),
        key=lambda path: path.name.lower(),
    )
    if len(files) == 1 and files[0].suffix.lower() in RASTER_IMAGE_EXTENSIONS:
        return files[0]
    return None


def load_raster_image(file_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load a PNG/JPEG-family image using the volume layout expected by the UI.

    Grayscale files use ``(slices, height, width)`` and color files use
    ``(slices, height, width, channels)``. This retains clinically meaningful
    color while grayscale PNGs preserve their original sample depth (including
    16-bit data).
    """
    path = Path(file_path)
    if path.suffix.lower() not in RASTER_IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported raster image type: {path.suffix.lower()}")

    with Image.open(path) as opened_image:
        image = ImageOps.exif_transpose(opened_image)
        source_format = opened_image.format or path.suffix.lstrip(".").upper()
        source_mode = image.mode

        if source_mode not in {"1", "L", "I", "F", "I;16", "I;16B", "I;16L"}:
            image = image.convert("RGB")

        pixel_array = np.asarray(image)

    if pixel_array.ndim not in {2, 3} or (pixel_array.ndim == 3 and pixel_array.shape[-1] != 3):
        raise ValueError(
            f"Expected a grayscale or RGB image, got shape {pixel_array.shape}"
        )

    height, width = pixel_array.shape[:2]
    volume = pixel_array[np.newaxis, ...]
    metadata = {
        "FileType": "Raster Image",
        "Format": source_format,
        "OriginalMode": source_mode,
        "Width": int(width),
        "Height": int(height),
        "Slices": 1,
    }
    return volume, metadata
