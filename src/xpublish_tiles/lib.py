"""Library utility functions for xpublish-tiles."""

import io

import numpy as np
from PIL import Image


def check_transparent_pixels(image_bytes):
    """Check the percentage of transparent pixels in a PNG image."""
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode != "RGBA":
        img = img.convert("RGBA")

    arr = np.array(img)
    transparent_mask = arr[:, :, 3] == 0
    transparent_count = np.sum(transparent_mask)
    total_pixels = arr.shape[0] * arr.shape[1]

    return (transparent_count / total_pixels) * 100
