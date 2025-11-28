# utils/preprocessing.py
"""
Image preprocessing utilities for CarRacing-v3.

Converts RGB frames to grayscale, resizes, normalizes to [0,1].

Used by all RL agents.
"""

import cv2
import numpy as np


def preprocess_frame(frame, img_h=84, img_w=84):
    """
    Convert RGB frame (H, W, 3) → grayscale resized (img_h x img_w)
    normalized to [0,1].

    Args:
        frame: raw RGB image from the environment
        img_h: output height
        img_w: output width

    Returns:
        np.ndarray with shape (img_h, img_w) dtype float32
    """
    # Convert to grayscale
    img = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # Resize
    img = cv2.resize(img, (img_w, img_h), interpolation=cv2.INTER_AREA)

    # Normalize
    img = img.astype(np.float32) / 255.0

    return img
