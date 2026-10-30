"""Weld crop preprocessing.

Adaptive LAB-space gamma correction and photometric TTA views.
"""

from typing import List

import cv2
import numpy as np


def apply_brightness_correction(image: np.ndarray) -> np.ndarray:
    """Adaptive gamma correction in LAB space.

    Dark images (mean L < 100) are brightened (gamma=1.5), bright images
    (mean L > 150) are darkened (gamma=0.7), otherwise left unchanged.

    Args:
        image: BGR image (as read by cv2.imread).

    Returns:
        Gamma-corrected BGR image.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    mean_l = np.mean(l_channel)
    if mean_l < 100:
        gamma = 1.5
    elif mean_l > 150:
        gamma = 0.7
    else:
        gamma = 1.0

    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    l_corrected = cv2.LUT(l_channel, table)

    lab_corrected = cv2.merge([l_corrected, a, b])
    return cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2BGR)


def apply_clahe_correction(image: np.ndarray, clip_limit: float = 3.0, tile_grid_size=(8, 8)) -> np.ndarray:
    """Local contrast enhancement (CLAHE) on the L channel, as an alternative
    to the global gamma correction above.

    Global gamma correction rescales the whole crop by one factor derived
    from its mean brightness, so a weld seam sitting in a shadowed corner of
    an otherwise well-lit crop gets little benefit. CLAHE instead equalizes
    contrast in local tiles, which can reveal texture in shadowed regions
    that a single global gamma factor would leave under-exposed.

    Not used by the default pipeline (pqdai.core.pipeline still uses
    apply_brightness_correction, which is what the classifier was trained and
    validated with). Provided to A/B test whether it reduces the dark-weld
    error rate in practice.

    Args:
        image: BGR image (as read by cv2.imread).
        clip_limit: CLAHE contrast clipping threshold (higher = more contrast).
        tile_grid_size: number of tiles per axis for local histogram equalization.

    Returns:
        Contrast-enhanced BGR image.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_corrected = clahe.apply(l_channel)

    lab_corrected = cv2.merge([l_corrected, a, b])
    return cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2BGR)


def generate_photometric_tta_views(corrected_bgr: np.ndarray) -> List[np.ndarray]:
    """Generate small photometric-jitter variants of an already
    brightness-corrected crop, for averaging classifier predictions across
    plausible exposure variation instead of trusting a single forward pass.

    The gamma/brightness range here (0.85/1.15, +-15) was found by direct
    experiment (2026-08-20) to swing a real borderline weld's prediction
    across the full GOOD<->BAD range while leaving a genuinely confident
    weld's prediction untouched -- i.e. it's a real fragility probe, not an
    arbitrary augmentation. Unlike pad_ratio (crop boundary) jitter, which
    was tested on the same pair of images and changed nothing.
    """
    views = [corrected_bgr]
    for gamma in (0.85, 1.15):
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        views.append(cv2.LUT(corrected_bgr, table))
    for delta in (-15, 15):
        views.append(np.clip(corrected_bgr.astype(np.int16) + delta, 0, 255).astype(np.uint8))
    return views
