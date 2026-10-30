"""Extract an upright rectangular crop from an oriented bounding box (OBB).

Glue between the detector (locates a weld) and the classifier (judges a
pre-cropped weld).
"""

import cv2
import numpy as np


def normalized_polygon_to_pixels(polygon_norm, img_w: int, img_h: int) -> np.ndarray:
    """Convert a LABEL-format polygon (x1 y1 x2 y2 x3 y3 x4 y4, 0-1 normalized)
    into (4, 2) pixel coordinates."""
    pts = np.array(polygon_norm, dtype=np.float32).reshape(4, 2)
    pts[:, 0] *= img_w
    pts[:, 1] *= img_h
    return pts


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 arbitrary corner points as (top-left, top-right, bottom-right, bottom-left)."""
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()

    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]      # top-left: smallest x+y
    ordered[2] = pts[np.argmax(s)]      # bottom-right: largest x+y
    ordered[1] = pts[np.argmin(diff)]   # top-right: smallest y-x
    ordered[3] = pts[np.argmax(diff)]   # bottom-left: largest y-x
    return ordered


def extract_obb_crop(image: np.ndarray, polygon_px: np.ndarray, pad_ratio: float = 0.0):
    """Perspective-warp the quadrilateral region defined by `polygon_px` into
    an upright rectangle sized to its actual side lengths.

    Args:
        image: source image (BGR, as read by cv2.imread).
        polygon_px: (4, 2) array of pixel corner coordinates, any order.
        pad_ratio: expand the output rectangle by this fraction (0.0 = no padding).

    Returns:
        Cropped BGR image, or None if the box degenerates to zero area.
    """
    rect = order_points(polygon_px)
    tl, tr, br, bl = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(round(width_a)), int(round(width_b)))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(round(height_a)), int(round(height_b)))

    if pad_ratio > 0:
        max_width = int(round(max_width * (1 + pad_ratio)))
        max_height = int(round(max_height * (1 + pad_ratio)))

    if max_width <= 0 or max_height <= 0:
        return None

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1],
    ], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, matrix, (max_width, max_height))
