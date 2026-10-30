"""Draws a single-image ImageVerdict onto its source image.

Same drawing logic as the video overlays, so the webapp's image-upload path
has a visual result.
"""

import cv2
import numpy as np

COLOR_GOOD = (0, 200, 0)     # BGR green, matches pqdai/video/video_visualize.py
COLOR_BAD = (0, 0, 255)      # BGR red


def draw_verdict(image: np.ndarray, verdict) -> np.ndarray:
    canvas = image.copy()
    for weld in verdict.objects:
        color = COLOR_BAD if weld.class_name == verdict.unsafe_class else COLOR_GOOD
        pts = weld.polygon.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=3)

        if weld.used_retrieval:
            label = f"{weld.class_name} {weld.retrieval_vote_confidence:.2f} [retrieved]"
        else:
            class_prob = weld.cnn_probabilities[0] if weld.class_name == verdict.unsafe_class else weld.cnn_probabilities[-1]
            label = f"{weld.class_name} {class_prob:.2f}"
        origin = tuple(pts[0, 0])
        cv2.putText(canvas, label, (origin[0], max(origin[1] - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    banner = f"OVERALL: {verdict.overall_class_name} ({len(verdict.objects)} object(s))"
    banner_color = COLOR_BAD if verdict.overall_class_name == verdict.unsafe_class else COLOR_GOOD
    cv2.putText(canvas, banner, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, banner_color, 2, cv2.LINE_AA)
    return canvas
