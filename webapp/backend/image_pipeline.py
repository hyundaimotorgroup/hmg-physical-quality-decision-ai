"""Loads the photo QualityPipeline ONCE at backend startup and reuses it across requests.

The video path (main.py's run_pipeline) shells out to ``python -m pqdai.video.run`` per job, which
reloads the models every time -- fine there since a video job takes minutes anyway.  A single-image
request should return in a couple of seconds, so the pipeline is built once here.

The domain is chosen with the ``PQDAI_DOMAIN`` environment variable (default ``weld``);
``PQDAI_DETECTOR_WEIGHTS`` / ``PQDAI_CLASSIFIER_WEIGHTS`` override the released weights.
"""

import os

from pqdai.core.pipeline import QualityPipeline
from pqdai.core.runtime import build_pipeline

_pipeline: QualityPipeline | None = None


def get_pipeline() -> QualityPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = build_pipeline(
            os.environ.get("PQDAI_DOMAIN", "weld"),
            detector_weights=os.environ.get("PQDAI_DETECTOR_WEIGHTS") or None,
            classifier_weights=os.environ.get("PQDAI_CLASSIFIER_WEIGHTS") or None,
        )
    return _pipeline
