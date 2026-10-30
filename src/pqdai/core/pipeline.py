"""End-to-end quality judgment pipeline (domain-agnostic).

raw image -> TiledDetector (OBB) -> obb_crop -> brightness correction ->
CropClassifier (CNN + entropy) -> [entropy > alpha: cosine k-NN + hard vote
against the reference bank] -> per-weld verdict -> image-level verdict.

This is the glue between "where is the weld" (detection) and "here is an
already-cropped weld" (classification).
"""

from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

from .tiled_detector import TiledDetector, Detection
from .obb_crop import extract_obb_crop
from .preprocessing import apply_brightness_correction, generate_photometric_tta_views
from ..models.crop_classifier import CropClassifier
from .retrieval import retrieve_topk_neighbors
from .voting import hard_vote
from .uncertainty import compute_all_uncertainties, compute_entropy

DEFAULT_CLASS_NAMES = ['BAD', 'GOOD']
CLASS_NAMES = DEFAULT_CLASS_NAMES  # kept for backwards compatibility


@dataclass
class ObjectVerdict:
    polygon: np.ndarray
    detection_confidence: float
    predicted_class: int
    class_name: str
    entropy: float
    used_retrieval: bool
    cnn_probabilities: np.ndarray
    retrieval_neighbor_labels: Optional[np.ndarray] = None
    retrieval_neighbor_scores: Optional[np.ndarray] = None
    retrieval_vote_confidence: Optional[float] = None
    # Std of per-view entropy across the photometric TTA ensemble (see
    # _judge_crop) -- a high value flags a crop whose classification is
    # fragile under plausible lighting variation, even if this specific
    # exposure happened to look confident. Diagnostic only; does not affect
    # the verdict itself.
    tta_entropy_std: float = 0.0
    # Classifier's penultimate-layer feature vector for this crop (see
    # CropClassifier._register_hook). Not used by any existing caller --
    # exposed so cross-frame consumers (e.g. video tracking) can re-identify
    # the same physical weld across frames via embedding similarity without
    # re-running the classifier.
    embedding: Optional[np.ndarray] = None


@dataclass
class ImageVerdict:
    image_path: str
    objects: List[ObjectVerdict] = field(default_factory=list)
    overall_class_name: str = 'GOOD'  # nothing detected -> nothing to flag as unsafe
    unsafe_class: str = 'BAD'         # class index 0; any object of this class fails the whole item

    @property
    def overall_class(self) -> int:
        return 0 if self.overall_class_name == self.unsafe_class else 1


class QualityPipeline:
    def __init__(
        self,
        detector: TiledDetector,
        classifier: CropClassifier,
        reference_embeddings: np.ndarray,
        reference_labels: np.ndarray,
        alpha: float = 0.1,
        k: int = 3,
        crop_pad_ratio: float = 0.1,
        preprocess_fn=apply_brightness_correction,
        use_retrieval: bool = True,
        class_names=None,
    ):
        # Convention: class index 0 is the UNSAFE class (hard-vote ties resolve toward it, and any
        # object of this class makes the whole item unsafe).
        self.class_names = list(class_names) if class_names else list(DEFAULT_CLASS_NAMES)
        self.detector = detector
        self.classifier = classifier
        self.reference_embeddings = reference_embeddings
        self.reference_labels = reference_labels
        self.alpha = alpha
        self.k = k
        self.crop_pad_ratio = crop_pad_ratio
        self.preprocess_fn = preprocess_fn
        # When False, the verdict is always the (TTA-averaged) CNN argmax --
        # retrieval never runs and never overrides it. Lets you see what the
        # classifier's own evidence says on its own, with BAD+GOOD always
        # summing to 1, instead of a vote fraction that can contradict it.
        self.use_retrieval = use_retrieval

    def _judge_crop(self, crop_bgr: np.ndarray) -> dict:
        corrected = self.preprocess_fn(crop_bgr)

        # Photometric TTA: a single forward pass on one specific exposure can
        # look falsely confident (entropy near 0) for a crop that is actually
        # right on the decision boundary -- averaging over a few plausible
        # lighting variants surfaces that fragility as entropy, so it reaches
        # the alpha gate instead of silently skipping it. Retrieval/embedding
        # still keys off the base (un-jittered) view -- only the entropy used
        # for the gate decision is TTA-averaged.
        # All TTA views share the base crop's exact pixel dimensions (each is
        # a color/gamma jitter of the same array, never resized -- see
        # generate_photometric_tta_views), so they're safe to run as one
        # batched model.predict() call instead of one call per view --
        # CropClassifier.predict_batch_with_uncertainty's same-size
        # precondition holds here by construction. This is the single
        # biggest per-weld inference cost (5 forward passes per weld,
        # previously 5 separate model.predict() calls) -- see
        # the project profiling notes (docs/architecture.md) for the
        # profiling that found individual predict() call overhead, not GPU
        # compute, was this pipeline's actual video-processing bottleneck.
        views = generate_photometric_tta_views(corrected)
        view_results = self.classifier.predict_batch_with_uncertainty(views)
        result = view_results[0]

        per_view_probs = np.stack([r['probabilities'] for r in view_results])
        mean_probs = per_view_probs.mean(axis=0)
        per_view_entropy = np.array([compute_entropy(p) for p in per_view_probs])

        result['probabilities'] = mean_probs
        result['uncertainty'] = compute_all_uncertainties(mean_probs)
        result['tta_entropy_std'] = float(per_view_entropy.std())

        entropy = result['uncertainty']['entropy']
        used_retrieval = (entropy > self.alpha) and self.use_retrieval

        if used_retrieval and result['embedding'] is not None:
            indices, scores = retrieve_topk_neighbors(
                result['embedding'], self.reference_embeddings, self.k, metric='cosine',
            )
            neighbor_labels = self.reference_labels[indices]
            # The CNN's own softmax probabilities (result['probabilities']) are NOT
            # what decides the class below -- retrieval overrides them once entropy
            # crosses alpha, so the vote fraction (not the CNN prob) is the honest
            # "confidence" for this weld's final label.
            predicted_class, vote_confidence = hard_vote(neighbor_labels, return_confidence=True)
            result['retrieval_neighbor_labels'] = neighbor_labels
            result['retrieval_neighbor_scores'] = scores
            result['retrieval_vote_confidence'] = vote_confidence
        else:
            predicted_class = int(np.argmax(mean_probs))
            result['retrieval_neighbor_labels'] = None
            result['retrieval_neighbor_scores'] = None
            result['retrieval_vote_confidence'] = None

        result['final_predicted_class'] = predicted_class
        result['used_retrieval'] = used_retrieval
        return result

    def judge_image(self, image_path: str) -> ImageVerdict:
        detections: List[Detection] = self.detector.detect(image_path)
        image = cv2.imread(image_path)

        verdict = ImageVerdict(image_path=image_path, overall_class_name=self.class_names[-1], unsafe_class=self.class_names[0])
        if image is None or not detections:
            return verdict

        for det in detections:
            crop = extract_obb_crop(image, det.polygon, pad_ratio=self.crop_pad_ratio)
            if crop is None or crop.size == 0:
                continue

            judged = self._judge_crop(crop)
            final_class = judged['final_predicted_class']
            weld_verdict = ObjectVerdict(
                polygon=det.polygon,
                detection_confidence=det.confidence,
                predicted_class=final_class,
                class_name=self.class_names[final_class],
                entropy=judged['uncertainty']['entropy'],
                used_retrieval=judged['used_retrieval'],
                cnn_probabilities=judged['probabilities'],
                retrieval_neighbor_labels=judged['retrieval_neighbor_labels'],
                retrieval_neighbor_scores=judged['retrieval_neighbor_scores'],
                retrieval_vote_confidence=judged['retrieval_vote_confidence'],
                tta_entropy_std=judged['tta_entropy_std'],
                embedding=judged['embedding'],
            )
            verdict.objects.append(weld_verdict)

        # Safety-first aggregation: any BAD weld makes the whole image BAD.
        unsafe, safe = self.class_names[0], self.class_names[-1]
        verdict.unsafe_class = unsafe
        verdict.overall_class_name = unsafe if any(w.class_name == unsafe for w in verdict.objects) else safe
        return verdict
