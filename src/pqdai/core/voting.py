"""Hard/soft voting for aggregating retrieved neighbor labels."""

import numpy as np
from typing import Dict, Optional
from scipy import stats

CLASS_NAMES = ['BAD', 'GOOD']


def hard_vote(retrieved_labels: np.ndarray, return_confidence: bool = False):
    """Majority vote over neighbor labels. Ties favor label 0 (BAD) for safety."""
    if len(retrieved_labels) == 0:
        raise ValueError("Cannot vote with empty labels")

    mode_result = stats.mode(retrieved_labels, keepdims=True)
    voted_label = int(mode_result.mode[0])

    if return_confidence:
        vote_count = np.sum(retrieved_labels == voted_label)
        confidence = vote_count / len(retrieved_labels)
        return voted_label, confidence

    return voted_label


def soft_vote(
    retrieved_labels: np.ndarray,
    distances: Optional[np.ndarray] = None,
    temperature: float = 1.0,
) -> Dict[str, float]:
    """Inverse-distance-weighted (or uniform) class probability vote."""
    if len(retrieved_labels) == 0:
        raise ValueError("Cannot vote with empty labels")

    if distances is not None:
        epsilon = 1e-8
        weights = 1.0 / (distances + epsilon)
        if temperature != 1.0:
            weights = weights ** (1.0 / temperature)
        weights = weights / np.sum(weights)
    else:
        weights = np.ones(len(retrieved_labels)) / len(retrieved_labels)

    probs = np.zeros(2)
    for label, weight in zip(retrieved_labels, weights):
        probs[int(label)] += weight

    predicted_class = int(np.argmax(probs))
    return {
        'probabilities': probs,
        'predicted_class': predicted_class,
        'confidence': float(probs[predicted_class]),
    }


def analyze_vote_distribution(retrieved_labels: np.ndarray) -> Dict:
    unique, counts = np.unique(retrieved_labels, return_counts=True)
    distribution = {CLASS_NAMES[int(label)]: int(count) for label, count in zip(unique, counts)}
    majority_label = int(unique[np.argmax(counts)])
    majority_count = int(counts[np.argmax(counts)])
    return {
        'distribution': distribution,
        'majority_label': majority_label,
        'majority_class': CLASS_NAMES[majority_label],
        'majority_count': majority_count,
        'majority_ratio': majority_count / len(retrieved_labels),
        'total_votes': len(retrieved_labels),
    }
