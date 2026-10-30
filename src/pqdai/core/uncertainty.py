"""Uncertainty metrics for classifier probabilities.

Entropy / margin / max-prob metrics used to gate the retrieval re-check.
"""

import numpy as np
from typing import Dict


def compute_entropy(probs: np.ndarray, epsilon: float = 1e-10) -> float:
    """Shannon entropy of class probabilities. Binary max entropy = log(2) ~ 0.693."""
    probs = np.clip(probs, epsilon, 1.0)
    return float(-np.sum(probs * np.log(probs)))


def compute_margin(probs: np.ndarray) -> float:
    """Gap between top-2 probabilities. Lower = more uncertain."""
    if len(probs) < 2:
        return 0.0
    sorted_probs = np.sort(probs)[::-1]
    return float(sorted_probs[0] - sorted_probs[1])


def compute_max_prob(probs: np.ndarray) -> float:
    return float(np.max(probs))


def compute_all_uncertainties(probs: np.ndarray) -> Dict[str, float]:
    return {
        'entropy': compute_entropy(probs),
        'margin': compute_margin(probs),
        'max_prob': compute_max_prob(probs),
    }


def is_ambiguous(entropy: float, alpha: float) -> bool:
    """Samples with entropy > alpha are routed to selective retrieval."""
    return entropy > alpha
