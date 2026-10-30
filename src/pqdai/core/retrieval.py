"""K-NN similarity search over embeddings.

Cosine / L2 top-k retrieval used by the selective-retrieval re-check.
"""

import numpy as np
from typing import Tuple, Literal


def cosine_similarity_batch(
    query_embedding: np.ndarray,
    reference_embeddings: np.ndarray,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """Cosine similarity between one query and N references. Higher = more similar."""
    if query_embedding.ndim > 1:
        query_embedding = query_embedding.flatten()

    query_norm = np.linalg.norm(query_embedding)
    query_normalized = query_embedding / (query_norm + epsilon)

    ref_norms = np.linalg.norm(reference_embeddings, axis=1, keepdims=True)
    ref_normalized = reference_embeddings / (ref_norms + epsilon)

    return np.dot(ref_normalized, query_normalized)


def l2_distance_batch(
    query_embedding: np.ndarray,
    reference_embeddings: np.ndarray,
) -> np.ndarray:
    """L2 distance between one query and N references. Lower = more similar."""
    if query_embedding.ndim > 1:
        query_embedding = query_embedding.flatten()
    diff = query_embedding[np.newaxis, :] - reference_embeddings
    return np.linalg.norm(diff, axis=1)


def retrieve_topk_neighbors(
    query_embedding: np.ndarray,
    reference_embeddings: np.ndarray,
    k: int,
    metric: Literal['cosine', 'l2'] = 'cosine',
    exclude_index: int = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (indices, scores) of the top-k nearest reference embeddings."""
    n_references = len(reference_embeddings)
    if k > n_references:
        k = n_references

    if metric == 'cosine':
        scores = cosine_similarity_batch(query_embedding, reference_embeddings)
        if exclude_index is not None and 0 <= exclude_index < len(scores):
            scores[exclude_index] = -np.inf
        topk_indices = np.argsort(scores)[::-1][:k]
        topk_scores = scores[topk_indices]
    elif metric == 'l2':
        scores = l2_distance_batch(query_embedding, reference_embeddings)
        if exclude_index is not None and 0 <= exclude_index < len(scores):
            scores[exclude_index] = np.inf
        topk_indices = np.argsort(scores)[:k]
        topk_scores = scores[topk_indices]
    else:
        raise ValueError(f"Unknown metric: {metric}. Use 'cosine' or 'l2'")

    return topk_indices, topk_scores
