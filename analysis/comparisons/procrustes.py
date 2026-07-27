"""PCA followed by orthogonal Procrustes alignment of matched trajectories.

The filename intentionally follows the requested spelling: proscrustes.py.
"""

from __future__ import annotations
import numpy as np
from scipy.linalg import orthogonal_procrustes


def _pca_scores(representation: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray]:
    observations = np.asarray(representation, dtype=np.float64).T  # [time, features]
    observations = observations - observations.mean(axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(observations, full_matrices=False)
    k = min(n_components, len(singular_values))
    scores = observations @ vt[:k].T
    explained = singular_values[:k] ** 2
    explained_ratio = explained / max(np.sum(singular_values**2), 1e-15)
    return scores, explained_ratio


def procrustes(x, y, n_components: int = 5, allow_scaling: bool = True) -> dict:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("x and y must be 2D [features, time] with equal time dimensions")
    max_valid = min(x.shape[0], y.shape[0], x.shape[1] - 1)
    k = min(n_components, max_valid)
    if k < 1:
        raise ValueError("No valid Procrustes dimensions are available")

    x_scores, x_explained = _pca_scores(x, k)
    y_scores, y_explained = _pca_scores(y, k)
    x_scores -= x_scores.mean(axis=0, keepdims=True)
    y_scores -= y_scores.mean(axis=0, keepdims=True)

    x_norm = np.linalg.norm(x_scores)
    y_norm = np.linalg.norm(y_scores)
    if x_norm <= 1e-15 or y_norm <= 1e-15:
        raise ValueError("Procrustes cannot align a zero-variance representation")

    # Normalize both trajectories so residuals are comparable across datasets.
    x_normalized = x_scores / x_norm
    y_normalized = y_scores / y_norm
    rotation, scale = orthogonal_procrustes(x_normalized, y_normalized)
    fitted_scale = float(scale) if allow_scaling else 1.0
    x_aligned = fitted_scale * (x_normalized @ rotation)
    residual = x_aligned - y_normalized
    disparity = float(np.sum(residual**2))
    similarity = float(1.0 - disparity / 2.0)

    return {
        "procrustes_similarity": similarity,
        "disparity": disparity,
        "n_components": int(k),
        "allow_scaling": bool(allow_scaling),
        "fitted_scale": fitted_scale,
        "rotation": rotation,
        "x_pca_scores": x_scores,
        "y_pca_scores": y_scores,
        "x_aligned": x_aligned,
        "y_normalized": y_normalized,
        "residuals": residual,
        "x_explained_variance_ratio": x_explained,
        "y_explained_variance_ratio": y_explained,
    }
