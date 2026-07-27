"""CCA-family comparisons for matrices shaped [features, time]."""

from __future__ import annotations

import numpy as np


def _as_2d_float_array(x, name: str) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"{name} must be 2D [features, time], got {x.shape}")
    if x.shape[1] < 2:
        raise ValueError(f"{name} must contain at least 2 time points, got {x.shape}")
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return x


def _center_rows(x: np.ndarray) -> np.ndarray:
    return x - x.mean(axis=1, keepdims=True)


def _inv_sqrtm_psd(matrix: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    values, vectors = np.linalg.eigh(matrix)
    values = np.maximum(values, eps)
    return vectors @ np.diag(1.0 / np.sqrt(values)) @ vectors.T


def cca(x, y, reg: float = 1e-6, max_components: int | None = None) -> dict:
    x = _as_2d_float_array(x, "x")
    y = _as_2d_float_array(y, "y")
    if x.shape[1] != y.shape[1]:
        raise ValueError(f"x and y must share time dimension, got {x.shape} and {y.shape}")
    if reg < 0:
        raise ValueError("reg must be nonnegative")

    x_centered = _center_rows(x)
    y_centered = _center_rows(y)
    nx, n_time = x_centered.shape
    ny = y_centered.shape[0]

    k_max = min(nx, ny, n_time - 1)
    k = k_max if max_components is None else min(max_components, k_max)
    if k < 1:
        raise ValueError("No valid CCA components are available")

    cxx = (x_centered @ x_centered.T) / (n_time - 1) + reg * np.eye(nx)
    cyy = (y_centered @ y_centered.T) / (n_time - 1) + reg * np.eye(ny)
    cxy = (x_centered @ y_centered.T) / (n_time - 1)

    whitened = _inv_sqrtm_psd(cxx) @ cxy @ _inv_sqrtm_psd(cyy)
    u, _, vt = np.linalg.svd(whitened, full_matrices=False)
    x_weights = _inv_sqrtm_psd(cxx) @ u[:, :k]
    y_weights = _inv_sqrtm_psd(cyy) @ vt[:k].T
    x_scores = x_weights.T @ x_centered
    y_scores = y_weights.T @ y_centered

    correlations = np.asarray([
        np.corrcoef(x_scores[i], y_scores[i])[0, 1] for i in range(k)
    ])
    correlations = np.clip(np.nan_to_num(correlations, nan=0.0), -1.0, 1.0)

    return {
        "correlations": correlations,
        "x_scores": x_scores,
        "y_scores": y_scores,
        "x_weights": x_weights,
        "y_weights": y_weights,
        "mean_correlation": float(correlations.mean()),
        "n_components": int(k),
    }


def _pca_reduce(x, variance_threshold: float = 0.99, max_components: int | None = None) -> dict:
    x = _as_2d_float_array(x, "x")
    if not 0 < variance_threshold <= 1:
        raise ValueError("variance_threshold must be in (0, 1]")

    centered = _center_rows(x)
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    explained = singular_values**2
    total = explained.sum()
    if total <= 0:
        raise ValueError("PCA cannot be computed because the representation has zero variance")
    explained_ratio = explained / total
    k = int(np.searchsorted(np.cumsum(explained_ratio), variance_threshold) + 1)
    if max_components is not None:
        k = min(k, max_components)
    k = max(1, k)

    return {
        "reduced": singular_values[:k, None] * vt[:k],
        "explained_variance_ratio": explained_ratio[:k],
        "n_components": int(k),
    }


def svcca(
    x,
    y,
    variance_threshold: float = 0.99,
    max_pca_components: int | None = None,
    max_cca_components: int | None = None,
    reg: float = 1e-6,
) -> dict:
    x_pca = _pca_reduce(x, variance_threshold, max_pca_components)
    y_pca = _pca_reduce(y, variance_threshold, max_pca_components)
    result = cca(x_pca["reduced"], y_pca["reduced"], reg, max_cca_components)
    result.update({
        "x_reduced": x_pca["reduced"],
        "y_reduced": y_pca["reduced"],
        "x_explained_variance_ratio": x_pca["explained_variance_ratio"],
        "y_explained_variance_ratio": y_pca["explained_variance_ratio"],
        "x_pca_components": x_pca["n_components"],
        "y_pca_components": y_pca["n_components"],
        "svcca_score": result["mean_correlation"],
    })
    return result


def pwcca(x, y, reg: float = 1e-6, max_components: int | None = None, weight_side: str = "x") -> dict:
    x = _as_2d_float_array(x, "x")
    y = _as_2d_float_array(y, "y")
    result = cca(x, y, reg, max_components)

    if weight_side == "x":
        original, scores = _center_rows(x), result["x_scores"]
    elif weight_side == "y":
        original, scores = _center_rows(y), result["y_scores"]
    else:
        raise ValueError("weight_side must be 'x' or 'y'")

    norms = np.maximum(np.linalg.norm(scores, axis=1, keepdims=True), 1e-12)
    projections = original @ (scores / norms).T
    weights = np.abs(projections).sum(axis=0)
    weights = np.full_like(weights, 1.0 / len(weights)) if weights.sum() <= 1e-12 else weights / weights.sum()
    result.update({
        "weights": weights,
        "pwcca_score": float(weights @ result["correlations"]),
        "weight_side": weight_side,
    })
    return result
