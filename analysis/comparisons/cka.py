"""Centered kernel alignment for matrices shaped [features, time]."""

from __future__ import annotations
import numpy as np


def linear_cka(x, y, debiased: bool = False) -> dict:
    x = np.asarray(x, dtype=np.float64).T  # [time, features]
    y = np.asarray(y, dtype=np.float64).T
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("x and y must be 2D [features, time] with equal time dimensions")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("x and y must contain only finite values")

    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    if debiased:
        # Feature-space unbiased estimator from Kornblith et al.; requires n > 2.
        n = x.shape[0]
        if n <= 2:
            raise ValueError("Debiased CKA requires more than 2 observations")
        dot_xy = np.linalg.norm(x.T @ y, "fro") ** 2
        dot_xx = np.linalg.norm(x.T @ x, "fro") ** 2
        dot_yy = np.linalg.norm(y.T @ y, "fro") ** 2
        sum_x = np.linalg.norm(x, axis=1) ** 2
        sum_y = np.linalg.norm(y, axis=1) ** 2
        hsic = dot_xy - n / (n - 2) * np.dot(sum_x, sum_y) + sum_x.sum() * sum_y.sum() / ((n - 1) * (n - 2))
        norm_x = dot_xx - n / (n - 2) * np.dot(sum_x, sum_x) + sum_x.sum() ** 2 / ((n - 1) * (n - 2))
        norm_y = dot_yy - n / (n - 2) * np.dot(sum_y, sum_y) + sum_y.sum() ** 2 / ((n - 1) * (n - 2))
    else:
        hsic = np.linalg.norm(x.T @ y, "fro") ** 2
        norm_x = np.linalg.norm(x.T @ x, "fro") ** 2
        norm_y = np.linalg.norm(y.T @ y, "fro") ** 2

    denominator = np.sqrt(max(norm_x, 0.0) * max(norm_y, 0.0))
    score = 0.0 if denominator <= 1e-15 else float(hsic / denominator)
    return {
        "cka_score": float(np.clip(score, -1.0, 1.0)),
        "hsic": float(hsic),
        "x_self_hsic": float(norm_x),
        "y_self_hsic": float(norm_y),
        "debiased": bool(debiased),
    }
