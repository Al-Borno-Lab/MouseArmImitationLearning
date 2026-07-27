"""Representational similarity analysis across matched time points."""

from __future__ import annotations
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr


def rsa(x, y, distance_metric: str = "correlation", correlation_method: str = "spearman") -> dict:
    x = np.asarray(x, dtype=np.float64).T  # observations=time
    y = np.asarray(y, dtype=np.float64).T
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("x and y must be 2D [features, time] with equal time dimensions")
    if x.shape[0] < 3:
        raise ValueError("RSA requires at least 3 time points")

    x_distances = pdist(x, metric=distance_metric)
    y_distances = pdist(y, metric=distance_metric)
    if not np.all(np.isfinite(x_distances)) or not np.all(np.isfinite(y_distances)):
        raise ValueError("RSA produced non-finite distances; check for constant observations/features")

    if correlation_method == "spearman":
        statistic, pvalue = spearmanr(x_distances, y_distances)
    elif correlation_method == "pearson":
        statistic, pvalue = pearsonr(x_distances, y_distances)
    else:
        raise ValueError("correlation_method must be 'spearman' or 'pearson'")

    return {
        "rsa_score": float(statistic),
        "p_value": float(pvalue),
        "distance_metric": distance_metric,
        "correlation_method": correlation_method,
        "x_rdm_condensed": x_distances,
        "y_rdm_condensed": y_distances,
        "x_rdm": squareform(x_distances),
        "y_rdm": squareform(y_distances),
    }
