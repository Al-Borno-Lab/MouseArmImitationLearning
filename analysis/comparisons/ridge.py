"""Cross-validated ridge mapping from model activations to firing rates."""

from __future__ import annotations
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


def cross_validated_ridge(
    firing_rates,
    activations,
    alpha: float = 1.0,
    n_splits: int = 5,
    shuffle: bool = False,
    random_state: int | None = 0,
    standardize: bool = True,
) -> dict:
    y = np.asarray(firing_rates, dtype=np.float64).T  # [time, neurons]
    x = np.asarray(activations, dtype=np.float64).T   # [time, units]
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("firing_rates and activations must be [features, time] with equal time dimensions")
    if not 2 <= n_splits <= x.shape[0]:
        raise ValueError(f"n_splits must be between 2 and {x.shape[0]}")
    if alpha < 0:
        raise ValueError("alpha must be nonnegative")

    splitter = KFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state if shuffle else None,
    )
    predictions = np.full_like(y, np.nan)
    fold_results = []

    for fold_index, (train_idx, test_idx) in enumerate(splitter.split(x)):
        x_train, x_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if standardize:
            x_scaler = StandardScaler().fit(x_train)
            y_scaler = StandardScaler().fit(y_train)
            x_train_fit = x_scaler.transform(x_train)
            x_test_fit = x_scaler.transform(x_test)
            y_train_fit = y_scaler.transform(y_train)
        else:
            x_scaler = y_scaler = None
            x_train_fit, x_test_fit, y_train_fit = x_train, x_test, y_train

        model = Ridge(alpha=alpha, fit_intercept=True)
        model.fit(x_train_fit, y_train_fit)
        fold_prediction = model.predict(x_test_fit)
        if standardize:
            fold_prediction = y_scaler.inverse_transform(fold_prediction)
        predictions[test_idx] = fold_prediction

        fold_r2 = r2_score(y_test, fold_prediction, multioutput="raw_values")
        fold_correlations = np.asarray([
            np.corrcoef(y_test[:, i], fold_prediction[:, i])[0, 1]
            if np.std(y_test[:, i]) > 0 and np.std(fold_prediction[:, i]) > 0 else 0.0
            for i in range(y.shape[1])
        ])
        fold_results.append({
            "fold": fold_index,
            "train_indices": train_idx,
            "test_indices": test_idx,
            "r2_per_neuron": np.nan_to_num(fold_r2, nan=0.0),
            "correlation_per_neuron": np.nan_to_num(fold_correlations, nan=0.0),
            "mean_r2": float(np.nanmean(fold_r2)),
            "mean_correlation": float(np.nanmean(fold_correlations)),
            "coefficients": model.coef_,
            "intercept": model.intercept_,
        })

    overall_r2 = r2_score(y, predictions, multioutput="raw_values")
    overall_correlations = np.asarray([
        np.corrcoef(y[:, i], predictions[:, i])[0, 1]
        if np.std(y[:, i]) > 0 and np.std(predictions[:, i]) > 0 else 0.0
        for i in range(y.shape[1])
    ])
    residuals = y - predictions

    return {
        "alpha": float(alpha),
        "n_splits": int(n_splits),
        "shuffle": bool(shuffle),
        "random_state": random_state,
        "standardize": bool(standardize),
        "predictions": predictions.T,
        "targets": y.T,
        "residuals": residuals.T,
        "r2_per_neuron": np.nan_to_num(overall_r2, nan=0.0),
        "correlation_per_neuron": np.nan_to_num(overall_correlations, nan=0.0),
        "mean_r2": float(np.nanmean(overall_r2)),
        "median_r2": float(np.nanmedian(overall_r2)),
        "mean_correlation": float(np.nanmean(overall_correlations)),
        "folds": fold_results,
    }
