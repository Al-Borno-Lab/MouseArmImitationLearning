"""Run all neural/DNN comparison metrics and save one JSON results file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analysis.comparisons.cca import cca, pwcca, svcca
from analysis.comparisons.cka import linear_cka
from analysis.comparisons.procrustes import procrustes
from analysis.comparisons.ridge import cross_validated_ridge
from analysis.comparisons.rsa import rsa


def load_json_matrix(path: str | Path, name: str) -> np.ndarray:
    with open(path, "r", encoding="utf-8") as file:
        matrix = np.asarray(json.load(file), dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a JSON matrix [features, time], got {matrix.shape}")
    if matrix.shape[1] < 3:
        raise ValueError(f"{name} must contain at least 3 time points, got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return matrix


def find_json_files(folder: str | Path) -> list[Path]:
    """Recursively find non-hidden JSON files in a folder."""
    folder = Path(folder)
    files = [
        path
        for path in folder.rglob("*.json")
        if path.is_file()
        and not any(
            part.startswith(".")
            for part in path.relative_to(folder).parts
        )
    ]
    return sorted(files, key=lambda path: path.as_posix())


def _build_filename_index(files: list[Path]) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for path in files:
        index.setdefault(path.name, []).append(path)
    return index


def match_folder_inputs(
    firing_rates_dir: str | Path,
    activations_dir: str | Path,
) -> list[tuple[Path, Path, Path]]:
    """Match activation JSON files to firing-rate JSON files."""
    firing_rates_dir = Path(firing_rates_dir)
    activations_dir = Path(activations_dir)

    firing_files = find_json_files(firing_rates_dir)
    activation_files = find_json_files(activations_dir)

    if not activation_files:
        raise ValueError(f"No JSON files found in activations folder: {activations_dir}")
    if not firing_files:
        raise ValueError(f"No JSON files found in firing-rates folder: {firing_rates_dir}")

    firing_by_name = _build_filename_index(firing_files)
    matches: list[tuple[Path, Path, Path]] = []
    missing: list[str] = []

    for activation_path in activation_files:
        relative_path = activation_path.relative_to(activations_dir)
        same_relative_path = firing_rates_dir / relative_path

        if same_relative_path.is_file():
            firing_path = same_relative_path
        else:
            candidates = firing_by_name.get(activation_path.name, [])
            if len(candidates) == 1:
                firing_path = candidates[0]
            elif len(candidates) == 0:
                missing.append(relative_path.as_posix())
                continue
            else:
                candidate_text = ", ".join(
                    candidate.relative_to(firing_rates_dir).as_posix()
                    for candidate in candidates
                )
                raise ValueError(
                    f"Ambiguous firing-rate match for activation file "
                    f"{relative_path.as_posix()!r}. Multiple firing-rate files share "
                    f"the filename {activation_path.name!r}: {candidate_text}"
                )

        matches.append((relative_path, firing_path, activation_path))

    if missing:
        preview = ", ".join(missing[:10])
        suffix = "" if len(missing) <= 10 else f", ... ({len(missing)} missing total)"
        raise ValueError(
            "No corresponding firing-rate JSON was found for these activation files: "
            f"{preview}{suffix}"
        )

    return matches


def load_input_pair(
    firing_rates_input: str | Path,
    activations_input: str | Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load one JSON pair or matched folders and concatenate along time."""
    firing_rates_input = Path(firing_rates_input)
    activations_input = Path(activations_input)

    if firing_rates_input.is_file() and activations_input.is_file():
        firing_rates = load_json_matrix(firing_rates_input, "firing rates")
        activations = load_json_matrix(activations_input, "activations")
        if firing_rates.shape[1] != activations.shape[1]:
            raise ValueError(
                "Time dimensions differ: "
                f"firing rates {firing_rates.shape}, activations {activations.shape}"
            )
        return firing_rates, activations, {
            "mode": "single_file",
            "matched_file_count": 1,
            "matched_files": [{
                "name": activations_input.name,
                "firing_rates_path": str(firing_rates_input),
                "activations_path": str(activations_input),
                "time_start": 0,
                "time_stop": int(firing_rates.shape[1]),
                "time_points": int(firing_rates.shape[1]),
            }],
        }

    if firing_rates_input.is_dir() and activations_input.is_dir():
        matches = match_folder_inputs(firing_rates_input, activations_input)
        firing_matrices: list[np.ndarray] = []
        activation_matrices: list[np.ndarray] = []
        matched_files: list[dict[str, Any]] = []
        expected_firing_features: int | None = None
        expected_activation_features: int | None = None
        time_cursor = 0

        for relative_path, firing_path, activation_path in matches:
            firing_matrix = load_json_matrix(
                firing_path,
                f"firing rates ({relative_path.as_posix()})",
            )
            activation_matrix = load_json_matrix(
                activation_path,
                f"activations ({relative_path.as_posix()})",
            )
            if firing_matrix.shape[1] != activation_matrix.shape[1]:
                raise ValueError(
                    f"Time dimensions differ for {relative_path.as_posix()}: "
                    f"firing rates {firing_matrix.shape}, activations {activation_matrix.shape}"
                )
            if expected_firing_features is None:
                expected_firing_features = firing_matrix.shape[0]
            elif firing_matrix.shape[0] != expected_firing_features:
                raise ValueError(
                    f"Firing-rate feature count changed across files. Expected "
                    f"{expected_firing_features}, got {firing_matrix.shape[0]} in {firing_path}"
                )
            if expected_activation_features is None:
                expected_activation_features = activation_matrix.shape[0]
            elif activation_matrix.shape[0] != expected_activation_features:
                raise ValueError(
                    f"Activation feature count changed across files. Expected "
                    f"{expected_activation_features}, got {activation_matrix.shape[0]} "
                    f"in {activation_path}"
                )

            time_points = firing_matrix.shape[1]
            firing_matrices.append(firing_matrix)
            activation_matrices.append(activation_matrix)
            matched_files.append({
                "name": relative_path.as_posix(),
                "firing_rates_path": str(firing_path),
                "activations_path": str(activation_path),
                "time_start": int(time_cursor),
                "time_stop": int(time_cursor + time_points),
                "time_points": int(time_points),
            })
            time_cursor += time_points

        return (
            np.concatenate(firing_matrices, axis=1),
            np.concatenate(activation_matrices, axis=1),
            {
                "mode": "folder",
                "matched_file_count": len(matches),
                "matched_files": matched_files,
            },
        )

    if not firing_rates_input.exists():
        raise FileNotFoundError(f"Firing-rates input does not exist: {firing_rates_input}")
    if not activations_input.exists():
        raise FileNotFoundError(f"Activations input does not exist: {activations_input}")
    raise ValueError(
        "The two inputs must both be JSON files or both be folders. "
        f"Got firing_rates={firing_rates_input} and activations={activations_input}"
    )


def json_safe(value: Any, decimals: int = 2) -> Any:
    """Convert NumPy values to JSON-safe objects and round floating outputs."""
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.floating):
            return np.round(value, decimals=decimals).tolist()
        return value.tolist()
    if isinstance(value, np.floating):
        return round(float(value), decimals)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, float):
        return round(value, decimals)
    if isinstance(value, dict):
        return {key: json_safe(item, decimals=decimals) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item, decimals=decimals) for item in value]
    return value


def make_regularization_grid(
    min_reg: float = 1e-8,
    max_reg: float = 1e-2,
    num_values: int = 7,
) -> list[float]:
    """Create an inclusive logarithmically spaced CCA regularization grid."""
    if min_reg <= 0:
        raise ValueError(f"min_reg must be positive, got {min_reg}")
    if max_reg <= 0:
        raise ValueError(f"max_reg must be positive, got {max_reg}")
    if min_reg > max_reg:
        raise ValueError(f"min_reg ({min_reg}) cannot exceed max_reg ({max_reg})")
    if num_values < 1:
        raise ValueError(f"num_values must be at least 1, got {num_values}")
    if num_values == 1:
        return [float(min_reg)]
    return [
        float(value)
        for value in np.logspace(
            np.log10(min_reg),
            np.log10(max_reg),
            num=num_values,
        )
    ]


def format_reg_key(reg: float) -> str:
    """Return a stable, readable dictionary key for a regularization value."""
    return f"{reg:.6e}"


def print_table(title: str, headers: list[str], rows: list[list[Any]]) -> None:
    """Print a compact plain-text table without an external dependency."""
    formatted_rows = [[str(item) for item in row] for row in rows]
    widths = [len(header) for header in headers]

    for row in formatted_rows:
        for index, item in enumerate(row):
            widths[index] = max(widths[index], len(item))

    print()
    print(title)
    print(" | ".join(header.ljust(widths[i]) for i, header in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))

    for row in formatted_rows:
        print(" | ".join(item.ljust(widths[i]) for i, item in enumerate(row)))


def make_component_grid(max_components: int, min_components: int = 5, num_values: int = 20) -> list[int]:
    """
    Create up to ``num_values`` unique component counts spanning
    ``min_components`` through ``max_components`` inclusively.

    If the valid range contains fewer than ``num_values`` integers, every
    integer in the range is returned.
    """
    if max_components < 1:
        raise ValueError(f"max_components must be at least 1, got {max_components}")

    start = min(min_components, max_components)

    if max_components - start + 1 <= num_values:
        return list(range(start, max_components + 1))

    values = np.linspace(start, max_components, num=num_values)
    grid = sorted({int(round(value)) for value in values})

    # Rounding can occasionally produce fewer than num_values unique entries.
    # Fill any gaps deterministically while preserving the requested endpoints.
    if len(grid) < num_values:
        for value in range(start, max_components + 1):
            if value not in grid:
                grid.append(value)
                if len(grid) == num_values:
                    break
        grid.sort()

    return grid



def _mean_neuronwise_correlation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Mean Pearson correlation across neural output features."""
    correlations = []

    for neuron_index in range(y_true.shape[1]):
        true_values = y_true[:, neuron_index]
        predicted_values = y_pred[:, neuron_index]

        true_std = np.std(true_values)
        predicted_std = np.std(predicted_values)

        if true_std == 0.0 or predicted_std == 0.0:
            continue

        correlation = np.corrcoef(true_values, predicted_values)[0, 1]
        if np.isfinite(correlation):
            correlations.append(float(correlation))

    if not correlations:
        return float("nan")

    return float(np.mean(correlations))


def leave_one_reach_out_ridge(
    firing_rates: np.ndarray,
    activations: np.ndarray,
    matched_files: list[dict[str, Any]],
    alpha_grid: list[float],
    inner_folds: int,
    selection_metric: str,
    standardize: bool,
) -> dict[str, Any]:
    """
    Nested reach-level cross-validation.

    Outer loop:
        Hold out one complete reach for final testing.

    Inner loop:
        Use grouped cross-validation across the remaining reaches to select alpha.
        No time points from an inner validation reach appear in its training fold.
    """
    if len(matched_files) < 3:
        raise ValueError(
            "Nested leave-one-reach-out ridge requires at least 3 matched reaches"
        )
    if selection_metric not in {"r2", "correlation"}:
        raise ValueError(
            f"selection_metric must be 'r2' or 'correlation', got {selection_metric}"
        )
    if inner_folds < 2:
        raise ValueError(f"inner_folds must be at least 2, got {inner_folds}")
    if not alpha_grid:
        raise ValueError("alpha_grid cannot be empty")

    x_all = activations.T
    y_all = firing_rates.T
    per_reach = []

    reach_ranges = [
        (
            int(file_info["time_start"]),
            int(file_info["time_stop"]),
        )
        for file_info in matched_files
    ]

    for held_out_index, file_info in enumerate(matched_files):
        test_start, test_stop = reach_ranges[held_out_index]

        outer_test_mask = np.zeros(x_all.shape[0], dtype=bool)
        outer_test_mask[test_start:test_stop] = True
        outer_train_mask = ~outer_test_mask

        x_outer_train = x_all[outer_train_mask]
        y_outer_train = y_all[outer_train_mask]
        x_outer_test = x_all[outer_test_mask]
        y_outer_test = y_all[outer_test_mask]

        # Build reach-group labels for the outer-training samples.
        outer_train_groups = []
        for reach_index, (start, stop) in enumerate(reach_ranges):
            if reach_index == held_out_index:
                continue
            outer_train_groups.extend([reach_index] * (stop - start))
        outer_train_groups = np.asarray(outer_train_groups, dtype=np.int64)

        unique_training_reaches = np.unique(outer_train_groups)
        actual_inner_folds = min(inner_folds, len(unique_training_reaches))
        if actual_inner_folds < 2:
            raise ValueError(
                "Not enough training reaches for grouped inner cross-validation"
            )

        group_cv = GroupKFold(n_splits=actual_inner_folds)
        alpha_validation = []

        for alpha in alpha_grid:
            fold_r2 = []
            fold_correlation = []

            for inner_train_indices, inner_validation_indices in group_cv.split(
                x_outer_train,
                y_outer_train,
                groups=outer_train_groups,
            ):
                x_inner_train = x_outer_train[inner_train_indices]
                y_inner_train = y_outer_train[inner_train_indices]
                x_inner_validation = x_outer_train[inner_validation_indices]
                y_inner_validation = y_outer_train[inner_validation_indices]

                if standardize:
                    inner_model = make_pipeline(
                        StandardScaler(),
                        Ridge(alpha=alpha),
                    )
                else:
                    inner_model = Ridge(alpha=alpha)

                inner_model.fit(x_inner_train, y_inner_train)
                inner_prediction = inner_model.predict(x_inner_validation)

                fold_r2.append(
                    float(
                        r2_score(
                            y_inner_validation,
                            inner_prediction,
                            multioutput="variance_weighted",
                        )
                    )
                )
                fold_correlation.append(
                    _mean_neuronwise_correlation(
                        y_inner_validation,
                        inner_prediction,
                    )
                )

            finite_correlations = [
                value for value in fold_correlation if np.isfinite(value)
            ]

            alpha_validation.append(
                {
                    "alpha": float(alpha),
                    "mean_r2": float(np.mean(fold_r2)),
                    "mean_correlation": (
                        float(np.mean(finite_correlations))
                        if finite_correlations
                        else float("nan")
                    ),
                }
            )

        if selection_metric == "r2":
            best_alpha_result = max(
                alpha_validation,
                key=lambda result: result["mean_r2"],
            )
        else:
            valid_results = [
                result
                for result in alpha_validation
                if np.isfinite(result["mean_correlation"])
            ]
            if not valid_results:
                raise ValueError(
                    "All inner-validation correlations were non-finite"
                )
            best_alpha_result = max(
                valid_results,
                key=lambda result: result["mean_correlation"],
            )

        selected_alpha = float(best_alpha_result["alpha"])

        if standardize:
            final_model = make_pipeline(
                StandardScaler(),
                Ridge(alpha=selected_alpha),
            )
        else:
            final_model = Ridge(alpha=selected_alpha)

        final_model.fit(x_outer_train, y_outer_train)
        outer_prediction = final_model.predict(x_outer_test)

        reach_r2 = float(
            r2_score(
                y_outer_test,
                outer_prediction,
                multioutput="variance_weighted",
            )
        )
        reach_correlation = _mean_neuronwise_correlation(
            y_outer_test,
            outer_prediction,
        )

        per_reach.append(
            {
                "held_out_reach": file_info.get("name", str(held_out_index)),
                "time_points": int(test_stop - test_start),
                "selected_alpha": selected_alpha,
                "inner_validation_r2": best_alpha_result["mean_r2"],
                "inner_validation_correlation": best_alpha_result[
                    "mean_correlation"
                ],
                "r2": reach_r2,
                "mean_correlation": reach_correlation,
            }
        )

    r2_values = np.asarray(
        [result["r2"] for result in per_reach],
        dtype=np.float64,
    )
    correlation_values = np.asarray(
        [result["mean_correlation"] for result in per_reach],
        dtype=np.float64,
    )
    selected_alphas = np.asarray(
        [result["selected_alpha"] for result in per_reach],
        dtype=np.float64,
    )

    finite_correlations = correlation_values[np.isfinite(correlation_values)]

    return {
        "method": "nested_leave_one_reach_out",
        "selection_metric": selection_metric,
        "inner_folds": int(inner_folds),
        "alpha_grid": [float(alpha) for alpha in alpha_grid],
        "number_of_reaches": len(per_reach),
        "mean_r2": float(np.mean(r2_values)),
        "median_r2": float(np.median(r2_values)),
        "mean_correlation": (
            float(np.mean(finite_correlations))
            if finite_correlations.size > 0
            else float("nan")
        ),
        "median_correlation": (
            float(np.median(finite_correlations))
            if finite_correlations.size > 0
            else float("nan")
        ),
        "median_selected_alpha": float(np.median(selected_alphas)),
        "per_reach": per_reach,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare firing rates and DNN activations using CCA-family, CKA, RSA, Procrustes, and ridge metrics.")
    parser.add_argument("firing_rates_input", help="Firing-rate JSON file or folder of JSON files")
    parser.add_argument("activations_input", help="Activation JSON file or folder of JSON files")
    parser.add_argument("output_dir", help="Directory for the results JSON file")
    parser.add_argument(
        "--output-filename",
        default="neural_comparison_results.json",
        help=(
            "Optional output JSON filename. "
            "Defaults to neural_comparison_results.json"
        ),
    )

    cca_group = parser.add_argument_group("CCA / SVCCA / PWCCA")
    cca_group.add_argument(
        "--reg-min",
        type=float,
        default=1e-8,
        help="Smallest CCA-family regularization value in the logarithmic sweep.",
    )
    cca_group.add_argument(
        "--reg-max",
        type=float,
        default=1e-2,
        help="Largest CCA-family regularization value in the logarithmic sweep.",
    )
    cca_group.add_argument(
        "--reg-runs",
        type=int,
        default=7,
        help="Number of logarithmically spaced regularization values.",
    )
    cca_group.add_argument("--svcca-variance", type=float, default=0.90)
    cca_group.add_argument("--pwcca-weight-side", choices=["x", "y"], default="x")
    cca_group.add_argument(
        "--component-min",
        type=int,
        default=5,
        help="Smallest component count included in the automatic sweep.",
    )
    cca_group.add_argument(
        "--component-runs",
        type=int,
        default=20,
        help="Maximum number of evenly spaced component counts to evaluate.",
    )

    cka_group = parser.add_argument_group("CKA")
    cka_group.add_argument("--cka-debiased", action="store_true")

    rsa_group = parser.add_argument_group("RSA")
    rsa_group.add_argument("--rsa-distance", default="correlation")
    rsa_group.add_argument("--rsa-correlation", choices=["spearman", "pearson"], default="spearman")

    procrustes_group = parser.add_argument_group("Procrustes")
    procrustes_group.add_argument("--no-procrustes-scaling", action="store_true")

    ridge_group = parser.add_argument_group("Cross-validated ridge")
    ridge_group.add_argument(
        "--ridge-alpha",
        type=float,
        default=1.0,
        help="Fixed alpha used only for single-file time-split ridge.",
    )
    ridge_group.add_argument(
        "--ridge-alpha-min",
        type=float,
        default=1e-4,
        help="Smallest alpha in the nested folder-mode ridge search.",
    )
    ridge_group.add_argument(
        "--ridge-alpha-max",
        type=float,
        default=1e8,
        help="Largest alpha in the nested folder-mode ridge search.",
    )
    ridge_group.add_argument(
        "--ridge-alpha-runs",
        type=int,
        default=11,
        help="Number of logarithmically spaced alphas in folder mode.",
    )
    ridge_group.add_argument(
        "--ridge-inner-folds",
        type=int,
        default=5,
        help="Number of grouped reach-level folds used to select alpha.",
    )
    ridge_group.add_argument(
        "--ridge-selection-metric",
        choices=["r2", "correlation"],
        default="r2",
        help="Inner-validation metric used to choose alpha.",
    )
    ridge_group.add_argument("--ridge-folds", type=int, default=5)
    ridge_group.add_argument("--ridge-shuffle", action="store_true")
    ridge_group.add_argument("--ridge-random-state", type=int, default=0)
    ridge_group.add_argument("--no-ridge-standardize", action="store_true")
    return parser


def center_each_reach(
    matrix: np.ndarray,
    matched_files: list[dict[str, Any]],
) -> np.ndarray:
    """Subtract each feature's within-reach temporal mean."""
    centered = matrix.copy()

    for file_info in matched_files:
        start = int(file_info["time_start"])
        stop = int(file_info["time_stop"])

        if start < 0 or stop > centered.shape[1] or start >= stop:
            raise ValueError(
                f"Invalid reach range for {file_info.get('name', 'unknown')}: "
                f"[{start}, {stop})"
            )

        reach = centered[:, start:stop]
        centered[:, start:stop] = (
            reach - reach.mean(axis=1, keepdims=True)
        )

    return centered


def make_ridge_alpha_grid(args: argparse.Namespace) -> list[float]:
    if args.ridge_alpha_min <= 0 or args.ridge_alpha_max <= 0:
        raise ValueError("Ridge alpha bounds must be positive")
    if args.ridge_alpha_min > args.ridge_alpha_max:
        raise ValueError("--ridge-alpha-min cannot exceed --ridge-alpha-max")
    if args.ridge_alpha_runs < 1:
        raise ValueError("--ridge-alpha-runs must be at least 1")

    if args.ridge_alpha_runs == 1:
        return [float(args.ridge_alpha_min)]

    return [
        float(value)
        for value in np.logspace(
            np.log10(args.ridge_alpha_min),
            np.log10(args.ridge_alpha_max),
            num=args.ridge_alpha_runs,
        )
    ]


def run_analysis(
    firing_rates: np.ndarray,
    activations: np.ndarray,
    input_metadata: dict[str, Any],
    args: argparse.Namespace,
    component_grid: list[int],
    regularization_grid: list[float],
) -> dict[str, Any]:
    """Run all metrics once on the supplied matrices."""
    cca_results = {}
    svcca_results = {}
    pwcca_results = {}
    procrustes_results = {}

    for reg in regularization_grid:
        reg_key = format_reg_key(reg)
        cca_results[reg_key] = {}
        svcca_results[reg_key] = {}
        pwcca_results[reg_key] = {}

        for n_components in component_grid:
            component_key = str(n_components)

            cca_results[reg_key][component_key] = cca(
                firing_rates,
                activations,
                reg=reg,
                max_components=n_components,
            )

            svcca_results[reg_key][component_key] = svcca(
                firing_rates,
                activations,
                variance_threshold=args.svcca_variance,
                max_pca_components=n_components,
                max_cca_components=n_components,
                reg=reg,
            )

            pwcca_results[reg_key][component_key] = pwcca(
                firing_rates,
                activations,
                reg=reg,
                max_components=n_components,
                weight_side=args.pwcca_weight_side,
            )

    for n_components in component_grid:
        component_key = str(n_components)
        procrustes_results[component_key] = procrustes(
            firing_rates,
            activations,
            n_components=n_components,
            allow_scaling=not args.no_procrustes_scaling,
        )

    cka_result = linear_cka(
        firing_rates,
        activations,
        debiased=args.cka_debiased,
    )
    rsa_result = rsa(
        firing_rates,
        activations,
        args.rsa_distance,
        args.rsa_correlation,
    )

    if input_metadata["mode"] == "folder":
        ridge_result = leave_one_reach_out_ridge(
            firing_rates=firing_rates,
            activations=activations,
            matched_files=input_metadata["matched_files"],
            alpha_grid=make_ridge_alpha_grid(args),
            inner_folds=args.ridge_inner_folds,
            selection_metric=args.ridge_selection_metric,
            standardize=not args.no_ridge_standardize,
        )
    else:
        ridge_result = cross_validated_ridge(
            firing_rates,
            activations,
            alpha=args.ridge_alpha,
            n_splits=args.ridge_folds,
            shuffle=args.ridge_shuffle,
            random_state=args.ridge_random_state,
            standardize=not args.no_ridge_standardize,
        )
        ridge_result = {
            "method": "time_split_cross_validation",
            **ridge_result,
        }

    compact_cca = {
        reg_key: {
            component_key: result["mean_correlation"]
            for component_key, result in component_results.items()
        }
        for reg_key, component_results in cca_results.items()
    }

    compact_svcca = {
        reg_key: {
            component_key: {
                "score": result["svcca_score"],
                **(
                    {"x_pca_components": result["x_pca_components"]}
                    if "x_pca_components" in result
                    else {}
                ),
                **(
                    {"y_pca_components": result["y_pca_components"]}
                    if "y_pca_components" in result
                    else {}
                ),
            }
            for component_key, result in component_results.items()
        }
        for reg_key, component_results in svcca_results.items()
    }

    compact_pwcca = {
        reg_key: {
            component_key: result["pwcca_score"]
            for component_key, result in component_results.items()
        }
        for reg_key, component_results in pwcca_results.items()
    }

    compact_procrustes = {
        component_key: result["procrustes_similarity"]
        for component_key, result in procrustes_results.items()
    }

    compact_ridge = (
        {
            "method": ridge_result["method"],
            "selection_metric": ridge_result["selection_metric"],
            "inner_folds": ridge_result["inner_folds"],
            "alpha_grid": ridge_result["alpha_grid"],
            "number_of_reaches": ridge_result["number_of_reaches"],
            "median_selected_alpha": ridge_result[
                "median_selected_alpha"
            ],
            "mean_r2": ridge_result["mean_r2"],
            "median_r2": ridge_result["median_r2"],
            "mean_correlation": ridge_result["mean_correlation"],
            "median_correlation": ridge_result[
                "median_correlation"
            ],
            "per_reach": ridge_result["per_reach"],
        }
        if ridge_result["method"] == "nested_leave_one_reach_out"
        else {
            "method": ridge_result["method"],
            "mean_r2": ridge_result["mean_r2"],
            "mean_correlation": ridge_result["mean_correlation"],
        }
    )

    return {
        "metrics": {
            "cca_mean_correlation": compact_cca,
            "svcca": compact_svcca,
            "pwcca_score": compact_pwcca,
            "procrustes_similarity": compact_procrustes,
            "cka_score": cka_result["cka_score"],
            "rsa_score": rsa_result["rsa_score"],
            "ridge": compact_ridge,
        },
        "_print_data": {
            "cca_results": cca_results,
            "svcca_results": svcca_results,
            "pwcca_results": pwcca_results,
            "procrustes_results": procrustes_results,
            "cka_result": cka_result,
            "rsa_result": rsa_result,
            "ridge_result": ridge_result,
        },
    }


def print_analysis(
    title: str,
    analysis: dict[str, Any],
    component_grid: list[int],
    regularization_grid: list[float],
) -> None:
    print()
    print("=" * len(title))
    print(title)
    print("=" * len(title))

    print_data = analysis["_print_data"]
    cca_results = print_data["cca_results"]
    svcca_results = print_data["svcca_results"]
    pwcca_results = print_data["pwcca_results"]
    procrustes_results = print_data["procrustes_results"]
    cka_result = print_data["cka_result"]
    rsa_result = print_data["rsa_result"]
    ridge_result = print_data["ridge_result"]

    component_headers = ["Regularization"] + [
        str(n_components) for n_components in component_grid
    ]

    cca_rows = []
    svcca_rows = []
    pwcca_rows = []

    for reg in regularization_grid:
        reg_key = format_reg_key(reg)

        cca_rows.append(
            [reg_key]
            + [
                f"{cca_results[reg_key][str(n_components)]['mean_correlation']:.2f}"
                for n_components in component_grid
            ]
        )
        svcca_rows.append(
            [reg_key]
            + [
                f"{svcca_results[reg_key][str(n_components)]['svcca_score']:.2f}"
                for n_components in component_grid
            ]
        )
        pwcca_rows.append(
            [reg_key]
            + [
                f"{pwcca_results[reg_key][str(n_components)]['pwcca_score']:.2f}"
                for n_components in component_grid
            ]
        )

    print_table("CCA sweep", component_headers, cca_rows)
    print_table("SVCCA sweep", component_headers, svcca_rows)
    print_table("PWCCA sweep", component_headers, pwcca_rows)

    print_table(
        title="Procrustes component sweep",
        headers=["Components", "Procrustes"],
        rows=[
            [
                n_components,
                f"{procrustes_results[str(n_components)]['procrustes_similarity']:.2f}",
            ]
            for n_components in component_grid
        ],
    )

    print_table(
        title="Metrics without component/regularization sweeps",
        headers=["Metric", "Score"],
        rows=[
            ["CKA", f"{cka_result['cka_score']:.2f}"],
            ["RSA", f"{rsa_result['rsa_score']:.2f}"],
            [
                (
                    "Nested LORO ridge mean R2"
                    if ridge_result["method"]
                    == "nested_leave_one_reach_out"
                    else "Ridge mean R2"
                ),
                f"{ridge_result['mean_r2']:.2f}",
            ],
            [
                (
                    "Nested LORO ridge mean correlation"
                    if ridge_result["method"]
                    == "nested_leave_one_reach_out"
                    else "Ridge mean correlation"
                ),
                f"{ridge_result['mean_correlation']:.2f}",
            ],
        ],
    )

    if ridge_result["method"] == "nested_leave_one_reach_out":
        print_table(
            title="Nested leave-one-reach-out ridge",
            headers=["Held-out reach", "Alpha", "R2", "Correlation"],
            rows=[
                [
                    result["held_out_reach"],
                    f"{result['selected_alpha']:.2e}",
                    f"{result['r2']:.2f}",
                    f"{result['mean_correlation']:.2f}",
                ]
                for result in ridge_result["per_reach"]
            ],
        )


def main() -> None:
    args = build_parser().parse_args()
    firing_rates, activations, input_metadata = load_input_pair(
        args.firing_rates_input,
        args.activations_input,
    )

    max_valid_components = min(
        firing_rates.shape[0],
        activations.shape[0],
        firing_rates.shape[1] - 1,
    )

    if args.component_min < 1:
        raise ValueError(
            f"--component-min must be at least 1, got {args.component_min}"
        )
    if args.component_runs < 1:
        raise ValueError(
            f"--component-runs must be at least 1, got {args.component_runs}"
        )

    component_grid = make_component_grid(
        max_valid_components,
        min_components=args.component_min,
        num_values=args.component_runs,
    )
    regularization_grid = make_regularization_grid(
        min_reg=args.reg_min,
        max_reg=args.reg_max,
        num_values=args.reg_runs,
    )

    analyses: dict[str, dict[str, Any]] = {}

    analyses["raw"] = run_analysis(
        firing_rates=firing_rates,
        activations=activations,
        input_metadata=input_metadata,
        args=args,
        component_grid=component_grid,
        regularization_grid=regularization_grid,
    )

    if input_metadata["mode"] == "folder":
        centered_firing_rates = center_each_reach(
            firing_rates,
            input_metadata["matched_files"],
        )
        centered_activations = center_each_reach(
            activations,
            input_metadata["matched_files"],
        )

        analyses["per_reach_centered"] = run_analysis(
            firing_rates=centered_firing_rates,
            activations=centered_activations,
            input_metadata=input_metadata,
            args=args,
            component_grid=component_grid,
            regularization_grid=regularization_grid,
        )

    output = {
        "inputs": {
            "firing_rates_input": str(Path(args.firing_rates_input)),
            "activations_input": str(Path(args.activations_input)),
            **input_metadata,
        },
        "shapes": {
            "firing_rates": list(firing_rates.shape),
            "activations": list(activations.shape),
        },
        "settings": {
            **{
                key: value
                for key, value in vars(args).items()
                if key not in {"reg_min", "reg_max"}
            },
            "component_sweep": {
                "minimum_requested_components": args.component_min,
                "number_of_requested_values": args.component_runs,
                "maximum_valid_components": max_valid_components,
                "component_grid": component_grid,
            },
            "regularization_sweep": {
                "minimum_regularization": format_reg_key(args.reg_min),
                "maximum_regularization": format_reg_key(args.reg_max),
                "number_of_requested_values": args.reg_runs,
                "regularization_grid": [
                    format_reg_key(reg) for reg in regularization_grid
                ],
            },
            "analysis_conditions": list(analyses.keys()),
            "per_reach_centering": {
                "enabled_for_folder_mode": True,
                "operation": (
                    "For each matched reach and each feature independently, "
                    "subtract the temporal mean of that reach."
                ),
            },
        },
        "analyses": {
            condition: {
                "preprocessing": (
                    "none"
                    if condition == "raw"
                    else "per_reach_feature_centering"
                ),
                **analysis["metrics"],
            }
            for condition, analysis in analyses.items()
        },
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_filename = Path(args.output_filename)
    if output_filename.name != args.output_filename:
        raise ValueError(
            "--output-filename must be a filename only, not a path: "
            f"{args.output_filename}"
        )
    if output_filename.suffix.lower() != ".json":
        raise ValueError(
            "--output-filename must end with .json, got: "
            f"{args.output_filename}"
        )

    output_path = output_dir / output_filename

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(json_safe(output, decimals=2), file, indent=2)

    print(f"Saved all comparison results to: {output_path}")
    print(f"Input mode:          {input_metadata['mode']}")
    print(f"Matched files:       {input_metadata['matched_file_count']}")
    print(f"Firing rates shape:  {firing_rates.shape}")
    print(f"Activations shape:   {activations.shape}")
    print(f"SVCCA variance threshold: {args.svcca_variance:.2f}")

    print_analysis(
        title="RAW ANALYSIS",
        analysis=analyses["raw"],
        component_grid=component_grid,
        regularization_grid=regularization_grid,
    )

    if "per_reach_centered" in analyses:
        print_analysis(
            title="PER-REACH CENTERED ANALYSIS",
            analysis=analyses["per_reach_centered"],
            component_grid=component_grid,
            regularization_grid=regularization_grid,
        )


if __name__ == "__main__":
    main()