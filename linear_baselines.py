#!/usr/bin/env python3

"""Train simple Ridge/Lasso-style baselines on the master table without external deps."""

from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


DEFAULT_TARGET = "Fmax_eps_per_A"
DEFAULT_LENGTH_FEATURE = "N"
EXCLUDED_FEATURES = {"Rank"}


def load_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Cannot find dataset at {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("Dataset has no header row")
        return [row for row in reader]


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def detect_numeric_columns(rows: List[Dict[str, str]]) -> List[str]:
    if not rows:
        return []
    columns = list(rows[0].keys())
    numeric = []
    for column in columns:
        all_numeric = True
        for row in rows:
            value = parse_float(row.get(column))
            if value is None and row.get(column, "").strip() != "":
                all_numeric = False
                break
        if all_numeric:
            numeric.append(column)
    return numeric


def compute_column_means(rows: List[Dict[str, str]], columns: Sequence[str]) -> Dict[str, float]:
    means: Dict[str, float] = {}
    for column in columns:
        values = [parse_float(row.get(column)) for row in rows]
        filtered = [value for value in values if value is not None]
        if filtered:
            means[column] = sum(filtered) / len(filtered)
        else:
            means[column] = 0.0
    return means


def build_matrices(
    rows: List[Dict[str, str]],
    target_column: str,
    length_column: str,
) -> Tuple[List[List[float]], List[float], List[float], List[str]]:
    numeric_columns = detect_numeric_columns(rows)
    if target_column not in numeric_columns:
        raise ValueError(f"Target column '{target_column}' must be numeric")
    if length_column not in numeric_columns:
        raise ValueError(f"Length column '{length_column}' must be numeric")
    feature_columns = [
        column
        for column in numeric_columns
        if column != target_column and column not in EXCLUDED_FEATURES
    ]
    if not feature_columns:
        raise ValueError("No numeric features available for modeling")

    column_means = compute_column_means(rows, feature_columns)

    features: List[List[float]] = []
    targets: List[float] = []
    lengths: List[float] = []
    for row in rows:
        target_value = parse_float(row.get(target_column))
        length_value = parse_float(row.get(length_column))
        if target_value is None or length_value is None:
            continue
        feature_vector: List[float] = []
        for column in feature_columns:
            value = parse_float(row.get(column))
            if value is None:
                value = column_means[column]
            feature_vector.append(value)
        features.append(feature_vector)
        targets.append(target_value)
        lengths.append(length_value)
    if not features:
        raise ValueError("No rows contained the target and length columns")
    return features, targets, lengths, feature_columns

#This doesn't cluster by sequence similarity, so it may overestimate performance on homologs.
def train_test_split(
    features: List[List[float]],
    targets: List[float],
    lengths: List[float],
    test_size: float,
    seed: int,
) -> Tuple[List[List[float]], List[List[float]], List[float], List[float], List[float], List[float]]:
    indices = list(range(len(features)))
    random.Random(seed).shuffle(indices)
    split = max(1, int(len(indices) * (1 - test_size)))
    train_idx = indices[:split]
    test_idx = indices[split:]
    if not test_idx:
        raise ValueError("Test split is empty; choose a larger test_size")

    def gather(idx_list: List[int]) -> Tuple[List[List[float]], List[float], List[float]]:
        return (
            [features[i] for i in idx_list],
            [targets[i] for i in idx_list],
            [lengths[i] for i in idx_list],
        )

    X_train, y_train, len_train = gather(train_idx)
    X_test, y_test, len_test = gather(test_idx)
    return X_train, X_test, y_train, y_test, len_train, len_test


def compute_standardization(matrix: List[List[float]]) -> Tuple[List[float], List[float]]:
    rows = len(matrix)
    cols = len(matrix[0])
    means = [0.0] * cols
    stds = [0.0] * cols
    for j in range(cols):
        column_values = [matrix[i][j] for i in range(rows)]
        mean = sum(column_values) / rows
        variance = sum((value - mean) ** 2 for value in column_values) / rows
        std = math.sqrt(variance) if variance > 0 else 1.0
        means[j] = mean
        stds[j] = std
    return means, stds


#Ridge penalizes weights, so standardizing features is important.
def apply_standardization(matrix: List[List[float]], means: List[float], stds: List[float]) -> List[List[float]]:
    standardized: List[List[float]] = []
    for row in matrix:
        standardized.append([(value - mean) / std for value, mean, std in zip(row, means, stds)])
    return standardized


def dot(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(vec_a, vec_b))


def predict_row(weights: Sequence[float], bias: float, row: Sequence[float]) -> float:
    return bias + dot(weights, row)


def train_ridge(
    X: List[List[float]], y: List[float], alpha: float = 1.0, lr: float = 0.01, epochs: int = 5000
) -> Tuple[List[float], float]:
    rows = len(X)
    cols = len(X[0])
    weights = [0.0] * cols
    bias = sum(y) / rows
    for _ in range(epochs):
        errors = [predict_row(weights, bias, X[i]) - y[i] for i in range(rows)]
        grad_w = []
        for j in range(cols):
            grad = (2.0 / rows) * sum(errors[i] * X[i][j] for i in range(rows)) + 2 * alpha * weights[j]
            grad_w.append(grad)
        grad_b = (2.0 / rows) * sum(errors)
        weights = [weights[j] - lr * grad_w[j] for j in range(cols)]
        bias -= lr * grad_b
    return weights, bias


def sign(value: float) -> float:
    if value > 0:
        return 1.0
    if value < 0:
        return -1.0
    return 0.0


def train_lasso(
    X: List[List[float]],
    y: List[float],
    alpha: float = 0.001,
    lr: float = 0.005,
    epochs: int = 6000,
) -> Tuple[List[float], float]:
    rows = len(X)
    cols = len(X[0])
    weights = [0.0] * cols
    bias = sum(y) / rows
    for _ in range(epochs):
        errors = [predict_row(weights, bias, X[i]) - y[i] for i in range(rows)]
        grad_w = []
        for j in range(cols):
            grad = (2.0 / rows) * sum(errors[i] * X[i][j] for i in range(rows)) + alpha * sign(weights[j])
            grad_w.append(grad)
        grad_b = (2.0 / rows) * sum(errors)
        weights = [weights[j] - lr * grad_w[j] for j in range(cols)]
        bias -= lr * grad_b
    return weights, bias


def train_simple_linear(x: List[float], y: List[float]) -> Tuple[float, float]:
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    denominator = sum((x[i] - mean_x) ** 2 for i in range(n))
    slope = numerator / denominator if denominator else 0.0
    intercept = mean_y - slope * mean_x
    return slope, intercept


def evaluate(y_true: List[float], y_pred: List[float]) -> Tuple[float, float]:
    n = len(y_true)
    mse = sum((y_true[i] - y_pred[i]) ** 2 for i in range(n)) / n
    rmse = math.sqrt(mse)
    mean_y = sum(y_true) / n
    total = sum((value - mean_y) ** 2 for value in y_true)
    if total == 0:
        r2 = 0.0
    else:
        r2 = 1 - sum((y_true[i] - y_pred[i]) ** 2 for i in range(n)) / total
    return rmse, r2


def generate_folds(
    n_samples: int, folds: int, seed: int | None = None
) -> Iterable[Tuple[List[int], List[int]]]:
    indices = list(range(n_samples))
    if seed is not None:
        random.Random(seed).shuffle(indices)
    folds = max(2, min(folds, n_samples))
    for fold in range(folds):
        test_idx = indices[fold::folds]
        if not test_idx:
            continue
        test_set = set(test_idx)
        train_idx = [idx for idx in indices if idx not in test_set]
        if not train_idx:
            continue
        yield train_idx, test_idx


def tune_lasso_alpha(
    X: List[List[float]],
    y: List[float],
    alphas: Sequence[float],
    folds: int,
    lr: float,
    epochs: int,
    seed: int | None,
) -> float:
    best_alpha = alphas[0]
    best_rmse = float("inf")
    for alpha in alphas:
        fold_rmses: List[float] = []
        for train_idx, test_idx in generate_folds(len(X), folds, seed):
            X_train = [X[i] for i in train_idx]
            y_train = [y[i] for i in train_idx]
            X_test = [X[i] for i in test_idx]
            y_test = [y[i] for i in test_idx]
            weights, bias = train_lasso(X_train, y_train, alpha=alpha, lr=lr, epochs=epochs)
            preds = [predict_row(weights, bias, row) for row in X_test]
            rmse, _ = evaluate(y_test, preds)
            fold_rmses.append(rmse)
        if not fold_rmses:
            continue
        avg_rmse = sum(fold_rmses) / len(fold_rmses)
        if avg_rmse < best_rmse:
            best_rmse = avg_rmse
            best_alpha = alpha
    return best_alpha


def tune_ridge_alpha(
    X: List[List[float]],
    y: List[float],
    alphas: Sequence[float],
    folds: int,
    lr: float,
    epochs: int,
    seed: int | None,
) -> float:
    best_alpha = alphas[0]
    best_rmse = float("inf")
    for alpha in alphas:
        fold_rmses: List[float] = []
        for train_idx, test_idx in generate_folds(len(X), folds, seed):
            X_train = [X[i] for i in train_idx]
            y_train = [y[i] for i in train_idx]
            X_test = [X[i] for i in test_idx]
            y_test = [y[i] for i in test_idx]
            weights, bias = train_ridge(X_train, y_train, alpha=alpha, lr=lr, epochs=epochs)
            preds = [predict_row(weights, bias, row) for row in X_test]
            rmse, _ = evaluate(y_test, preds)
            fold_rmses.append(rmse)
        if not fold_rmses:
            continue
        avg_rmse = sum(fold_rmses) / len(fold_rmses)
        if avg_rmse < best_rmse:
            best_rmse = avg_rmse
            best_alpha = alpha
    return best_alpha


def evaluate_with_cross_validation(
    X: List[List[float]],
    y: List[float],
    lengths: List[float],
    eval_folds: int,
    seed: int,
    ridge_alpha: float,
    ridge_lr: float,
    ridge_epochs: int,
    ridge_grid: List[float] | None,
    ridge_cv_folds: int,
    lasso_alpha: float,
    lasso_lr: float,
    lasso_epochs: int,
    lasso_grid: List[float] | None,
    lasso_cv_folds: int,
    length_column: str,
) -> Tuple[Tuple[float, float, str], Tuple[float, float, str], Tuple[float, float]]:
    y_true_all: List[float] = []
    ridge_preds_all: List[float] = []
    lasso_preds_all: List[float] = []
    length_preds_all: List[float] = []
    ridge_alphas: List[float] = []
    lasso_alphas: List[float] = []
    for fold_idx, (train_idx, test_idx) in enumerate(
        generate_folds(len(X), eval_folds, seed)
    ):
        X_train = [X[i] for i in train_idx]
        y_train = [y[i] for i in train_idx]
        len_train = [lengths[i] for i in train_idx]
        X_test = [X[i] for i in test_idx]
        y_test = [y[i] for i in test_idx]
        len_test = [lengths[i] for i in test_idx]

        means, stds = compute_standardization(X_train)
        X_train_scaled = apply_standardization(X_train, means, stds)
        X_test_scaled = apply_standardization(X_test, means, stds)
        inner_seed = None if seed is None else seed + fold_idx + 1

        if ridge_grid:
            inner_folds = min(ridge_cv_folds, len(X_train))
            ridge_alpha_fold = tune_ridge_alpha(
                X_train_scaled,
                y_train,
                ridge_grid,
                inner_folds,
                ridge_lr,
                ridge_epochs,
                inner_seed,
            )
            ridge_alphas.append(ridge_alpha_fold)
        else:
            ridge_alpha_fold = ridge_alpha

        ridge_weights, ridge_bias = train_ridge(
            X_train_scaled,
            y_train,
            alpha=ridge_alpha_fold,
            lr=ridge_lr,
            epochs=ridge_epochs,
        )
        ridge_preds = [predict_row(ridge_weights, ridge_bias, row) for row in X_test_scaled]

        if lasso_grid:
            inner_folds = min(lasso_cv_folds, len(X_train))
            lasso_alpha_fold = tune_lasso_alpha(
                X_train_scaled,
                y_train,
                lasso_grid,
                inner_folds,
                lr=lasso_lr,
                epochs=lasso_epochs,
                seed=inner_seed,
            )
            lasso_alphas.append(lasso_alpha_fold)
        else:
            lasso_alpha_fold = lasso_alpha

        lasso_weights, lasso_bias = train_lasso(
            X_train_scaled,
            y_train,
            alpha=lasso_alpha_fold,
            lr=lasso_lr,
            epochs=lasso_epochs,
        )
        lasso_preds = [predict_row(lasso_weights, lasso_bias, row) for row in X_test_scaled]

        slope, intercept = train_simple_linear(len_train, y_train)
        length_preds = [intercept + slope * value for value in len_test]

        y_true_all.extend(y_test)
        ridge_preds_all.extend(ridge_preds)
        lasso_preds_all.extend(lasso_preds)
        length_preds_all.extend(length_preds)

    ridge_rmse, ridge_r2 = evaluate(y_true_all, ridge_preds_all)
    lasso_rmse, lasso_r2 = evaluate(y_true_all, lasso_preds_all)
    length_rmse, length_r2 = evaluate(y_true_all, length_preds_all)

    def format_alpha(name: str, alphas: List[float], default_alpha: float) -> str:
        if not alphas:
            return name
        avg_alpha = sum(alphas) / len(alphas)
        return f"{name} (alpha≈{avg_alpha:g})"

    ridge_label = format_alpha("Ridge (GD)", ridge_alphas, ridge_alpha)
    lasso_label = format_alpha("Lasso", lasso_alphas, lasso_alpha)
    return (ridge_rmse, ridge_r2, ridge_label), (lasso_rmse, lasso_r2, lasso_label), (
        length_rmse,
        length_r2,
    )


def run_baselines(
    dataset_path: Path,
    target_column: str,
    length_column: str,
    test_size: float,
    seed: int,
    eval_cv_folds: int,
    ridge_alpha: float,
    ridge_lr: float,
    ridge_epochs: int,
    ridge_grid: List[float] | None,
    ridge_cv_folds: int,
    lasso_alpha: float,
    lasso_lr: float,
    lasso_epochs: int,
    lasso_grid: List[float] | None,
    lasso_cv_folds: int,
) -> None:
    rows = load_rows(dataset_path)
    X, y, lengths, feature_columns = build_matrices(rows, target_column, length_column)
    if eval_cv_folds > 1:
        ridge_metrics, lasso_metrics, length_metrics = evaluate_with_cross_validation(
            X,
            y,
            lengths,
            eval_cv_folds,
            seed,
            ridge_alpha,
            ridge_lr,
            ridge_epochs,
            ridge_grid,
            ridge_cv_folds,
            lasso_alpha,
            lasso_lr,
            lasso_epochs,
            lasso_grid,
            lasso_cv_folds,
            length_column,
        )
        print("Linear baselines on", dataset_path)
        print(f"Samples: {len(X)}  |  Features: {len(feature_columns)}  |  Target: {target_column}")
        print(f"Evaluation: {eval_cv_folds}-fold cross-validation\n")
        header = f"{'Model':<22}{'RMSE':>12}{'R^2':>12}"
        print(header)
        print(f"{'-' * len(header)}")
        ridge_rmse, ridge_r2, ridge_label = ridge_metrics
        lasso_rmse, lasso_r2, lasso_label = lasso_metrics
        length_rmse, length_r2 = length_metrics
        print(f"{ridge_label:<22}{ridge_rmse:>12.4f}{ridge_r2:>12.4f}")
        print(f"{lasso_label:<22}{lasso_rmse:>12.4f}{lasso_r2:>12.4f}")
        print(f"{f'Length-only ({length_column})':<22}{length_rmse:>12.4f}{length_r2:>12.4f}")
        return
    X_train, X_test, y_train, y_test, len_train, len_test = train_test_split(
        X, y, lengths, test_size, seed
    )

    means, stds = compute_standardization(X_train)
    X_train_scaled = apply_standardization(X_train, means, stds)
    X_test_scaled = apply_standardization(X_test, means, stds)

    if ridge_grid:
        best_ridge_alpha = tune_ridge_alpha(
            X_train_scaled,
            y_train,
            ridge_grid,
            folds=ridge_cv_folds,
            lr=ridge_lr,
            epochs=ridge_epochs,
            seed=seed,
        )
    else:
        best_ridge_alpha = ridge_alpha

    ridge_weights, ridge_bias = train_ridge(
        X_train_scaled, y_train, alpha=best_ridge_alpha, lr=ridge_lr, epochs=ridge_epochs
    )
    ridge_preds = [predict_row(ridge_weights, ridge_bias, row) for row in X_test_scaled]

    if lasso_grid:
        best_alpha = tune_lasso_alpha(
            X_train_scaled,
            y_train,
            lasso_grid,
            folds=lasso_cv_folds,
            lr=lasso_lr,
            epochs=lasso_epochs,
            seed=seed,
        )
    else:
        best_alpha = lasso_alpha

    lasso_weights, lasso_bias = train_lasso(
        X_train_scaled, y_train, alpha=best_alpha, lr=lasso_lr, epochs=lasso_epochs
    )
    lasso_preds = [predict_row(lasso_weights, lasso_bias, row) for row in X_test_scaled]

    slope, intercept = train_simple_linear(len_train, y_train)
    length_preds = [intercept + slope * value for value in len_test]

    ridge_rmse, ridge_r2 = evaluate(y_test, ridge_preds)
    lasso_rmse, lasso_r2 = evaluate(y_test, lasso_preds)
    length_rmse, length_r2 = evaluate(y_test, length_preds)

    print("Linear baselines on", dataset_path)
    print(f"Samples: {len(X)}  |  Features: {len(feature_columns)}  |  Target: {target_column}")
    print(f"Test fraction: {test_size}\n")
    header = f"{'Model':<22}{'RMSE':>12}{'R^2':>12}"
    print(header)
    print(f"{'-' * (len(header))}")
    ridge_name = "Ridge (GD)"
    if ridge_grid:
        ridge_name += f" (alpha={best_ridge_alpha:g})"
    print(f"{ridge_name:<22}{ridge_rmse:>12.4f}{ridge_r2:>12.4f}")
    lasso_name = "Lasso"
    if lasso_grid:
        lasso_name += f" (alpha={best_alpha:g})"
    print(f"{lasso_name:<22}{lasso_rmse:>12.4f}{lasso_r2:>12.4f}")
    print(f"{f'Length-only ({length_column})':<22}{length_rmse:>12.4f}{length_r2:>12.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="master_table_beta.csv", help="Path to master table CSV")
    parser.add_argument(
        "--target", default=DEFAULT_TARGET, help="Target column to predict"
    )
    parser.add_argument(
        "--length-column",
        default=DEFAULT_LENGTH_FEATURE,
        help="Column to use for the length-only baseline",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction for test split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--eval-cv-folds",
        type=int,
        default=0,
        help="Use k-fold CV (>=2) for final evaluation instead of a hold-out test split",
    )
    parser.add_argument("--ridge-alpha", type=float, default=1.0, help="L2 penalty strength")
    parser.add_argument(
        "--ridge-lr",
        type=float,
        default=0.01,
        help="Learning rate for Ridge gradient descent",
    )
    parser.add_argument(
        "--ridge-epochs",
        type=int,
        default=5000,
        help="Training epochs for Ridge",
    )
    parser.add_argument(
        "--ridge-grid",
        default="",
        help="Comma-separated list of Ridge alphas for cross-validation",
    )
    parser.add_argument(
        "--ridge-cv-folds",
        type=int,
        default=5,
        help="Number of folds for Ridge alpha selection",
    )
    parser.add_argument("--lasso-alpha", type=float, default=0.001, help="L1 penalty strength")
    parser.add_argument(
        "--lasso-lr",
        type=float,
        default=0.005,
        help="Learning rate for Lasso subgradient descent",
    )
    parser.add_argument(
        "--lasso-epochs",
        type=int,
        default=6000,
        help="Training epochs for Lasso",
    )
    parser.add_argument(
        "--lasso-grid",
        default="",
        help="Comma-separated list of alpha values for cross-validation",
    )
    parser.add_argument(
        "--lasso-cv-folds",
        type=int,
        default=5,
        help="Number of folds for Lasso alpha selection",
    )
    return parser.parse_args()


def parse_grid(text: str, flag_name: str) -> List[float]:
    values: List[float] = []
    for part in text.split(","):
        stripped = part.strip()
        if not stripped:
            continue
        try:
            values.append(float(stripped))
        except ValueError:
            raise ValueError(f"Invalid alpha value '{stripped}' in {flag_name}")
    return values


def main() -> None:
    args = parse_args()
    ridge_grid = parse_grid(args.ridge_grid, "--ridge-grid")
    lasso_grid = parse_grid(args.lasso_grid, "--lasso-grid")
    run_baselines(
        dataset_path=Path(args.data),
        target_column=args.target,
        length_column=args.length_column,
        test_size=args.test_size,
        seed=args.seed,
        eval_cv_folds=args.eval_cv_folds,
        ridge_alpha=args.ridge_alpha,
        ridge_lr=args.ridge_lr,
        ridge_epochs=args.ridge_epochs,
        ridge_grid=ridge_grid,
        ridge_cv_folds=args.ridge_cv_folds,
        lasso_alpha=args.lasso_alpha,
        lasso_lr=args.lasso_lr,
        lasso_epochs=args.lasso_epochs,
        lasso_grid=lasso_grid,
        lasso_cv_folds=args.lasso_cv_folds,
    )


if __name__ == "__main__":
    main()
