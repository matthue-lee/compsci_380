#!/usr/bin/env python3

"""Train nonlinear regressors (Random Forest, XGBoost) on the master table."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit("numpy is required to run this script (pip install numpy)") from exc

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit("pandas is required to run this script (pip install pandas)") from exc

try:
    import matplotlib.pyplot as plt
    from sklearn.base import clone
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import KFold, LeaveOneOut
    from sklearn.pipeline import Pipeline
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit(
        "matplotlib and scikit-learn are required (pip install matplotlib scikit-learn)"
    ) from exc

try:  # XGBoost is optional; fall back gracefully if unavailable.
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover - optional dependency
    XGBRegressor = None


DEFAULT_TARGET = "Fmax_eps_per_A"
EXCLUDED_FEATURES = {"Rank"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="master_table_beta.csv", help="Path to CSV dataset")
    parser.add_argument("--target", default=DEFAULT_TARGET, help="Target column to predict")
    parser.add_argument("--folds", type=int, default=5, help="Number of folds for shuffled K-Fold CV")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--skip-loocv",
        action="store_true",
        help="Skip the Leave-One-Out CV evaluation (reduces runtime)",
    )
    parser.add_argument(
        "--output-dir",
        default="figures",
        help="Directory for saving feature importance plots",
    )
    parser.add_argument(
        "--top-features",
        type=int,
        default=15,
        help="Number of top features to visualize for each model",
    )
    return parser.parse_args()


def load_feature_matrix(path: Path, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
    if not path.exists():
        raise FileNotFoundError(f"Cannot find dataset at {path}")
    df = pd.read_csv(path)
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in {path}")
    target = pd.to_numeric(df[target_column], errors="coerce")
    df = df.assign(**{target_column: target}).dropna(subset=[target_column])

    feature_data = {}
    ordered_columns: List[str] = []
    for column in df.columns:
        if column == target_column or column in EXCLUDED_FEATURES:
            continue
        numeric = pd.to_numeric(df[column], errors="coerce")
        if numeric.notna().sum() == 0:
            continue
        feature_data[column] = numeric
        ordered_columns.append(column)
    if not feature_data:
        raise ValueError("No numeric features available after parsing")

    features = pd.DataFrame(feature_data, index=df.index)[ordered_columns]
    non_constant = features.loc[:, features.std(skipna=True) > 0]
    if non_constant.empty:
        raise ValueError("All numeric columns are constant; cannot train models")
    return non_constant, df[target_column]


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_random_forest(seed: int) -> Pipeline:
    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=5,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=seed,
        n_jobs=-1,
    )
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])


def build_xgboost(seed: int) -> Pipeline | None:
    if XGBRegressor is None:
        return None
    model = XGBRegressor(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.05,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=seed,
        n_jobs=1,
        verbosity=0,
    )
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])


def evaluate_model(
    model_label: str,
    model: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    splitter,
    fold_label: str,
) -> Tuple[str, str, float, float, float]:
    true_values: List[float] = []
    predictions: List[float] = []
    for train_idx, test_idx in splitter.split(X):
        estimator = clone(model)
        estimator.fit(X[train_idx], y[train_idx])
        preds = estimator.predict(X[test_idx])
        predictions.extend(preds.tolist())
        true_values.extend(y[test_idx].tolist())
    y_true = np.asarray(true_values)
    y_pred = np.asarray(predictions)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math_sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return model_label, fold_label, mae, rmse, r2


def math_sqrt(value: float) -> float:
    return float(np.sqrt(value))


def save_feature_importances(
    model_label: str,
    pipeline: Pipeline,
    feature_names: Sequence[str],
    output_dir: Path,
    top_k: int,
) -> None:
    model = pipeline.named_steps.get("model")
    if model is None or not hasattr(model, "feature_importances_"):
        return
    importances = np.asarray(model.feature_importances_, dtype=float)
    if importances.size == 0:
        return
    order = np.argsort(importances)[::-1]
    if top_k > 0:
        order = order[:top_k]
    sel_importances = importances[order]
    sel_features = [feature_names[idx] for idx in order]
    ensure_output_dir(output_dir)
    height = max(3, 0.4 * len(sel_features) + 1)
    fig, ax = plt.subplots(figsize=(8, height))
    ax.barh(range(len(sel_features)), sel_importances, color="#2A9D8F")
    ax.set_yticks(range(len(sel_features)))
    ax.set_yticklabels(sel_features)
    ax.invert_yaxis()
    ax.set_xlabel("Feature importance")
    ax.set_title(f"{model_label} feature importance")
    fig.tight_layout()
    path = output_dir / f"feature_importance_{model_label.lower().replace(' ', '_')}.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved {path}")


def print_results(summary: Sequence[Tuple[str, str, float, float, float]]) -> None:
    header = f"{'Model':<16}{'CV':<10}{'MAE':>10}{'RMSE':>10}{'R^2':>10}"
    print(header)
    print("-" * len(header))
    for model_name, label, mae, rmse, r2 in summary:
        print(
            f"{model_name:<16}{label:<10}"
            f"{mae:>10.4f}{rmse:>10.4f}{r2:>10.4f}"
        )


def run() -> None:
    args = parse_args()
    if args.folds < 2:
        raise ValueError("Number of folds must be at least 2")
    features, target = load_feature_matrix(Path(args.data), args.target)
    feature_names = list(features.columns)
    X = features.to_numpy(dtype=np.float32)
    y = target.to_numpy(dtype=np.float32)

    models: List[Tuple[str, Pipeline]] = [("Random Forest", build_random_forest(args.seed))]
    xgb_pipeline = build_xgboost(args.seed)
    if xgb_pipeline is not None:
        models.append(("XGBoost", xgb_pipeline))
    else:
        print("[warning] xgboost not installed; skipping XGBoost model", file=sys.stderr)

    summary = []
    if args.folds > 1:
        kfold = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        for label, pipeline in models:
            summary.append(
                evaluate_model(label, pipeline, X, y, kfold, f"{args.folds}-fold")
            )
            if not args.skip_loocv:
                loocv = LeaveOneOut()
                summary.append(evaluate_model(label, pipeline, X, y, loocv, "LOOCV"))
    else:
        for label, pipeline in models:
            if not args.skip_loocv:
                loocv = LeaveOneOut()
                summary.append(evaluate_model(label, pipeline, X, y, loocv, "LOOCV"))

    print(
        f"Samples: {X.shape[0]}  |  Features: {X.shape[1]}  |  Target: {args.target}\n"
    )
    print_results(summary)

    output_dir = Path(args.output_dir)
    for label, pipeline in models:
        fitted = clone(pipeline)
        fitted.fit(X, y)
        save_feature_importances(label, fitted, feature_names, output_dir, args.top_features)


if __name__ == "__main__":
    run()
