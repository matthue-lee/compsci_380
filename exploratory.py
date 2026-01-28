from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DATA_PATH = Path(__file__).with_name("dataset.csv")
OUTPUT_DIR = Path("figures")
NUMERIC_COLUMNS = [
    "Rank",
    "N",
    "Fmax_eps_per_A",
    "Ln_A",
    "Lm_A",
    "Lf_A",
]
CATEGORY_COLUMNS = ["Pattern", "CATH"]


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Cannot find dataset at {path}")
    df = pd.read_csv(path)
    for column in CATEGORY_COLUMNS:
        if column in df.columns:
            df[column] = df[column].fillna("Unknown")
    return df


def summarize_dataframe(df: pd.DataFrame) -> None:
    print("===== Dataset Overview =====")
    print(f"Rows: {df.shape[0]}  Columns: {df.shape[1]}")
    print("\nMissing values per column:")
    print(df.isna().sum())
    numeric_present = [column for column in NUMERIC_COLUMNS if column in df.columns]
    if numeric_present:
        print("\nNumeric summary:")
        print(df[numeric_present].describe().round(2))
    for column in CATEGORY_COLUMNS:
        if column in df.columns:
            print(f"\nTop {column} categories:")
            print(df[column].value_counts().head(10))


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_numeric_distributions(df: pd.DataFrame) -> None:
    numeric_present = [column for column in NUMERIC_COLUMNS if column in df.columns]
    if not numeric_present:
        return
    n_cols = 3
    n_rows = math.ceil(len(numeric_present) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 3.5 * n_rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for column, axis in zip(numeric_present, axes):
        sns.histplot(df[column], ax=axis, kde=True, color="#2A9D8F", bins=20)
        axis.set_title(f"{column} distribution")
        axis.set_xlabel(column)
    for axis in axes[len(numeric_present) :]:
        axis.axis("off")
    fig.tight_layout()
    path = OUTPUT_DIR / "numeric_distributions.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved {path}")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    numeric_present = [column for column in NUMERIC_COLUMNS if column in df.columns]
    if not numeric_present:
        return
    corr = df[numeric_present].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="viridis", square=True)
    plt.title("Numeric feature correlations")
    plt.tight_layout()
    path = OUTPUT_DIR / "correlation_heatmap.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved {path}")


def plot_pairwise_relationships(df: pd.DataFrame) -> None:
    numeric_present = [column for column in NUMERIC_COLUMNS if column in df.columns]
    if not numeric_present:
        return
    subset = df[numeric_present + ["Pattern"]].copy()
    g = sns.pairplot(
        subset,
        vars=[column for column in numeric_present if column != "Rank"],
        hue="Pattern",
        corner=True,
        diag_kind="hist",
        plot_kws={"alpha": 0.7, "s": 40},
    )
    g.fig.suptitle("Pairwise relationships", y=1.02)
    path = OUTPUT_DIR / "pairplot.png"
    g.savefig(path, dpi=300)
    plt.close(g.fig)
    print(f"Saved {path}")


def plot_fmax_relationships(df: pd.DataFrame) -> None:
    target = "Fmax_eps_per_A"
    candidate_features = [
        ("N", "Fmax vs chain length"),
        ("Ln_A", "Fmax vs longest helix"),
        ("Lm_A", "Fmax vs longest strand"),
    ]
    available = [feature for feature in candidate_features if feature[0] in df.columns]
    if target not in df.columns or not available:
        return
    fig, axes = plt.subplots(1, len(available), figsize=(6 * len(available), 5), sharey=True)
    if len(available) == 1:
        axes = [axes]
    for (feature, title), axis in zip(available, axes):
        sns.regplot(
            data=df,
            x=feature,
            y=target,
            scatter_kws={"s": 60, "alpha": 0.7},
            line_kws={"color": "#264653"},
            ax=axis,
        )
        axis.set_title(title)
        axis.set_xlabel(feature)
        axis.set_ylabel(target)
    fig.suptitle("Key Fmax relationships", y=0.98)
    fig.tight_layout()
    path = OUTPUT_DIR / "fmax_relationships.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved {path}")


def plot_pattern_force(df: pd.DataFrame) -> None:
    if "Pattern" not in df.columns or "Fmax_eps_per_A" not in df.columns:
        return
    order = (
        df.groupby("Pattern")["Fmax_eps_per_A"].median().sort_values(ascending=False).index
    )
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=df,
        x="Pattern",
        y="Fmax_eps_per_A",
        order=order,
        palette="Spectral",
    )
    plt.xticks(rotation=45, ha="right")
    plt.title("Fmax_eps_per_A by pattern")
    plt.tight_layout()
    path = OUTPUT_DIR / "pattern_vs_fmax.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved {path}")


def plot_cath_force(df: pd.DataFrame) -> None:
    if "CATH" not in df.columns or "Fmax_eps_per_A" not in df.columns:
        return
    cath_counts = df["CATH"].value_counts()
    top_categories = cath_counts.head(12).index
    filtered = df[df["CATH"].isin(top_categories)].copy()
    order = (
        filtered.groupby("CATH")["Fmax_eps_per_A"].median().sort_values(ascending=False).index
    )
    plt.figure(figsize=(12, 6))
    sns.boxplot(
        data=filtered,
        x="CATH",
        y="Fmax_eps_per_A",
        order=order,
        palette="rocket",
    )
    plt.xticks(rotation=45, ha="right")
    plt.title("Fmax_eps_per_A by top CATH classes")
    plt.tight_layout()
    path = OUTPUT_DIR / "cath_vs_fmax.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved {path}")


def plot_cath_counts(df: pd.DataFrame) -> None:
    if "CATH" not in df.columns:
        return
    cath_series = df["CATH"].value_counts().head(15)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=cath_series.values, y=cath_series.index, palette="crest")
    plt.xlabel("Count")
    plt.ylabel("CATH category")
    plt.title("Most common CATH classifications")
    plt.tight_layout()
    path = OUTPUT_DIR / "cath_counts.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved {path}")


def run_analysis() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    df = load_dataset()
    summarize_dataframe(df)
    ensure_output_dir()
    plot_fmax_relationships(df)
    plot_numeric_distributions(df)
    plot_correlation_heatmap(df)
    plot_pairwise_relationships(df)
    plot_pattern_force(df)
    plot_cath_force(df)
    plot_cath_counts(df)


if __name__ == "__main__":
    run_analysis()
