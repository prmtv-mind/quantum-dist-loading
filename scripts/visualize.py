"""Generate publication-quality visualizations."""

import sys
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logging import setup_logger

logger = setup_logger("visualize", level="INFO")

plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


def load_summary(mode: str = "demo") -> Optional[pd.DataFrame]:
    """
    Load summary CSV.

    Returns:
        DataFrame or None if file not found
    """
    summary_path = project_root / "results" / f"summary_{mode}.csv"
    if not summary_path.exists():
        logger.error(f"Summary not found: {summary_path}")
        return None

    df = pd.read_csv(summary_path)
    logger.info(f"Loaded summary: {len(df)} configurations")
    return df


def plot_distribution_difficulty(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot 1: Fidelity by distribution type."""
    df_best = df.loc[df.groupby("distribution")["fidelity_mean"].idxmax()]

    fig, ax = plt.subplots(figsize=(12, 6))

    distributions = sorted(df_best["distribution"].unique())
    circuits = sorted(df_best["circuit"].unique())

    x = np.arange(len(distributions))
    width = 0.25

    for i, circuit in enumerate(circuits):
        subset = df_best[df_best["circuit"] == circuit]
        subset_reindex = subset.set_index("distribution").reindex(distributions)

        means = subset_reindex["fidelity_mean"].fillna(0).values
        stds = subset_reindex["fidelity_std"].fillna(0).values

        ax.bar(
            x + i * width,
            means,
            width,
            label=circuit.capitalize(),
            alpha=0.8,
            yerr=stds,
            capsize=5
        )

    ax.set_ylabel("Mean Fidelity", fontsize=12)
    ax.set_xlabel("Target Distribution", fontsize=12)
    ax.set_title("VQC Distribution Loading Difficulty by Type", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels([d.replace("_", " ").title() for d in distributions], rotation=45)
    ax.legend()
    # FIX: Use tuple instead of list
    ax.set_ylim((0, 1.05))
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "distribution_difficulty.png", dpi=300, bbox_inches="tight")
    logger.info("✓ Saved distribution_difficulty.png")
    plt.close()


def plot_depth_scaling(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot 2: Fidelity vs circuit depth."""
    distributions = sorted(df["distribution"].unique())
    n_dist = len(distributions)
    n_cols = 3
    n_rows = (n_dist + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten()

    for ax, dist_name in zip(axes, distributions):
        dist_data = df[df["distribution"] == dist_name]

        for circuit in sorted(dist_data["circuit"].unique()):
            circuit_data = dist_data[dist_data["circuit"] == circuit]
            circuit_data = circuit_data.sort_values("depth")

            ax.errorbar(
                circuit_data["depth"],
                circuit_data["fidelity_mean"],
                yerr=circuit_data["fidelity_std"],
                label=circuit.capitalize(),
                marker="o",
                linewidth=2,
                markersize=8,
                alpha=0.8,
                capsize=5
            )

        ax.set_xlabel("Circuit Depth", fontsize=11)
        ax.set_ylabel("Mean Fidelity", fontsize=11)
        ax.set_title(dist_name.replace("_", " ").title(), fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        # FIX: Use tuple instead of list
        ax.set_ylim((0, 1.05))
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for ax in axes[n_dist:]:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / "depth_scaling.png", dpi=300, bbox_inches="tight")
    logger.info("✓ Saved depth_scaling.png")
    plt.close()


def plot_architecture_comparison(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot 3: Architecture vs Distribution heatmap."""
    agg_df = df.groupby(["distribution", "circuit"]).agg({
        "fidelity_mean": "mean"
    }).reset_index()

    pivot = agg_df.pivot(index="circuit", columns="distribution", values="fidelity_mean")

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Mean Fidelity"},
        ax=ax
    )

    ax.set_title("Architecture Performance Across Distributions", fontsize=14, fontweight="bold")
    ax.set_xlabel("Distribution", fontsize=12)
    ax.set_ylabel("Circuit Architecture", fontsize=12)

    plt.tight_layout()
    plt.savefig(output_dir / "architecture_comparison.png", dpi=300, bbox_inches="tight")
    logger.info("✓ Saved architecture_comparison.png")
    plt.close()


def plot_optimizer_comparison(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot 4: COBYLA vs SPSA comparison."""
    optimizers = df["optimizer"].unique()
    if len(optimizers) < 2:
        logger.warning("Not all optimizers present; skipping optimizer comparison")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Fidelity
    opt_data = df.groupby("optimizer").agg({
        "fidelity_mean": ["mean", "std"]
    }).reset_index()

    opt_names = opt_data["optimizer"].values
    opt_means = opt_data[("fidelity_mean", "mean")].values
    opt_stds = opt_data[("fidelity_mean", "std")].values

    axes[0].bar(opt_names, opt_means, yerr=opt_stds, alpha=0.7, capsize=5, color=['steelblue', 'coral'])
    axes[0].set_ylabel("Mean Fidelity", fontsize=12)
    axes[0].set_title("Final Fidelity: COBYLA vs SPSA", fontsize=12, fontweight="bold")
    axes[0].set_ylim((0, 1))
    axes[0].grid(axis="y", alpha=0.3)

    # Iterations
    opt_iters = df.groupby("optimizer").agg({
        "iterations_mean": ["mean", "std"]
    }).reset_index()

    iter_names = opt_iters["optimizer"].values
    iter_means = opt_iters[("iterations_mean", "mean")].values
    iter_stds = opt_iters[("iterations_mean", "std")].values

    axes[1].bar(iter_names, iter_means, yerr=iter_stds, alpha=0.7, capsize=5, color=['steelblue', 'coral'])
    axes[1].set_ylabel("Mean Iterations to Convergence", fontsize=12)
    axes[1].set_title("Convergence Speed: COBYLA vs SPSA", fontsize=12, fontweight="bold")
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "optimizer_comparison.png", dpi=300, bbox_inches="tight")
    logger.info("✓ Saved optimizer_comparison.png")
    plt.close()


def main(mode: str = "demo") -> None:
    """Generate all visualizations."""
    logger.info("="*70)
    logger.info("GENERATING VISUALIZATIONS")
    logger.info("="*70)

    df = load_summary(mode=mode)
    if df is None:
        logger.error("Cannot generate visualizations without summary data")
        return

    output_dir = project_root / "results" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_distribution_difficulty(df, output_dir)
    plot_depth_scaling(df, output_dir)
    plot_architecture_comparison(df, output_dir)
    plot_optimizer_comparison(df, output_dir)

    logger.info(f"\n✓ All visualizations saved to {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate visualizations from experiment results")
    parser.add_argument("--mode", choices=["demo", "full"], default="full")
    args = parser.parse_args()

    main(mode=args.mode)
