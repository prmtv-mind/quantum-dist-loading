"""
Generate LaTeX tables for paper from summary results.

Outputs:
- results/table_1_distribution_difficulty.tex
- results/table_2_architecture_performance.tex
- results/table_3_optimizer_comparison.tex
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logging import setup_logger

logger = setup_logger("generate_tables")


def generate_table_1_distribution_difficulty(summary: pd.DataFrame, output_path: Path):
    """
    Table 1: Distribution Loading Difficulty

    Columns: Distribution | Best Depth | Mean Fidelity ± Std | Ranks difficulty
    """

    # Get best depth per distribution
    best_by_dist = summary.loc[summary.groupby("distribution")["fidelity_mean"].idxmax()]
    best_by_dist = best_by_dist.sort_values("fidelity_mean", ascending=False)

    # Create table
    table_data = []
    for rank, (idx, row) in enumerate(best_by_dist.iterrows(), 1):
        dist_name = row["distribution"].replace("_", "\\_")
        depth = int(row["depth"])
        fidelity = row["fidelity_mean"]
        std = row["fidelity_std"] if pd.notna(row["fidelity_std"]) else 0.0

        table_data.append({
            "Rank": rank,
            "Distribution": dist_name,
            "Depth": depth,
            "Fidelity": f"{fidelity:.4f}",
            "Std Dev": f"{std:.4f}",
        })

    df_table = pd.DataFrame(table_data)

    # Generate LaTeX
    latex = r"""
\begin{table}[h]
\centering
\caption{Distribution Loading Difficulty Ranking. Each row shows the best-achieved
fidelity for a given target distribution across all circuit depths, architectures,
and optimizers. Distributions are ranked by achieved fidelity; higher fidelity
indicates easier loading for the VQC.}
\label{tab:distribution_difficulty}
\begin{tabular}{c l c c c}
\toprule
\textbf{Rank} & \textbf{Distribution} & \textbf{Depth} &
\textbf{Mean Fidelity} & \textbf{Std Dev} \\
\midrule
"""

    for _, row in df_table.iterrows():
        latex += (f"{row['Rank']} & {row['Distribution']} & {row['Depth']} & "
                 f"{row['Fidelity']} & {row['Std Dev']} \\\\\n")

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""

    with open(output_path, "w") as f:
        f.write(latex)

    logger.info(f"✓ Generated Table 1: {output_path}")


def generate_table_2_architecture_performance(summary: pd.DataFrame, output_path: Path):
    """
    Table 2: Architecture Performance Across Distributions

    Matrix: Rows=Architectures, Cols=Distributions, Values=Mean Fidelity
    """

    # Pivot
    pivot = summary.pivot_table(
        index="circuit",
        columns="distribution",
        values="fidelity_mean",
        aggfunc="mean"
    )

    # Generate LaTeX
    latex = r"""
\begin{table}[h]
\centering
\caption{Circuit Architecture Performance. Each cell shows mean fidelity achieved
by a circuit architecture on a given distribution type, averaged across depths,
optimizers, and random seeds. Rows are circuit types; columns are target distributions.}
\label{tab:architecture_performance}
\begin{tabular}{l | """ + " c " * len(pivot.columns) + r"""}
\toprule
\textbf{Architecture}"""

    # Header row
    for col in pivot.columns:
        latex += f" & {col.replace('_', '\\\\_')}"

    latex += r""" \\
\midrule
"""

    # Data rows
    for arch in pivot.index:
        latex += f"{arch}"
        for dist in pivot.columns:
            val = pivot.loc[arch, dist]
            if pd.notna(val):
                latex += f" & {val:.4f}"
            else:
                latex += " & —"
        latex += r" \\" + "\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""

    with open(output_path, "w") as f:
        f.write(latex)

    logger.info(f"✓ Generated Table 2: {output_path}")


def generate_table_3_optimizer_comparison(summary: pd.DataFrame, output_path: Path):
    """
    Table 3: Optimizer Comparison

    Columns: Optimizer | Mean Fidelity | Mean Iterations | Convergence Std Dev
    """

    if "optimizer" not in summary.columns:
        logger.warning("Optimizer column not in summary; skipping Table 3")
        return

    opt_stats = summary.groupby("optimizer").agg({
        "fidelity_mean": ["mean", "std"],
        "iterations_mean": ["mean", "std"],
    }).reset_index()

    opt_stats.columns = ["optimizer", "fid_mean", "fid_std", "iter_mean", "iter_std"]

    # Generate LaTeX
    latex = r"""
\begin{table}[h]
\centering
\caption{Optimizer Performance Comparison. COBYLA and SPSA are compared on
mean fidelity achieved (lower is better for cost), average iterations to
convergence, and consistency (fidelity standard deviation across all configurations).}
\label{tab:optimizer_comparison}
\begin{tabular}{l c c c c}
\toprule
\textbf{Optimizer} & \textbf{Mean Fidelity} & \textbf{Fid. Std Dev} &
\textbf{Mean Iter.} & \textbf{Iter. Std Dev} \\
\midrule
"""

    for _, row in opt_stats.iterrows():
        latex += (f"{row['optimizer']} & {row['fid_mean']:.4f} & {row['fid_std']:.4f} & "
                 f"{row['iter_mean']:.1f} & {row['iter_std']:.1f} \\\\\n")

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""

    with open(output_path, "w") as f:
        f.write(latex)

    logger.info(f"✓ Generated Table 3: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["demo", "full"], default="full")
    args = parser.parse_args()

    # Load summary
    summary_path = project_root / "results" / f"summary_{args.mode}.csv"
    summary = pd.read_csv(summary_path)

    logger.info("="*70)
    logger.info("GENERATING LATEX TABLES")
    logger.info("="*70)

    output_dir = project_root / "results" / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    generate_table_1_distribution_difficulty(summary, output_dir / "table_1.tex")
    generate_table_2_architecture_performance(summary, output_dir / "table_2.tex")
    generate_table_3_optimizer_comparison(summary, output_dir / "table_3.tex")

    logger.info(f"\n✓ All tables saved to {output_dir}")
    logger.info("\nPaste into paper using \\input{path/to/table_X.tex}")
