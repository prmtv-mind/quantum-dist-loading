"""Aggregate individual run results into summary statistics."""

import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logging import setup_logger

logger = setup_logger("aggregate_results", level="INFO")


def load_results(mode: str = "demo") -> List[Dict]:
    """Load all JSON result files for a given mode."""
    results_dir = project_root / "results" / "raw" / mode

    if not results_dir.exists():
        logger.error(f"Results directory not found: {results_dir}")
        return []

    results = []
    json_files = list(results_dir.glob("*.json"))
    logger.info(f"Found {len(json_files)} result files")

    for json_file in sorted(json_files):
        try:
            with open(json_file) as f:
                results.append(json.load(f))
        except Exception as e:
            logger.warning(f"Failed to load {json_file}: {e}")

    logger.info(f"Successfully loaded {len(results)} results")
    return results


def aggregate_results(results: List[Dict]) -> pd.DataFrame:
    """
    Aggregate results across seeds.

    Groups by configuration (distribution, circuit, depth, n_qubits, optimizer),
    computes mean and std of all metrics across random seeds.
    """
    df = pd.DataFrame(results)

    # Filter successes
    df = df[df["status"] == "success"].copy()
    logger.info(f"Aggregating {len(df)} successful runs")

    # Metrics to aggregate
    metrics = [
        "l2_distance",
        "kl_divergence",
        "jensen_shannon",
        "ks_statistic",
        "fidelity",
        "initial_cost",
        "final_cost",
        "iterations",
    ]

    groupby_cols = ["distribution", "circuit", "depth", "n_qubits", "optimizer"]

    # Build aggregation dictionary - FIX: use list for all aggregations
    agg_dict: Dict[str, List[str]] = {}
    for metric in metrics:
        agg_dict[metric] = ["mean", "std"]

    # Add count (number of seeds)
    agg_dict["seed"] = ["count"]

    # Perform aggregation
    summary = df.groupby(groupby_cols).agg(agg_dict).reset_index()

    # Flatten multi-level columns
    summary.columns = [
        "_".join(col).rstrip("_") if col[1] else col[0]
        for col in summary.columns.values
    ]

    # Rename count column for clarity
    summary = summary.rename(columns={"seed_count": "n_seeds"})

    logger.info(f"Created summary with {len(summary)} configurations")

    return summary


def save_summary(summary: pd.DataFrame, mode: str = "demo") -> Path:
    """Save summary to CSV."""
    output_path = project_root / "results" / f"summary_{mode}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    logger.info(f"✓ Saved summary to {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Aggregate experiment results")
    parser.add_argument("--mode", choices=["demo", "full"], default="demo")
    args = parser.parse_args()

    # Load and aggregate
    results = load_results(mode=args.mode)
    if not results:
        logger.error("No results to aggregate")
        sys.exit(1)

    summary = aggregate_results(results)
    save_summary(summary, mode=args.mode)

    # Display summary statistics
    logger.info("\n" + "="*70)
    logger.info("SUMMARY STATISTICS")
    logger.info("="*70)
    logger.info(f"Total configurations: {len(summary)}")
    logger.info(f"Mean fidelity: {summary['fidelity_mean'].mean():.4f} ± {summary['fidelity_mean'].std():.4f}")
    logger.info(f"Best fidelity: {summary['fidelity_mean'].max():.4f}")
    logger.info(f"Worst fidelity: {summary['fidelity_mean'].min():.4f}")

    # Display top configurations
    logger.info("\nTop 10 configurations (highest fidelity):")
    top = summary.nlargest(10, "fidelity_mean")[[
        "distribution", "circuit", "depth", "optimizer", "fidelity_mean"
    ]]
    print(top.to_string(index=False))
