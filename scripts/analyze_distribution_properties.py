"""
Analyze distribution properties and correlate with fidelity.

Computes:
- Shannon entropy
- Skewness
- Kurtosis
- Bimodality coefficient

Then correlates with achieved mean fidelity to identify patterns.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import entropy, skew, kurtosis
import logging

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.distributions.generators import (
    BinomialDistribution,
    UniformDistribution,
    PoissonDistribution,
    GeometricDistribution,
    BimodalDistribution,
)
from src.utils.logging import setup_logger

logger = setup_logger("analyze_distributions", level="INFO")


def compute_distribution_properties() -> pd.DataFrame:
    """
    Compute statistical properties of all target distributions.

    Returns:
        DataFrame with properties for each distribution
    """
    distributions = {
        "binomial": BinomialDistribution(n_qubits=4),
        "uniform": UniformDistribution(n_qubits=4),
        "poisson_1.5": PoissonDistribution(n_qubits=4, lam=1.5),
        "poisson_2.5": PoissonDistribution(n_qubits=4, lam=2.5),
        "geometric": GeometricDistribution(n_qubits=4),
        "bimodal": BimodalDistribution(n_qubits=4),
    }

    properties = []

    for dist_name, dist_gen in distributions.items():
        p = dist_gen.generate()

        # Shannon entropy: H(p) = -Σ p_i log(p_i)
        shannon_entropy = entropy(p)

        # Skewness: measure of asymmetry
        dist_skewness = float(skew(p))

        # Excess kurtosis: measure of tail heaviness
        dist_kurtosis = float(kurtosis(p))

        # Bimodality coefficient (from Hartigan & Hartigan 1985)
        # BC = (skewness^2 + 1) / (kurtosis + 3 * (n-1)^2 / (n-2) / (n-3))
        # Bimodal when BC > 0.555
        n = len(p)
        if n > 3:
            bc = (dist_skewness**2 + 1) / (dist_kurtosis + 3 * (n - 1)**2 / ((n - 2) * (n - 3)))
        else:
            bc = np.nan

        # Max probability (concentration)
        max_prob = float(np.max(p))

        # Number of significant modes (prob > 0.1 * max)
        n_modes = int(np.sum(p > 0.1 * max_prob))

        properties.append({
            "distribution": dist_name,
            "shannon_entropy": shannon_entropy,
            "skewness": dist_skewness,
            "kurtosis": dist_kurtosis,
            "bimodality_coeff": bc,
            "max_probability": max_prob,
            "n_significant_modes": n_modes,
        })

        logger.info(f"{dist_name}:")
        logger.info(f"  H(p) = {shannon_entropy:.4f}")
        logger.info(f"  Skewness = {dist_skewness:.4f}")
        logger.info(f"  Kurtosis = {dist_kurtosis:.4f}")
        logger.info(f"  BC = {bc:.4f} {'(bimodal)' if bc > 0.555 else '(unimodal)'}")

    return pd.DataFrame(properties)


def correlate_with_fidelity(mode: str = "demo"):
    """
    Correlate distribution properties with achieved mean fidelity.

    Merges distribution properties with summary results and computes
    Pearson/Spearman correlations.
    """
    # Load summary
    summary_path = project_root / "results" / f"summary_{mode}.csv"
    if not summary_path.exists():
        logger.error(f"Summary not found: {summary_path}")
        return

    summary_df = pd.read_csv(summary_path)

    # Compute properties
    properties_df = compute_distribution_properties()

    # Merge on distribution name
    merged = summary_df.merge(properties_df, on="distribution")

    # Compute correlations
    from scipy.stats import pearsonr, spearmanr

    properties_cols = [
        "shannon_entropy",
        "skewness",
        "kurtosis",
        "bimodality_coeff",
        "max_probability",
    ]

    logger.info("\n" + "="*70)
    logger.info("CORRELATION ANALYSIS: Distribution Properties vs Fidelity")
    logger.info("="*70)

    for prop in properties_cols:
        # Remove NaN values
        valid = merged[[prop, "fidelity_mean"]].dropna()

        if len(valid) < 2:
            continue

        pearson_r, pearson_p = pearsonr(valid[prop], valid["fidelity_mean"])
        spearman_r, spearman_p = spearmanr(valid[prop], valid["fidelity_mean"])

        logger.info(f"\n{prop}:")
        logger.info(f"  Pearson:  r = {pearson_r:+.4f}, p = {pearson_p:.4f}")
        logger.info(f"  Spearman: r = {spearman_r:+.4f}, p = {spearman_p:.4f}")

    # Save detailed analysis
    output_path = project_root / "results" / f"distribution_analysis_{mode}.csv"
    merged.to_csv(output_path, index=False)
    logger.info(f"\n✓ Saved detailed analysis to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["demo", "full"], default="demo")
    args = parser.parse_args()

    logger.info("Computing distribution properties...")
    props = compute_distribution_properties()

    logger.info("\nCorrelating with fidelity...")
    correlate_with_fidelity(mode=args.mode)
