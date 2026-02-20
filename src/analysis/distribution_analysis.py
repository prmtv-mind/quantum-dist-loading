"""
Analyze distribution properties and correlate with achieved fidelity.

This answers: Does distribution shape systematically predict VQC loading difficulty?
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class DistributionAnalyzer:
    """Analyze statistical properties of probability distributions."""

    @staticmethod
    def entropy(p: np.ndarray, epsilon: float = 1e-8) -> float:
        """
        Shannon entropy: H(p) = -sum(p_i * log(p_i))

        Range: [0, log(N)]
        - 0: single outcome certain
        - log(N): uniform distribution (maximum entropy)
        """
        p_smooth = p + epsilon
        return float(-np.sum(p_smooth * np.log(p_smooth)))

    @staticmethod
    def skewness(p: np.ndarray) -> float:
        """
        Skewness: measure of distribution asymmetry.

        - skew ≈ 0: symmetric
        - skew > 0: right-skewed tail
        - skew < 0: left-skewed tail
        """
        outcomes = np.arange(len(p))
        mean = np.sum(outcomes * p)
        std = np.sqrt(np.sum(((outcomes - mean) ** 2) * p))
        third_moment = np.sum(((outcomes - mean) ** 3) * p)

        if std < 1e-10:
            return 0.0

        return float(third_moment / (std ** 3))

    @staticmethod
    def kurtosis(p: np.ndarray) -> float:
        """
        Excess kurtosis: measure of peak sharpness and tail heaviness.

        - kurt ≈ 0: Gaussian-like
        - kurt > 0: sharp peak, fat tails
        - kurt < 0: flat peak, thin tails
        """
        outcomes = np.arange(len(p))
        mean = np.sum(outcomes * p)
        std = np.sqrt(np.sum(((outcomes - mean) ** 2) * p))
        fourth_moment = np.sum(((outcomes - mean) ** 4) * p)

        if std < 1e-10:
            return 0.0

        return float(fourth_moment / (std ** 4) - 3)

    @staticmethod
    def bimodality_coefficient(p: np.ndarray) -> float:
        """
        Bimodality Coefficient: indicator of multimodality.

        Threshold:
        - B < 0.555: likely unimodal
        - B ≥ 0.555: likely bimodal
        """
        n = len(p)
        skew = DistributionAnalyzer.skewness(p)
        kurt = DistributionAnalyzer.kurtosis(p)

        numerator = skew ** 2 + 1
        denominator = kurt + 3 * ((n - 1) ** 2) / ((n + 1) * (n + 3))

        if denominator < 1e-10:
            return 0.0

        return float(numerator / denominator)

    @staticmethod
    def compute_properties(p: np.ndarray, dist_name: str = "") -> Dict[str, float]:
        """Compute all properties for a distribution."""
        properties = {
            "distribution": dist_name,
            "entropy": DistributionAnalyzer.entropy(p),
            "skewness": DistributionAnalyzer.skewness(p),
            "kurtosis": DistributionAnalyzer.kurtosis(p),
            "bimodality_coeff": DistributionAnalyzer.bimodality_coefficient(p),
        }
        return properties
