"""Distribution fidelity and distance metrics."""

from typing import Dict, Any, Tuple, cast
import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp


class DistributionMetrics:
    """Compute metrics between probability distributions."""

    @staticmethod
    def l2_distance(p: np.ndarray, q: np.ndarray) -> float:
        """
        L2 (Euclidean) distance between distributions.

        DEFINITION:
        d_L2(p, q) = sqrt(sum((p[i] - q[i])^2))

        RANGE: [0, sqrt(2)] for binary distributions
        INTERPRETATION: 0 = identical, higher = more different

        Args:
            p: Target distribution
            q: Generated distribution

        Returns:
            L2 distance value
        """
        p = np.asarray(p, dtype=float)
        q = np.asarray(q, dtype=float)
        return float(np.linalg.norm(p - q))

    @staticmethod
    def kl_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> float:
        """
        Kullback-Leibler divergence D(p||q).

        DEFINITION:
        D_KL(p||q) = Σ p[i] * log(p[i]/q[i])

        RANGE: [0, ∞)
        INTERPRETATION: 0 = identical, higher = more different
        ASYMMETRIC: D_KL(p||q) ≠ D_KL(q||p)

        Args:
            p: Target distribution
            q: Generated distribution
            epsilon: Small value to avoid log(0)

        Returns:
            KL divergence value
        """
        p = np.asarray(p, dtype=float)
        q = np.asarray(q, dtype=float)

        # Add epsilon to avoid log(0)
        p_safe = np.clip(p, epsilon, 1)
        q_safe = np.clip(q, epsilon, 1)

        kl_value: float = float(np.sum(p_safe * np.log(p_safe / q_safe)))
        return kl_value

    @staticmethod
    def jensen_shannon(p: np.ndarray, q: np.ndarray) -> float:
        """
        Jensen-Shannon divergence (symmetric version of KL).

        DEFINITION:
        JS(p||q) = 0.5 * D_KL(p||m) + 0.5 * D_KL(q||m)
        where m = (p+q)/2

        RANGE: [0, 1]
        INTERPRETATION: 0 = identical, 1 = completely different
        SYMMETRIC: JS(p||q) = JS(q||p)

        Args:
            p: Target distribution
            q: Generated distribution

        Returns:
            Jensen-Shannon divergence value
        """
        p = np.asarray(p, dtype=float)
        q = np.asarray(q, dtype=float)

        js_value: float = float(jensenshannon(p, q))
        return js_value

    @staticmethod
    def ks_statistic(p: np.ndarray, q: np.ndarray) -> Tuple[float, float]:
        """
        Kolmogorov-Smirnov test statistic and p-value.

        DEFINITION:
        KS = max|F_p(x) - F_q(x)|
        where F are cumulative distribution functions

        RANGE: [0, 1]
        INTERPRETATION: 0 = identical CDFs, 1 = completely different

        Args:
            p: Target distribution
            q: Generated distribution

        Returns:
            Tuple of (KS statistic, p-value)
        """
        p = np.asarray(p, dtype=float)
        q = np.asarray(q, dtype=float)

        # Use cast() to handle scipy type stub issue
        # scipy.stats.ks_2samp returns tuple but type stubs are imprecise
        result: Tuple[Any, Any] = cast(Tuple[Any, Any], ks_2samp(p, q))

        return float(result[0]), float(result[1])

    @staticmethod
    def fidelity(p: np.ndarray, q: np.ndarray) -> float:
        """
        Quantum fidelity F(p, q) = (Σ √(p_i * q_i))^2.

        Measures overlap between distributions.

        DEFINITION:
        F = (Σ_i √(p_i * q_i))^2

        RANGE: [0, 1]
        INTERPRETATION: 1 = identical, 0 = orthogonal

        Args:
            p: Target distribution
            q: Generated distribution

        Returns:
            Fidelity value in [0, 1]
        """
        p = np.asarray(p, dtype=float)
        q = np.asarray(q, dtype=float)

        # Ensure non-negative for sqrt
        p_safe = np.clip(p, 0, 1)
        q_safe = np.clip(q, 0, 1)

        fidelity_value: float = float(np.sum(np.sqrt(p_safe * q_safe)) ** 2)
        return np.clip(fidelity_value, 0, 1)

    @staticmethod
    def hellinger_distance(p: np.ndarray, q: np.ndarray) -> float:
        """
        Hellinger distance between distributions.

        DEFINITION:
        H(p, q) = sqrt(0.5 * Σ (√p_i - √q_i)^2)

        RANGE: [0, 1]
        INTERPRETATION: 0 = identical, 1 = orthogonal

        Args:
            p: Target distribution
            q: Generated distribution

        Returns:
            Hellinger distance value
        """
        p = np.asarray(p, dtype=float)
        q = np.asarray(q, dtype=float)

        p_safe = np.clip(p, 0, 1)
        q_safe = np.clip(q, 0, 1)

        hellinger: float = float(
            np.sqrt(0.5 * np.sum((np.sqrt(p_safe) - np.sqrt(q_safe)) ** 2))
        )
        return np.clip(hellinger, 0, 1)

    @staticmethod
    def total_variation_distance(p: np.ndarray, q: np.ndarray) -> float:
        """
        Total variation distance: TV(p, q) = 0.5 * Σ |p_i - q_i|.

        DEFINITION:
        TV(p, q) = 0.5 * Σ_i |p_i - q_i|

        RANGE: [0, 1]
        INTERPRETATION: 0 = identical, 1 = disjoint support

        Args:
            p: Target distribution
            q: Generated distribution

        Returns:
            Total variation distance
        """
        p = np.asarray(p, dtype=float)
        q = np.asarray(q, dtype=float)

        tv_distance: float = float(0.5 * np.sum(np.abs(p - q)))
        return tv_distance

    @staticmethod
    def compute_all_metrics(p: np.ndarray, q: np.ndarray) -> Dict[str, float]:
        """
        Compute all available metrics between two distributions.

        Used for comprehensive benchmarking as per manuscript:
        "Systematic Benchmarking of Direct Variational Quantum Circuits"

        Args:
            p: Target distribution (should sum to 1)
            q: Generated distribution (should sum to 1)

        Returns:
            Dictionary with all metric values

        Metrics:
        - l2_distance: Euclidean distance (primary cost function)
        - kl_divergence: Information-theoretic divergence
        - jensen_shannon: Symmetric KL divergence
        - ks_statistic: Maximum CDF distance
        - ks_pvalue: Statistical significance of KS test
        - fidelity: Quantum overlap measure
        - hellinger_distance: Alternative symmetric distance
        - total_variation: Measure of support difference
        """
        p = np.asarray(p, dtype=float)
        q = np.asarray(q, dtype=float)

        # Normalize to probability distributions
        p_norm: np.ndarray = p / (np.sum(p) + 1e-10)
        q_norm: np.ndarray = q / (np.sum(q) + 1e-10)

        # Compute KS statistic explicitly
        ks_stat, ks_pval = DistributionMetrics.ks_statistic(p_norm, q_norm)

        metrics: Dict[str, float] = {
            "l2_distance": DistributionMetrics.l2_distance(p_norm, q_norm),
            "kl_divergence": DistributionMetrics.kl_divergence(p_norm, q_norm),
            "jensen_shannon": DistributionMetrics.jensen_shannon(p_norm, q_norm),
            "ks_statistic": ks_stat,
            "ks_pvalue": ks_pval,
            "fidelity": DistributionMetrics.fidelity(p_norm, q_norm),
            "hellinger_distance": DistributionMetrics.hellinger_distance(p_norm, q_norm),
            "total_variation": DistributionMetrics.total_variation_distance(p_norm, q_norm),
        }

        return metrics
