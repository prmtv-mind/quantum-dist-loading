"""
Target distribution generators for benchmarking experiments.

STATISTICAL CONCEPT:
We generate classical probability distributions that serve as "targets" for
our quantum circuits to reproduce. The quantum circuit outputs a probability
distribution (by measuring many times), and we compare it to the classical
target using various distance metrics.

PAPER ALIGNMENT (from VQC_Paper_Scaffold.docx):
- All distributions discretized to 2^n_qubits probability masses
- Normalized to sum to 1
- Laplace smoothing ε=1e-8 applied to all to prevent log(0) in KL/JS divergence
- Six distributions spanning complete shape spectrum:
  1. Binomial (symmetric)
  2. Uniform (flat/maximum-entropy)
  3. Poisson (moderately asymmetric)
  4. Geometric (extremely asymmetric)
  5. Bimodal (multimodal)
  + Additional: [other symmetric]
"""

import numpy as np
from typing import Tuple, Dict, Optional
from scipy.stats import binom, poisson
import logging

logger = logging.getLogger(__name__)


class DistributionGenerator:
    """Base class for probability distribution generation."""

    def __init__(self, n_qubits: int):
        """
        Initialize distribution generator.

        Args:
            n_qubits: Number of qubits determines discretization
                      (probability space has 2^n_qubits outcomes)
        """
        self.n_qubits = n_qubits
        self.n_outcomes = 2 ** n_qubits
        logger.info(f"Initialized {self.__class__.__name__}: n_qubits={n_qubits}, n_outcomes={self.n_outcomes}")

    def generate(self) -> np.ndarray:
        """
        Generate probability distribution.

        Returns:
            np.ndarray: Probability vector of length 2^n_qubits, sums to 1

        PAPER SPEC:
        - Length: 2^n_qubits
        - Sum: 1.0 (normalized)
        - All values: > 1e-8 (Laplace smoothing applied)
        - No NaN/inf values
        """
        raise NotImplementedError

    def _normalize(self, probabilities: np.ndarray,
                   epsilon: float = 1e-8) -> np.ndarray:
        """
        Normalize probability distribution with Laplace smoothing.

        Args:
            probabilities: Unnormalized probability values
            epsilon: Smoothing constant (paper: 1e-8)

        Returns:
            Normalized probability distribution

        PAPER JUSTIFICATION (from scaffold):
        Laplace smoothing prevents log(0) errors in KL divergence and
        Jensen-Shannon computations. Standard choice: ε=1e-8.
        """
        # Add epsilon smoothing
        smoothed = probabilities + epsilon

        # Normalize
        normalized = smoothed / smoothed.sum()

        return normalized


class BinomialDistribution(DistributionGenerator):
    """
    Binomial distribution: symmetric, serves as the baseline case.

    PAPER SPEC:
    Binomial(n_qubits, p=0.5): Uses scipy.stats.binom.pmf evaluated at
    k=0,...,n_qubits, then zero-padded to 2^n_qubits outcomes.
    Symmetric bell shape with peak at n_qubits/2.
    """

    def __init__(self, n_qubits: int, p: float = 0.5):
        super().__init__(n_qubits)
        self.p = p
        logger.info(f"Binomial distribution: n={self.n_qubits}, p={self.p}")

    def generate(self) -> np.ndarray:
        """
        Generate binomial probability distribution.

        PAPER IMPLEMENTATION:
        1. Compute P(X=k) for k=0 to n_qubits using scipy.stats.binom.pmf
        2. Zero-pad to 2^n_qubits outcomes
        3. Add tail probability to last outcome
        4. Normalize with ε=1e-8 smoothing
        """
        # Generate binomial probabilities for k=0 to n_qubits
        k_values = np.arange(self.n_qubits + 1)
        probs = binom.pmf(k_values, self.n_qubits, self.p)

        # Map to 2^n_qubits outcomes
        outcome_probs = np.zeros(self.n_outcomes)
        outcome_probs[:len(probs)] = probs

        # Account for tail probability
        tail_prob = 1.0 - outcome_probs.sum()
        outcome_probs[-1] += tail_prob

        # Normalize and smooth per paper spec
        return self._normalize(outcome_probs, epsilon=1e-8)


class UniformDistribution(DistributionGenerator):
    """
    Uniform distribution: flat, maximum entropy.

    PAPER SPEC:
    Uniform: np.ones(2^n_qubits) / (2^n_qubits). Trivially normalized.
    Circuit must learn to produce equal probability for all outcomes,
    suppressing all amplitude variation.
    """

    def generate(self) -> np.ndarray:
        """
        Generate uniform distribution over all outcomes.

        IMPLEMENTATION (per paper):
        - All outcomes equally likely: 1/2^n_qubits
        - Applied smoothing for consistency with other distributions
        """
        probs = np.ones(self.n_outcomes) / self.n_outcomes
        return self._normalize(probs, epsilon=1e-8)


class PoissonDistribution(DistributionGenerator):
    """
    Poisson distribution: moderately asymmetric.

    PAPER SPEC:
    λ ∈ {1.5, 2.5}. Discretization: Truncate to 2^n_qubits outcomes.
    Normalization: Sum to 1. Note: Requires smoothing for zero probabilities.
    """

    def __init__(self, n_qubits: int, lam: float = 1.5):
        super().__init__(n_qubits)
        self.lam = lam
        logger.info(f"Poisson distribution: λ={self.lam}")

    def generate(self) -> np.ndarray:
        """
        Generate Poisson probability distribution discretized to 2^n_qubits outcomes.
        """
        # Generate Poisson probabilities for k=0 to n_outcomes-1
        k_values = np.arange(self.n_outcomes)
        probs = poisson.pmf(k_values, self.lam)

        # Add tail probability
        tail_prob = 1.0 - probs.sum()
        probs[-1] += tail_prob

        # Normalize and smooth
        return self._normalize(probs, epsilon=1e-8)


class GeometricDistribution(DistributionGenerator):
    """
    Geometric distribution: extremely asymmetric, monotone decreasing.

    PAPER SPEC:
    Extreme single-mode asymmetry. Heavy mass at k=1, exponential tail decay.
    Represents processes with no "memory" — longer wait = less likely.

    QUANTUM CHALLENGE:
    Extreme asymmetry requires asymmetric amplitude envelope. Local gates
    (nearest-neighbor) have limited long-range influence.
    """

    def __init__(self, n_qubits: int, p: float = 0.5):
        super().__init__(n_qubits)
        self.p = p
        logger.info(f"Geometric distribution: p={self.p}")

    def generate(self) -> np.ndarray:
        """
        Generate geometric distribution.

        IMPLEMENTATION (per paper):
        1. Compute P(X=k) = (1-p)^(k-1) * p for k=1,...,2^n_qubits
        2. Map to array indices 0,...,2^n_qubits-1
        3. Add tail probability
        4. Normalize
        """
        k_values = np.arange(1, self.n_outcomes + 1)
        probs = (1 - self.p) ** (k_values - 1) * self.p

        # Account for tail probability
        tail_prob = 1.0 - probs.sum()
        probs[-1] += tail_prob

        return self._normalize(probs, epsilon=1e-8)


class BimodalDistribution(DistributionGenerator):
    """
    Bimodal distribution: two separated peaks, multimodal structure.

    PAPER SPEC:
    BimodalDistribution must accept mu1, mu2, sigma as constructor arguments
    with defaults mu1=N//4, mu2=3N//4, sigma=0.8.

    QUANTUM CHALLENGE (Extreme):
    Two separated probability peaks require circuit to maintain constructive
    interference at two distant regions while destructively interfering between them.
    """

    def __init__(self,
                 n_qubits: int,
                 mu1: Optional[int] = None,
                 mu2: Optional[int] = None,
                 sigma: float = 0.8):
        """
        Initialize bimodal distribution with type-safe parameter handling.

        Args:
            n_qubits: Number of qubits
            mu1: Mean of first Gaussian (default: N//4)
            mu2: Mean of second Gaussian (default: 3*N//4)
            sigma: Width of Gaussians (default: 0.8)
        """
        super().__init__(n_qubits)

        # Type-safe defaults (fix for Pylance errors)
        self.mu1: int = mu1 if mu1 is not None else (self.n_outcomes // 4)
        self.mu2: int = mu2 if mu2 is not None else (3 * self.n_outcomes // 4)
        self.sigma: float = sigma

        logger.info(
            f"Bimodal distribution: μ₁={self.mu1}, μ₂={self.mu2}, σ={self.sigma}"
        )

    def generate(self) -> np.ndarray:
        """
        Generate bimodal distribution as mixture of two Gaussians.

        IMPLEMENTATION (per paper):
        1. Create array k = 0, 1, ..., 2^n_qubits - 1
        2. Compute Gaussian1 = exp(-(k-μ₁)²/(2σ²))
        3. Compute Gaussian2 = exp(-(k-μ₂)²/(2σ²))
        4. Sum and normalize with ε=1e-8 smoothing
        """
        k_values = np.arange(self.n_outcomes, dtype=float)

        # Two Gaussian bumps
        gaussian1 = np.exp(-((k_values - self.mu1) ** 2) / (2 * self.sigma ** 2))
        gaussian2 = np.exp(-((k_values - self.mu2) ** 2) / (2 * self.sigma ** 2))

        # Combine
        probs = gaussian1 + gaussian2

        # Normalize and smooth
        return self._normalize(probs, epsilon=1e-8)


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    n_qubits = 3

    # Generate all distributions
    distributions = {
        "Binomial(p=0.5)": BinomialDistribution(n_qubits),
        "Uniform": UniformDistribution(n_qubits),
        "Poisson(λ=1.5)": PoissonDistribution(n_qubits, lam=1.5),
        "Geometric(p=0.5)": GeometricDistribution(n_qubits),
        "Bimodal": BimodalDistribution(n_qubits),
    }

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    for ax, (name, gen) in zip(axes, distributions.items()):
        dist = gen.generate()
        ax.bar(range(2**n_qubits), dist, color='steelblue', alpha=0.7, edgecolor='black')
        ax.set_title(name, fontsize=11, fontweight='bold')
        ax.set_ylabel("Probability")
        ax.set_xlabel("Basis State")
        ax.set_ylim([0, max([gen.generate().max() for gen in distributions.values()]) * 1.1])

        # Print statistics
        print(f"\n{name}:")
        print(f"  Min: {dist.min():.8f}, Max: {dist.max():.8f}")
        print(f"  Sum: {dist.sum():.10f}")
        print(f"  All > 1e-8: {(dist > 1e-8).all()}")

    plt.tight_layout()
    plt.savefig("all_distributions.png", dpi=150, bbox_inches='tight')
    print("\n✓ Saved distribution plots to all_distributions.png")
