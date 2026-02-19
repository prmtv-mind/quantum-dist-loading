"""
Target distribution generators for benchmarking experiments.

STATISTICAL CONCEPT:
We generate classical probability distributions that serve as "targets" for
our quantum circuits to reproduce. The quantum circuit outputs a probability
distribution (by measuring many times), and we compare it to the classical
target using various distance metrics.

TARGET DISTRIBUTIONS:
1. BINOMIAL (Symmetric): Represents number of heads in n coin flips
2. POISSON (Asymmetric): Represents rare event counts in a fixed interval
"""

import numpy as np
from typing import Tuple, Dict
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

        WHY THIS MATTERS:
        - 3 qubits → 8 possible outcomes (0 to 7)
        - 4 qubits → 16 possible outcomes (0 to 15)
        - Target distribution must be mapped to this finite space
        """
        self.n_qubits = n_qubits
        self.n_outcomes = 2 ** n_qubits
        logger.info(f"Initialized {self.__class__.__name__}: n_qubits={n_qubits}")

    def generate(self) -> np.ndarray:
        """
        Generate probability distribution.

        Returns:
            np.ndarray: Probability vector of length 2^n_qubits, sums to 1

        RETURNS:
        - 1D numpy array where p[i] = P(measuring basis state |i⟩)
        - Probabilities are normalized (sum to 1)
        - No zeros if possible (to avoid log(0) in KL divergence)
        """
        raise NotImplementedError

    def _normalize(self, probabilities: np.ndarray,
                   epsilon: float = 1e-8) -> np.ndarray:
        """
        Normalize probability distribution with smoothing.

        Args:
            probabilities: Unnormalized probability values
            epsilon: Small value to avoid log(0) in divergence calculations

        Returns:
            Normalized probability distribution

        SMOOTHING STRATEGY:
        When computing KL divergence D_KL(p||q), if any q[i]=0 while p[i]>0,
        we get log(0) = -∞. To avoid this:
        - Add epsilon to all probabilities (Laplace smoothing)
        - Renormalize so they sum to 1

        WHY EPSILON=1E-8?
        - Large enough to avoid numerical issues
        - Small enough to not significantly distort distribution
        - Standard choice in information theory applications
        """
        # Add epsilon smoothing
        smoothed = probabilities + epsilon

        # Normalize
        normalized = smoothed / smoothed.sum()

        return normalized


class BinomialDistribution(DistributionGenerator):
    """
    Binomial distribution: symmetric, represents coin flip outcomes.

    INTUITION:
    Flip a fair coin (p=0.5) n times. Count number of heads.
    Result ranges from 0 to n, with peak at n/2.

    MATHEMATICAL FORM:
    P(X=k) = C(n,k) * p^k * (1-p)^(n-k)

    QUANTUM RELEVANCE:
    Binomial is the "natural" distribution for measuring n qubits
    with uniform amplitudes. This makes it a good benchmark target.
    """

    def __init__(self, n_qubits: int, p: float = 0.5):
        """
        Initialize binomial distribution generator.

        Args:
            n_qubits: Number of qubits
            p: Probability parameter (0 < p < 1). p=0.5 gives symmetric distribution

        WHY P=0.5?
        - Symmetric distribution around mean
        - Most "natural" for quantum systems
        - Easiest to optimize for
        """
        super().__init__(n_qubits)
        self.p = p
        self.n = n_qubits  # Number of coin flips
        logger.info(f"Binomial distribution: n={self.n}, p={self.p}")

    def generate(self) -> np.ndarray:
        """
        Generate binomial probability distribution discretized to 2^n_qubits outcomes.

        IMPLEMENTATION DETAILS:
        1. Compute P(X=k) for k=0 to n_qubits using scipy
        2. For outcomes > n_qubits, assign them to the last outcome
           (This is a discretization strategy for small qubit numbers)
        3. Normalize the result
        """
        # Generate binomial probabilities for k=0 to n_qubits
        k_values = np.arange(self.n_qubits + 1)
        probs = binom.pmf(k_values, self.n, self.p)

        # Map to 2^n_qubits outcomes
        # For outcomes > n_qubits, sum them into last outcome
        outcome_probs = np.zeros(self.n_outcomes)
        outcome_probs[:len(probs)] = probs

        # Account for tail probability (should be small for n_qubits ≤ 4)
        tail_prob = 1.0 - outcome_probs.sum()
        outcome_probs[-1] += tail_prob

        # Normalize and smooth
        return self._normalize(outcome_probs)


class PoissonDistribution(DistributionGenerator):
    """
    Poisson distribution: asymmetric, represents rare event counts.

    INTUITION:
    Events happen randomly at average rate λ per time interval.
    Count number of events in one interval.
    Result ranges from 0 to ∞, skewed toward low values.

    MATHEMATICAL FORM:
    P(X=k) = (e^(-λ) * λ^k) / k!

    QUANTUM RELEVANCE:
    Asymmetric distributions are harder to load with quantum circuits.
    Poisson is a standard test for optimization difficulty.

    CHALLENGE FOR QUANTUM CIRCUITS:
    - Most quantum circuits naturally produce symmetric distributions
    - Loading asymmetric distributions requires depth and parameter tuning
    - Tests the "expressibility" of the ansatz
    """

    def __init__(self, n_qubits: int, lam: float = 1.5):
        """
        Initialize Poisson distribution generator.

        Args:
            n_qubits: Number of qubits
            lam: Lambda parameter (mean and variance). Default 1.5 gives good asymmetry

        LAMBDA CHOICES:
        - λ=1.5: Moderate asymmetry, useful test case
        - λ=2.5: More asymmetry, harder to load
        - λ=0.5: High asymmetry, very difficult
        """
        super().__init__(n_qubits)
        self.lam = lam
        logger.info(f"Poisson distribution: λ={self.lam}")

    def generate(self) -> np.ndarray:
        """
        Generate Poisson probability distribution discretized to 2^n_qubits outcomes.

        IMPLEMENTATION DETAILS:
        1. Compute P(X=k) for k=0 to 2^n_qubits using scipy
        2. Normalize the result (tail probability for k > 2^n_qubits is tiny for λ≤2.5)
        3. Apply smoothing to avoid log(0) in divergence calculations

        TRUNCATION NOTE:
        Poisson distribution has infinite support (k=0,1,2,...∞).
        We truncate at 2^n_qubits outcomes. For λ≤2.5 and n_qubits≥3,
        this captures >99.99% of probability mass.
        """
        # Generate Poisson probabilities for k=0 to n_outcomes-1
        k_values = np.arange(self.n_outcomes)
        probs = poisson.pmf(k_values, self.lam)

        # Add tail probability (small but non-zero)
        tail_prob = 1.0 - probs.sum()
        probs[-1] += tail_prob

        # Normalize and smooth
        return self._normalize(probs)


# Example usage and visualization
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Generate distributions
    n_qubits = 3
    binom_gen = BinomialDistribution(n_qubits, p=0.5)
    poisson_gen = PoissonDistribution(n_qubits, lam=1.5)

    binom_dist = binom_gen.generate()
    poisson_dist = poisson_gen.generate()

    # Print statistics
    print("Binomial Distribution (n=3, p=0.5):")
    print(f"  Probabilities: {binom_dist}")
    print(f"  Sum: {binom_dist.sum():.6f}")
    print(f"  Min: {binom_dist.min():.6f}, Max: {binom_dist.max():.6f}")

    print("\nPoisson Distribution (n=3, λ=1.5):")
    print(f"  Probabilities: {poisson_dist}")
    print(f"  Sum: {poisson_dist.sum():.6f}")
    print(f"  Min: {poisson_dist.min():.6f}, Max: {poisson_dist.max():.6f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].bar(range(2**n_qubits), binom_dist)
    axes[0].set_title("Binomial (n=3, p=0.5)")
    axes[0].set_ylabel("Probability")
    axes[0].set_xlabel("Basis State")

    axes[1].bar(range(2**n_qubits), poisson_dist)
    axes[1].set_title("Poisson (λ=1.5)")
    axes[1].set_ylabel("Probability")
    axes[1].set_xlabel("Basis State")

    plt.tight_layout()
    plt.savefig("distributions.png", dpi=150)
    print("\n✓ Saved distribution plots to distributions.png")
