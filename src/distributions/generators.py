"""
Target distribution generators for benchmarking experiments.

PAPER ALIGNMENT (from VQC_Paper_Scaffold.docx and assessment_report.docx):
- All distributions discretized to 2^n_qubits probability masses
- Normalized to sum to 1
- Laplace smoothing ε=1e-8 applied to all to prevent log(0) in KL/JS divergence
- Six distributions spanning complete shape spectrum:
  1. Binomial (basis-sparse — NOT multimodal; see assessment report Issue 1)
  2. Uniform (flat/maximum-entropy)
  3. Poisson λ=1.5 (moderately asymmetric)
  4. Poisson λ=2.5 (strongly asymmetric)
  5. Geometric (monotone, extremely asymmetric)
  6. Bimodal (multimodal, two separated Gaussian peaks)

ASSESSMENT REPORT FIX (Issue 1):
Binomial(n, p=0.5) should NOT be labelled 'multimodal'. It is correctly described
as 'BASIS-SPARSE': for n qubits, only (n+1) of the 2^n computational basis states
receive non-negligible probability mass. The remaining (2^n - n - 1) states must be
suppressed to near-zero — requiring near-complete destructive interference. This is
a structurally harder task than redistributing mass across already-occupied states.
For n=3: 4 of 8 states suppressed (50%); for n=4: 11 of 16 states suppressed (69%).
The difficulty of binomial loading increases with qubit count precisely because the
suppressed fraction grows with n.
"""

import numpy as np
from typing import Tuple, Dict, Optional
from scipy.stats import binom, poisson
import logging

logger = logging.getLogger(__name__)


class DistributionGenerator:
    """Base class for probability distribution generation."""

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.n_outcomes = 2 ** n_qubits
        logger.info(
            f"Initialized {self.__class__.__name__}: "
            f"n_qubits={n_qubits}, n_outcomes={self.n_outcomes}"
        )

    def generate(self) -> np.ndarray:
        raise NotImplementedError

    def _normalize(self, probabilities: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
        """
        Normalize probability distribution with Laplace smoothing.

        Smoothing with epsilon=1e-8 prevents log(0) in KL/JS divergence.
        This is consistent across all distributions per paper specification.
        """
        smoothed = probabilities + epsilon
        normalized = smoothed / smoothed.sum()
        return normalized


class BinomialDistribution(DistributionGenerator):
    """
    Binomial distribution: BASIS-SPARSE (not multimodal).

    PAPER SPEC:
    Binomial(n_qubits, p=0.5): Uses scipy.stats.binom.pmf evaluated at
    k=0,...,n_qubits, then zero-padded to 2^n_qubits outcomes.
    Symmetric bell shape with peak at n_qubits/2.

    ASSESSMENT REPORT FIX (Issue 1 — Critical):
    Do NOT describe binomial as 'multimodal'. The correct characterisation is
    'basis-sparse': only (n_qubits + 1) of the 2^n_qubits computational basis
    states receive non-negligible probability mass. The VQC is initialized from
    |0^n> and naturally spreads amplitude across all 2^n states via entanglement.
    Concentrating it back onto a sparse subset (k=0,...,n) while suppressing all
    other states (k=n+1,...,2^n-1) requires near-complete destructive interference
    on the suppressed states — qualitatively different from and substantially harder
    than redistributing amplitude among occupied states.

    The suppressed fraction increases with qubit count:
      n=3: 4 of 8 states suppressed (50% suppressed)
      n=4: 11 of 16 states suppressed (69% suppressed)
    This explains why binomial fidelity drops from n=3 to n=4.

    Scientific implication (novel insight from assessment report):
    Low fidelity on binomial is NOT a failure of optimisation. It is a structural
    property of basis-sparse distributions and grows intrinsically with qubit count.
    This is a publishable finding worth highlighting as a headline result.
    """

    def __init__(self, n_qubits: int, p: float = 0.5):
        super().__init__(n_qubits)
        self.p = p
        # Document basis-sparsity properties
        n_occupied = n_qubits + 1   # states with non-zero binomial mass
        n_suppressed = self.n_outcomes - n_occupied
        suppressed_fraction = n_suppressed / self.n_outcomes
        logger.info(
            f"BinomialDistribution: n={n_qubits}, p={p} "
            f"[BASIS-SPARSE: {n_occupied}/{self.n_outcomes} states occupied, "
            f"{n_suppressed} suppressed ({100*suppressed_fraction:.0f}%)]"
        )
        self.n_occupied = n_occupied
        self.n_suppressed = n_suppressed
        self.suppressed_fraction = suppressed_fraction

    def generate(self) -> np.ndarray:
        """
        Generate binomial probability distribution.

        PAPER IMPLEMENTATION:
        1. Compute P(X=k) for k=0 to n_qubits using scipy.stats.binom.pmf
        2. Zero-pad to 2^n_qubits outcomes — NOTE: indices k=n_qubits+1,...,2^n-1
           receive ZERO mass (these are the suppressed states)
        3. Add tail probability to last outcome
        4. Normalize with ε=1e-8 smoothing
        """
        k_values = np.arange(self.n_qubits + 1)
        probs = binom.pmf(k_values, self.n_qubits, self.p)

        outcome_probs = np.zeros(self.n_outcomes)
        outcome_probs[:len(probs)] = probs

        tail_prob = 1.0 - outcome_probs.sum()
        outcome_probs[-1] += tail_prob

        return self._normalize(outcome_probs, epsilon=1e-8)

    def get_sparsity_info(self) -> Dict[str, object]:
        """Return sparsity information for analysis and paper tables."""
        return {
            "description": "basis-sparse",
            "n_qubits": self.n_qubits,
            "n_occupied_states": self.n_occupied,
            "n_total_states": self.n_outcomes,
            "n_suppressed_states": self.n_suppressed,
            "suppressed_fraction": self.suppressed_fraction,
        }


class UniformDistribution(DistributionGenerator):
    """
    Uniform distribution: flat, maximum-entropy.

    PAPER SPEC:
    Uniform: np.ones(2^n_qubits) / (2^n_qubits). Trivially normalized.
    Circuit must learn to produce equal probability for all outcomes,
    suppressing all amplitude variation.
    """

    def generate(self) -> np.ndarray:
        probs = np.ones(self.n_outcomes) / self.n_outcomes
        return self._normalize(probs, epsilon=1e-8)


class PoissonDistribution(DistributionGenerator):
    """
    Poisson distribution: moderately asymmetric.

    PAPER SPEC:
    λ ∈ {1.5, 2.5}. Discretization: Truncate to 2^n_qubits outcomes.
    """

    def __init__(self, n_qubits: int, lam: float = 1.5):
        super().__init__(n_qubits)
        self.lam = lam
        logger.info(f"Poisson distribution: λ={self.lam}")

    def generate(self) -> np.ndarray:
        k_values = np.arange(self.n_outcomes)
        probs = poisson.pmf(k_values, self.lam)

        tail_prob = 1.0 - probs.sum()
        probs[-1] += tail_prob

        return self._normalize(probs, epsilon=1e-8)


class GeometricDistribution(DistributionGenerator):
    """
    Geometric distribution: extremely asymmetric, monotone decreasing.

    PAPER SPEC:
    P(k) = (1-p)^(k-1) * p for k=1,2,...,2^n_qubits.
    Heavy mass at k=1, exponential tail decay.
    """

    def __init__(self, n_qubits: int, p: float = 0.5):
        super().__init__(n_qubits)
        self.p = p
        logger.info(f"Geometric distribution: p={self.p}")

    def generate(self) -> np.ndarray:
        k_values = np.arange(1, self.n_outcomes + 1)
        probs = (1 - self.p) ** (k_values - 1) * self.p

        tail_prob = 1.0 - probs.sum()
        probs[-1] += tail_prob

        return self._normalize(probs, epsilon=1e-8)


class BimodalDistribution(DistributionGenerator):
    """
    Bimodal distribution: two separated peaks, multimodal structure.

    PAPER SPEC:
    BimodalDistribution must accept mu1, mu2, sigma as constructor arguments
    with defaults mu1=N//4, mu2=3N//4, sigma=0.8.
    P(k) ∝ exp(-(k-μ₁)²/(2σ²)) + exp(-(k-μ₂)²/(2σ²))

    QUANTUM CHALLENGE (Extreme):
    Two separated probability peaks require circuit to maintain constructive
    interference at two distant regions while destructively interfering between them.
    This is the hardest case for local (nearest-neighbor) entanglement patterns.
    """

    def __init__(self, n_qubits: int, mu1: Optional[int] = None,
                 mu2: Optional[int] = None, sigma: float = 0.8):
        super().__init__(n_qubits)
        self.mu1: int = mu1 if mu1 is not None else (self.n_outcomes // 4)
        self.mu2: int = mu2 if mu2 is not None else (3 * self.n_outcomes // 4)
        self.sigma: float = sigma

        logger.info(
            f"Bimodal distribution: μ₁={self.mu1}, μ₂={self.mu2}, σ={self.sigma}"
        )

    def generate(self) -> np.ndarray:
        k_values = np.arange(self.n_outcomes, dtype=float)

        gaussian1 = np.exp(-((k_values - self.mu1) ** 2) / (2 * self.sigma ** 2))
        gaussian2 = np.exp(-((k_values - self.mu2) ** 2) / (2 * self.sigma ** 2))

        probs = gaussian1 + gaussian2

        return self._normalize(probs, epsilon=1e-8)


# ============================================================================
# CONVENIENCE FACTORY
# ============================================================================

DISTRIBUTION_REGISTRY = {
    "binomial": lambda n: BinomialDistribution(n_qubits=n, p=0.5),
    "uniform": lambda n: UniformDistribution(n_qubits=n),
    "poisson_1.5": lambda n: PoissonDistribution(n_qubits=n, lam=1.5),
    "poisson_2.5": lambda n: PoissonDistribution(n_qubits=n, lam=2.5),
    "geometric": lambda n: GeometricDistribution(n_qubits=n, p=0.5),
    "bimodal": lambda n: BimodalDistribution(n_qubits=n),
}

DISTRIBUTION_LABELS = {
    "binomial": "Binomial (basis-sparse)",
    "uniform": "Uniform (max-entropy)",
    "poisson_1.5": r"Poisson ($\lambda$=1.5)",
    "poisson_2.5": r"Poisson ($\lambda$=2.5)",
    "geometric": "Geometric (monotone)",
    "bimodal": "Bimodal (Gaussian mix.)",
}

# Ordered from easiest (circuit-natural) to hardest
DIFFICULTY_ORDER = ["binomial", "uniform", "poisson_1.5", "poisson_2.5",
                    "geometric", "bimodal"]
