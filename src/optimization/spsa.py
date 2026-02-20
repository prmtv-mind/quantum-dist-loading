"""
SPSA (Simultaneous Perturbation Stochastic Approximation) optimizer.

THEORY:
SPSA is a noise-robust gradient estimation method. Instead of computing the
full gradient ∇C(θ) exactly (requiring 2n function evaluations for n parameters),
SPSA estimates it using only 2 function evaluations:

  1. Perturb all parameters simultaneously: δ ~ Bernoulli({±1})^n
  2. Evaluate at θ + c·δ and θ - c·δ  (only 2 evals, not 2n)
  3. Estimate gradient: g ≈ [C(θ+c·δ) - C(θ-c·δ)] / (2c) × δ^{-1}
  4. Update: θ ← θ - a·g

WHY FOR QUANTUM?
- Shot noise from finite measurements means true gradient is noisy anyway
- SPSA's stochastic perturbation aligns naturally with this noise
- Parameter count doesn't increase sample complexity (unlike finite differences)
- Proven effective for VQE and QAOA

HYPERPARAMETERS (from Spall 1998 recommendations):
  a_k = a / (A + k + 1)^α   where a=0.1, A=10, α=0.602
  c_k = c / (k + 1)^γ       where c=0.1, γ=0.101

The specific exponents (0.602, 0.101) come from optimal asymptotic theory for SPSA.
"""

import numpy as np
from typing import Callable, Tuple, Dict, Any
import logging
import time

logger = logging.getLogger(__name__)


class SPSAOptimizer:
    """
    Simultaneous Perturbation Stochastic Approximation optimizer.

    Estimates gradients stochastically using simultaneous perturbations,
    making it sample-efficient and noise-robust for quantum optimization.
    """

    def __init__(self,
                 cost_function: Callable,
                 initial_params: np.ndarray,
                 a: float = 0.1,
                 c: float = 0.1,
                 A: float = 10,
                 alpha: float = 0.602,
                 gamma: float = 0.101,
                 maxiter: int = 500,
                 random_seed: int = None):
        """
        Initialize SPSA optimizer.

        Args:
            cost_function: Callable that takes parameter array, returns scalar cost
            initial_params: Starting parameter vector
            a, c, A, alpha, gamma: Hyperparameters (see Spall 1998)
            maxiter: Maximum iterations
            random_seed: Seed for random number generator (perturbation randomness)

        HYPERPARAMETER MEANING:
        - a: Step size scale (larger → faster learning but more noise sensitivity)
        - c: Perturbation scale (larger → noisier gradient estimate, less overfitting)
        - A: Burnin parameter (a_k constant for first A iterations, then decay)
        - alpha, gamma: Decay rates (Spall theory: 0.602 and 0.101 are optimal)
        """
        self.cost_function = cost_function
        self.initial_params = initial_params.copy()
        self.n_params = len(initial_params)

        # SPSA hyperparameters
        self.a = a
        self.c = c
        self.A = A
        self.alpha = alpha
        self.gamma = gamma
        self.maxiter = maxiter

        # RNG for perturbations
        self.rng = np.random.RandomState(random_seed)

        # Tracking
        self.cost_history = []
        self.param_history = []
        self.iteration = 0
        self.gradient_history = []

        logger.info(
            f"Initialized SPSAOptimizer: "
            f"n_params={self.n_params}, maxiter={maxiter}, "
            f"a={a}, c={c}, alpha={alpha}, gamma={gamma}"
        )

    def _learning_rate(self, k: int) -> float:
        """
        Compute learning rate a_k for iteration k.

        a_k = a / (A + k + 1)^α

        Decays over iterations, allowing coarse initial exploration that
        becomes fine-tuned refinement as iterations progress.
        """
        return self.a / ((self.A + k + 1) ** self.alpha)

    def _perturbation_scale(self, k: int) -> float:
        """
        Compute perturbation scale c_k for iteration k.

        c_k = c / (k + 1)^γ

        Also decays, but typically slower than a_k. Ensures gradient
        estimate remains noisy for exploration in early iterations,
        becomes precise late in optimization.
        """
        return self.c / ((k + 1) ** self.gamma)

    def _estimate_gradient(self, theta: np.ndarray, c_k: float) -> np.ndarray:
        """
        Estimate gradient using simultaneous perturbations (2 function evals).

        Args:
            theta: Current parameters
            c_k: Perturbation scale for this iteration

        Returns:
            Gradient estimate (n_params,)

        IMPLEMENTATION:
        1. Generate random ±1 perturbation vector δ
        2. Evaluate cost at θ + c·δ and θ - c·δ
        3. Estimate gradient: g = [C(θ+c·δ) - C(θ-c·δ)] / (2c) × δ^{-1}

        WHY THIS WORKS:
        For smooth function, C(θ±c·δ) ≈ C(θ) ± c·∇C(θ)·δ
        So [C(θ+c·δ) - C(θ-c·δ)] / (2c) ≈ ∇C(θ)·δ
        Dividing by δ (element-wise) recovers ∇C(θ).
        """
        # Generate Bernoulli ±1 perturbations
        delta = 2 * self.rng.randint(0, 2, size=self.n_params) - 1  # {-1, +1}^n

        # Evaluate at perturbed points
        cost_plus = self.cost_function(theta + c_k * delta)
        cost_minus = self.cost_function(theta - c_k * delta)

        # Estimate gradient
        gradient = (cost_plus - cost_minus) / (2 * c_k) / delta

        return gradient

    def step(self) -> Tuple[float, np.ndarray]:
        """
        Execute one SPSA iteration.

        Returns:
            Tuple of (current_cost, current_params)
        """
        k = self.iteration
        theta_k = self.param_history[-1].copy() if self.param_history else self.initial_params.copy()

        # Compute learning rate and perturbation scale
        a_k = self._learning_rate(k)
        c_k = self._perturbation_scale(k)

        # Estimate gradient (2 function evals)
        grad_k = self._estimate_gradient(theta_k, c_k)
        self.gradient_history.append(grad_k.copy())

        # Update parameters
        theta_next = theta_k - a_k * grad_k

        # Evaluate cost at new point (for logging, not for update)
        cost_k = self.cost_function(theta_next)

        # Track
        self.cost_history.append(cost_k)
        self.param_history.append(theta_next.copy())
        self.iteration += 1

        if self.iteration % 50 == 0:
            logger.info(
                f"SPSA Iteration {self.iteration}: "
                f"cost={cost_k:.6f}, a_k={a_k:.6f}, c_k={c_k:.6f}"
            )

        return cost_k, theta_next

    def optimize(self) -> Tuple[np.ndarray, float, int]:
        """
        Run SPSA optimization to convergence.

        Returns:
            Tuple of (final_parameters, final_cost, num_iterations)
        """
        logger.info("Starting SPSA optimization")
        start_time = time.time()

        # Initialize history
        self.param_history = [self.initial_params.copy()]

        # Main optimization loop
        for k in range(self.maxiter):
            cost_k, theta_k = self.step()

        elapsed_time = time.time() - start_time
        final_cost = self.cost_history[-1]
        final_params = self.param_history[-1]

        logger.info(
            f"SPSA optimization complete: {self.iteration} iterations, "
            f"final cost={final_cost:.6f}, time={elapsed_time:.2f}s"
        )

        return final_params, final_cost, self.iteration

    def get_convergence_stats(self) -> Dict[str, float]:
        """
        Compute convergence statistics (same interface as COBYLA optimizer).

        Returns:
            Dictionary with convergence metrics
        """
        costs = np.array(self.cost_history)

        return {
            "initial_cost": float(costs[0]),
            "final_cost": float(costs[-1]),
            "cost_reduction": float(costs[0] - costs[-1]),
            "percent_reduction": float(100 * (costs[0] - costs[-1]) / (costs[0] + 1e-10)),
            "mean_cost": float(costs.mean()),
            "std_cost": float(costs.std()),
            "iterations": self.iteration,
        }


if __name__ == "__main__":
    # Simple quadratic test
    def test_cost(x):
        return np.sum((x - np.array([1.5, -2.0])) ** 2)

    initial = np.array([0.0, 0.0])
    optimizer = SPSAOptimizer(test_cost, initial, maxiter=100, random_seed=42)

    params, cost, iters = optimizer.optimize()
    print(f"Final parameters: {params}")
    print(f"Final cost: {cost:.6f}")
    print(f"Stats: {optimizer.get_convergence_stats()}")
