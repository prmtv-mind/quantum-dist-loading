"""
Optimizer wrapper for parameterized quantum circuits.

CLASSICAL OPTIMIZATION IN QUANTUM COMPUTING:
1. Start with random parameters θ₀
2. Evaluate quantum circuit: get probability distribution p(θ)
3. Compute cost: distance between p(θ) and target
4. Update parameters θ → θ - α∇cost
5. Repeat until convergence

We implement COBYLA (gradient-free) which works well for quantum circuits.
Reference: Dasgupta & Paine (2022) - arXiv:2208.13372
"""

import numpy as np
from typing import Callable, Tuple, Dict, Any, Optional
from scipy.optimize import minimize
import logging
import time

logger = logging.getLogger(__name__)


class QuantumOptimizer:
    """Wrapper around scipy optimizers for quantum circuit parameter optimization."""

    def __init__(self, cost_function: Callable,
                 initial_params: np.ndarray,
                 method: str = "COBYLA",
                 options: Optional[Dict[str, Any]] = None):
        """
        Initialize optimizer.

        Args:
            cost_function: Function to minimize. Should take parameter array
                          and return scalar cost value
            initial_params: Starting point for optimization
            method: Optimizer method ("COBYLA", "Nelder-Mead", etc.)
            options: Method-specific options (optional)

        COBYLA RATIONALE:
        - Gradient-free (no need to compute quantum gradients)
        - Noisy function tolerant (good for quantum)
        - Constraint handling (if needed later)
        - Well-tested on VQE problems
        """
        self.cost_function = cost_function
        self.initial_params = initial_params.copy()
        self.method = method
        # FIX: Properly handle None type
        self.options: Dict[str, Any] = options if options is not None else {}

        # Set default options for COBYLA
        if method == "COBYLA" and "maxiter" not in self.options:
            self.options["maxiter"] = 500
            self.options["tol"] = 1e-4

        # Tracking
        self.cost_history = []
        self.param_history = []
        self.iteration = 0

        logger.info(f"Initialized QuantumOptimizer: method={method}, options={self.options}")

    def callback(self, xk: np.ndarray) -> None:
        """
        Called after each optimization iteration to track progress.

        Args:
            xk: Current parameter vector
        """
        cost = self.cost_function(xk)
        self.cost_history.append(cost)
        self.param_history.append(xk.copy())
        self.iteration += 1

        if self.iteration % 50 == 0:
            logger.info(f"Iteration {self.iteration}: cost = {cost:.6f}")

    def optimize(self) -> Tuple[np.ndarray, float, int]:
        """
        Run optimization to convergence.

        Returns:
            Tuple of:
            - Final parameters (optimal θ)
            - Final cost value
            - Number of iterations

        IMPLEMENTATION NOTES:
        - Uses scipy.optimize.minimize with selected method
        - Callback tracks convergence history
        - Returns optimization result with stats
        """
        logger.info(f"Starting optimization with {self.method}")
        start_time = time.time()

        # Run optimization
        result = minimize(
            self.cost_function,
            self.initial_params,
            method=self.method,
            callback=self.callback,
            options=self.options
        )

        elapsed_time = time.time() - start_time

        # Get iteration count - use our callback counter (most reliable)
        n_iterations = self.iteration

        logger.info(
            f"Optimization complete: {n_iterations} iterations, "
            f"final cost = {result.fun:.6f}, "
            f"time = {elapsed_time:.2f}s"
        )

        return result.x, result.fun, n_iterations

    def get_convergence_stats(self) -> Dict[str, float]:
        """
        Compute convergence statistics.

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
