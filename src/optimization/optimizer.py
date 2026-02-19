"""Quantum optimizer for variational circuits."""

from typing import Callable, Dict, Any, Optional, Tuple
import numpy as np
from scipy.optimize import minimize


class QuantumOptimizer:
    """Optimize variational quantum circuit parameters using classical optimizer."""

    def __init__(self,
                 cost_function: Callable[[np.ndarray], float],
                 initial_params: np.ndarray,
                 method: str = "COBYLA",
                 options: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize optimizer.

        Args:
            cost_function: Function mapping parameters to cost value
            initial_params: Initial parameter values
            method: Optimization method (COBYLA, SLSQP, L-BFGS-B, etc.)
            options: Method-specific options dictionary
        """
        self.cost_function = cost_function
        self.initial_params = initial_params.copy()
        self.method = method
        self.options = options if options is not None else {}

        # Track convergence
        self.cost_history = []
        self.param_history = []
        self.n_evaluations = 0

    def _wrapped_cost_function(self, params: np.ndarray) -> float:
        """Wrap cost function to track convergence."""
        cost = self.cost_function(params)
        self.cost_history.append(cost)
        self.param_history.append(params.copy())
        self.n_evaluations += 1
        return cost

    def optimize(self) -> Tuple[np.ndarray, float, int]:
        """
        Run optimization.

        Returns:
            Tuple of (optimized_params, final_cost, n_iterations)
        """
        result = minimize(
            self._wrapped_cost_function,
            self.initial_params,
            method=self.method,
            options=self.options
        )

        return result.x, result.fun, result.nit

    def get_convergence_stats(self) -> Dict[str, Any]:
        """Get optimization convergence statistics."""
        if not self.cost_history:
            return {}

        costs = np.array(self.cost_history)

        return {
            "initial_cost": float(costs[0]),
            "final_cost": float(costs[-1]),
            "cost_improvement": float(costs[0] - costs[-1]),
            "min_cost": float(np.min(costs)),
            "mean_cost": float(np.mean(costs)),
            "std_cost": float(np.std(costs)),
            "n_evaluations": self.n_evaluations,
        }


# Example: simple test
if __name__ == "__main__":
    # Simple quadratic cost function
    def quadratic_cost(x):
        return np.sum((x - np.array([1.5, -2.0])) ** 2)

    initial = np.array([0.0, 0.0])
    optimizer = QuantumOptimizer(quadratic_cost, initial)

    params, cost, iters = optimizer.optimize()
    print(f"Optimized parameters: {params}")
    print(f"Final cost: {cost:.6f}")
    print(f"Iterations: {iters}")
    print(f"\nConvergence stats:")
    for key, val in optimizer.get_convergence_stats().items():
        print(f"  {key}: {val}")
