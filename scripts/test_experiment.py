"""
Test script: Complete single experiment end-to-end.
Updated for Qiskit 1.2.4 using modern Sampler API.
"""

import sys
from pathlib import Path
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.primitives import Sampler
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.circuits.linear import RyCZLinearAnsatz
from src.distributions.generators import BinomialDistribution
from src.metrics.fidelity import DistributionMetrics
from src.optimization.optimizer import QuantumOptimizer
from src.utils.logging import setup_logger

logger = setup_logger("test_experiment")


def get_quantum_distribution(circuit: QuantumCircuit,
                            shots: int = 1000) -> np.ndarray:
    """
    Simulate circuit and extract probability distribution using Sampler API.

    Args:
        circuit: Qiskit QuantumCircuit (no measurements)
        shots: Number of measurement shots

    Returns:
        Normalized probability distribution

    QISKIT 1.2.4 CHANGES:
    - Old: execute(circuit, backend).result().get_counts()
    - New: Sampler().run(circuit).result().quasi_dists

    This is more modular and type-safe.
    """
    # Add measurements to circuit
    meas_circuit = circuit.copy()
    meas_circuit.measure_all()

    # Create simulator with shot count
    simulator = AerSimulator()

    # Use Sampler primitive (modern Qiskit approach)
    sampler = Sampler()

    # Run with explicit shots
    job = sampler.run(meas_circuit, shots=shots)
    result = job.result()

    # Extract quasi-distribution (quasi-probabilities from measurements)
    quasi_dist = result.quasi_dists[0]

    # Convert to numpy array
    n_outcomes = 2 ** circuit.num_qubits
    probs = np.zeros(n_outcomes)

    for outcome_int, probability in quasi_dist.items():
        if outcome_int < n_outcomes:
            probs[outcome_int] = probability

    # Normalize (should already be normalized, but ensure)
    probs = probs / probs.sum() if probs.sum() > 0 else probs

    return probs


def run_experiment():
    """Execute complete single experiment."""
    logger.info("="*60)
    logger.info("STARTING TEST EXPERIMENT")
    logger.info("="*60)

    # CONFIGURATION
    n_qubits = 3
    circuit_depth = 2
    target_dist_name = "Binomial"

    logger.info(f"\nConfiguration:")
    logger.info(f"  n_qubits: {n_qubits}")
    logger.info(f"  circuit_depth: {circuit_depth}")
    logger.info(f"  target_distribution: {target_dist_name}")

    # STEP 1: Create quantum circuit
    logger.info(f"\n[1/4] Creating quantum circuit...")
    circuit_ansatz = RyCZLinearAnsatz(n_qubits=n_qubits, depth=circuit_depth)
    logger.info(f"  Circuit: {circuit_ansatz}")
    logger.info(f"  Parameters: {circuit_ansatz.get_parameter_count()}")

    # STEP 2: Generate target distribution
    logger.info(f"\n[2/4] Generating target distribution...")
    dist_gen = BinomialDistribution(n_qubits=n_qubits, p=0.5)
    target_dist = dist_gen.generate()
    logger.info(f"  Distribution: {target_dist}")
    logger.info(f"  Sum (should be 1.0): {target_dist.sum():.6f}")

    # STEP 3: Setup optimization
    logger.info(f"\n[3/4] Setting up optimization...")

    # Create cost function: L2 distance
    eval_count = [0]  # Track evaluations

    def cost_function(params: np.ndarray) -> float:
        """Cost function for optimization."""
        eval_count[0] += 1

        # Build circuit with parameters
        qc = circuit_ansatz.build_circuit(params)

        # Get quantum distribution
        quantum_dist = get_quantum_distribution(qc, shots=1000)

        # Compute L2 distance
        cost = DistributionMetrics.l2_distance(target_dist, quantum_dist)

        if eval_count[0] % 10 == 0:
            logger.info(f"  Evaluation {eval_count[0]}: cost = {cost:.6f}")

        return cost

    # Initialize parameters randomly
    np.random.seed(42)  # For reproducibility
    initial_params = np.random.uniform(0, 2*np.pi,
                                      circuit_ansatz.get_parameter_count())

    # Create optimizer
    optimizer = QuantumOptimizer(
        cost_function=cost_function,
        initial_params=initial_params,
        method="COBYLA",
        options={"maxiter": 100, "tol": 1e-4}
    )

    logger.info(f"  Optimizer: COBYLA")
    logger.info(f"  Max iterations: 100")
    logger.info(f"  Random seed: 42")

    # STEP 4: Run optimization
    logger.info(f"\n[4/4] Running optimization...")
    logger.info(f"  Starting with random parameters...")

    final_params, final_cost, n_iters = optimizer.optimize()

    # STEP 5: Evaluate result
    logger.info(f"\nOPTIMIZATION RESULTS:")
    logger.info(f"  Iterations: {n_iters}")
    logger.info(f"  Final cost: {final_cost:.6f}")

    stats = optimizer.get_convergence_stats()
    logger.info(f"\nCONVERGENCE STATISTICS:")
    for key, val in stats.items():
        if isinstance(val, float):
            logger.info(f"  {key}: {val:.6f}")
        else:
            logger.info(f"  {key}: {val}")

    # Get final quantum distribution
    final_circuit = circuit_ansatz.build_circuit(final_params)
    final_quantum_dist = get_quantum_distribution(final_circuit, shots=1000)

    # Compute all metrics
    all_metrics = DistributionMetrics.compute_all_metrics(target_dist, final_quantum_dist)
    logger.info(f"\nFINAL METRICS:")
    for metric_name, metric_value in all_metrics.items():
        logger.info(f"  {metric_name}: {metric_value:.6f}")

    # Visualize
    logger.info(f"\nGenerating visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Distribution comparison
    x = np.arange(len(target_dist))
    axes[0, 0].bar(x - 0.2, target_dist, width=0.4, label="Target", alpha=0.7)
    axes[0, 0].bar(x + 0.2, final_quantum_dist, width=0.4, label="Optimized", alpha=0.7)
    axes[0, 0].set_xlabel("Basis State")
    axes[0, 0].set_ylabel("Probability")
    axes[0, 0].set_title("Distribution Comparison (Final)")
    axes[0, 0].legend()

    # Convergence curve
    axes[0, 1].plot(optimizer.cost_history, linewidth=2)
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].set_ylabel("Cost (L2 Distance)")
    axes[0, 1].set_title("Optimization Convergence")
    axes[0, 1].grid(True, alpha=0.3)

    # Circuit diagram
    circuit_str = str(final_circuit)
    axes[1, 0].text(0.05, 0.95, circuit_str, transform=axes[1, 0].transAxes,
                   fontsize=8, verticalalignment='top', family='monospace')
    axes[1, 0].axis('off')
    axes[1, 0].set_title("Final Optimized Circuit")

    # Metrics table
    metrics_text = "Final Metrics:\n" + "\n".join(
        f"{k}: {v:.6f}" for k, v in all_metrics.items()
    )
    axes[1, 1].text(0.1, 0.9, metrics_text, transform=axes[1, 1].transAxes,
                   fontsize=10, verticalalignment='top', family='monospace')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plot_path = project_root / "results" / "test_experiment_result.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    logger.info(f"  Saved plot to: {plot_path}")

    logger.info("\n" + "="*60)
    logger.info("TEST EXPERIMENT COMPLETE")
    logger.info("="*60)

    return {
        "n_qubits": n_qubits,
        "circuit_depth": circuit_depth,
        "final_cost": final_cost,
        "iterations": n_iters,
        "metrics": all_metrics,
    }


if __name__ == "__main__":
    run_experiment()
