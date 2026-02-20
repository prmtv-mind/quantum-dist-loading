"""
Batch experiment runner for distribution loading benchmarking.
Compatible with Qiskit 1.x using Sampler primitive.
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Callable, Type
import logging
from tqdm import tqdm
from joblib import Parallel, delayed
import argparse
import time

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.circuits.linear import RyCZLinearAnsatz
from src.circuits.circular import RyCZCircularAnsatz
from src.circuits.alternating import RyCZAlternatingAnsatz
from src.distributions.generators import (
    BinomialDistribution,
    UniformDistribution,
    PoissonDistribution,
    GeometricDistribution,
    BimodalDistribution,
    DistributionGenerator,  # Add base class import
)
from src.metrics.fidelity import DistributionMetrics
from src.optimization.optimizer import QuantumOptimizer
from src.optimization.spsa import SPSAOptimizer
from src.utils.logging import setup_logger
from src.utils.qiskit_compat import get_probability_distribution

logger = setup_logger("run_experiments", level="INFO")


# ============================================================================
# CONFIGURATION
# ============================================================================

# FIX: Use proper type hints for distribution builders
DISTRIBUTIONS: Dict[str, Callable[[int], DistributionGenerator]] = {
    "binomial": lambda n: BinomialDistribution(n_qubits=n, p=0.5),
    "uniform": lambda n: UniformDistribution(n_qubits=n),
    "poisson_1.5": lambda n: PoissonDistribution(n_qubits=n, lam=1.5),
    "poisson_2.5": lambda n: PoissonDistribution(n_qubits=n, lam=2.5),
    "geometric": lambda n: GeometricDistribution(n_qubits=n, p=0.5),
    "bimodal": lambda n: BimodalDistribution(n_qubits=n),
}

CIRCUITS = {
    "linear": RyCZLinearAnsatz,
    "circular": RyCZCircularAnsatz,
    "alternating": RyCZAlternatingAnsatz,
}

OPTIMIZERS = ["COBYLA", "SPSA"]

FULL_FACTORIAL = {
    "distributions": list(DISTRIBUTIONS.keys()),
    "circuits": list(CIRCUITS.keys()),
    "depths": [1, 2, 3, 4],
    "n_qubits": [3, 4],
    "optimizers": OPTIMIZERS,
    "seeds": list(range(10)),
}

DEMO_MODE = {
    "distributions": ["binomial", "poisson_1.5"],
    "circuits": ["linear", "circular"],
    "depths": [1, 2],
    "n_qubits": [3],
    "optimizers": ["COBYLA"],
    "seeds": [0, 1, 2],
}


# ============================================================================
# SINGLE RUN FUNCTION
# ============================================================================

def run_single_experiment(
    distribution_name: str,
    circuit_name: str,
    depth: int,
    n_qubits: int,
    optimizer_name: str,
    seed: int,
    shots: int = 1000,
) -> Dict[str, Any]:
    """
    Execute a single optimization experiment.

    Returns:
        Dictionary with all metrics and metadata
    """
    try:
        np.random.seed(seed)

        # FIX: Properly instantiate distribution using callable
        dist_builder: Callable[[int], DistributionGenerator] = DISTRIBUTIONS[distribution_name]
        dist_gen: DistributionGenerator = dist_builder(n_qubits)
        target_dist = dist_gen.generate()

        # Create circuit
        circuit_class = CIRCUITS[circuit_name]
        circuit = circuit_class(n_qubits=n_qubits, depth=depth)

        # Cost function: L2 distance (using Qiskit 1.x Sampler)
        def cost_func(params: np.ndarray) -> float:
            qc = circuit.build_circuit(params)
            quantum_dist = get_probability_distribution(qc, shots=shots)
            return float(DistributionMetrics.l2_distance(target_dist, quantum_dist))

        # Initialize parameters
        initial_params = np.random.uniform(0, 2 * np.pi, circuit.get_parameter_count())

        # Run optimization
        if optimizer_name == "COBYLA":
            optimizer = QuantumOptimizer(
                cost_func,
                initial_params,
                method="COBYLA",
                options={"maxiter": 500, "tol": 1e-4}
            )
        else:  # SPSA
            optimizer = SPSAOptimizer(
                cost_func,
                initial_params,
                a=0.1,
                c=0.1,
                maxiter=500,
                random_seed=seed
            )

        # FIX: Ensure optimizer has results before accessing
        final_params, final_cost, n_iters = optimizer.optimize()

        # FIX: Check cost_history is not None before accessing
        if not hasattr(optimizer, "cost_history") or optimizer.cost_history is None:
            logger.error(f"Optimizer failed to track cost history")
            raise RuntimeError("Optimizer did not track cost history")

        initial_cost = float(optimizer.cost_history[0]) if len(optimizer.cost_history) > 0 else float('nan')

        # Evaluate final distribution
        final_qc = circuit.build_circuit(final_params)
        final_probs = get_probability_distribution(final_qc, shots=shots)

        # Compute all metrics
        all_metrics = DistributionMetrics.compute_all_metrics(target_dist, final_probs)

        # Circuit info
        gate_counts = circuit.get_gate_counts(final_qc)

        return {
            "distribution": distribution_name,
            "circuit": circuit_name,
            "depth": depth,
            "n_qubits": n_qubits,
            "optimizer": optimizer_name,
            "seed": seed,
            "status": "success",
            "iterations": int(n_iters),
            "initial_cost": initial_cost,
            "final_cost": float(final_cost),
            **{k: float(v) for k, v in all_metrics.items()},
            **{f"gate_{k}": int(v) for k, v in gate_counts.items()},
            "parameters": final_params.tolist(),
            "target_distribution": target_dist.tolist(),
            "achieved_distribution": final_probs.tolist(),
        }

    except Exception as e:
        logger.error(f"Run failed [{distribution_name}, {circuit_name}, depth {depth}, seed {seed}]: {e}")
        import traceback
        logger.debug(traceback.format_exc())

        return {
            "distribution": distribution_name,
            "circuit": circuit_name,
            "depth": depth,
            "n_qubits": n_qubits,
            "optimizer": optimizer_name,
            "seed": seed,
            "status": "error",
            "error_message": str(e),
        }


# ============================================================================
# BATCH RUNNER
# ============================================================================

def generate_experiment_configs(mode: str = "demo") -> List[Dict[str, Any]]:
    """Generate all experiment configurations."""
    config_template = DEMO_MODE if mode == "demo" else FULL_FACTORIAL

    configs: List[Dict[str, Any]] = []
    for dist_name in config_template["distributions"]:
        for circ_name in config_template["circuits"]:
            for depth in config_template["depths"]:
                for n_qubits in config_template["n_qubits"]:
                    for optimizer in config_template["optimizers"]:
                        for seed in config_template["seeds"]:
                            configs.append({
                                "distribution_name": dist_name,
                                "circuit_name": circ_name,
                                "depth": depth,
                                "n_qubits": n_qubits,
                                "optimizer_name": optimizer,
                                "seed": seed,
                            })

    return configs


def run_batch(mode: str = "demo", n_jobs: int = 1, resume: bool = False):
    """
    Run batch of experiments.

    Args:
        mode: "demo" or "full"
        n_jobs: Number of parallel workers (1 = sequential)
        resume: Resume from checkpoints if True
    """
    logger.info("="*70)
    logger.info(f"STARTING BATCH RUN: {mode.upper()} MODE, n_jobs={n_jobs}")
    logger.info("="*70)

    # Generate configs
    configs = generate_experiment_configs(mode=mode)
    total_runs = len(configs)

    logger.info(f"Total configurations: {total_runs}")
    logger.info(f"Expected time (demo, parallel): ~5-10 min")
    logger.info(f"Expected time (full, parallel): ~3-12 hrs")

    # Setup results directory
    results_dir = project_root / "results" / "raw" / mode
    results_dir.mkdir(parents=True, exist_ok=True)

    # Run experiments (with or without parallelization)
    # FIX: Convert generator to list to match type annotation
    if n_jobs == 1:
        results: list = []  # Explicit type annotation
        for config in tqdm(configs, desc="Running experiments"):
            result = run_single_experiment(**config)
            results.append(result)
    else:
        results_parallel = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(run_single_experiment)(**config) for config in configs
        )
        results: list = list(results_parallel)  # Ensure it's a list

    # Save results
    for i, result in enumerate(results):
        config_str = (
            f"{result['distribution']}_"
            f"{result['circuit']}_"
            f"d{result['depth']}_"
            f"q{result['n_qubits']}_"
            f"{result['optimizer']}_"
            f"s{result['seed']}"
        )
        result_file = results_dir / f"{config_str}.json"
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2)

    logger.info(f"\n✓ Saved {len(results)} results to {results_dir}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run distribution loading experiments")
    parser.add_argument("--mode", choices=["demo", "full"], default="demo")
    parser.add_argument("--n-jobs", type=int, default=1)

    args = parser.parse_args()

    start_time = time.time()
    results = run_batch(mode=args.mode, n_jobs=args.n_jobs)
    elapsed = time.time() - start_time

    logger.info(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
