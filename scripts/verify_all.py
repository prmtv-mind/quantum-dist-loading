# Replace the top imports section with this:
import sys
from pathlib import Path
import numpy as np

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
)
from src.metrics.fidelity import DistributionMetrics
from src.optimization.optimizer import QuantumOptimizer
from src.optimization.spsa import SPSAOptimizer
from src.utils.qiskit_compat import get_probability_distribution, get_exact_distribution

def verify_distributions():
    """Verify all distributions work correctly."""
    print("\n" + "="*70)
    print("VERIFYING DISTRIBUTIONS")
    print("="*70)

    n_qubits = 3

    dists = [
        ("Binomial", BinomialDistribution(n_qubits)),
        ("Uniform", UniformDistribution(n_qubits)),
        ("Poisson(1.5)", PoissonDistribution(n_qubits, lam=1.5)),
        ("Poisson(2.5)", PoissonDistribution(n_qubits, lam=2.5)),
        ("Geometric", GeometricDistribution(n_qubits)),
        ("Bimodal", BimodalDistribution(n_qubits)),
    ]

    for name, gen in dists:
        dist = gen.generate()

        # Checks per paper
        assert len(dist) == 2**n_qubits, f"{name}: wrong length"
        assert np.isclose(dist.sum(), 1.0), f"{name}: doesn't sum to 1"
        assert (dist > 1e-8).all(), f"{name}: values < 1e-8 (smoothing failed)"
        assert np.isfinite(dist).all(), f"{name}: contains NaN/inf"

        print(f"✓ {name:20s} - shape={dist.shape}, sum={dist.sum():.10f}, min={dist.min():.2e}")

    print(f"\n✓ All {len(dists)} distributions verified")


def verify_circuits():
    """Verify all circuit architectures work correctly."""
    print("\n" + "="*70)
    print("VERIFYING CIRCUITS")
    print("="*70)

    circuits = [
        ("Linear", RyCZLinearAnsatz),
        ("Circular", RyCZCircularAnsatz),
        ("Alternating", RyCZAlternatingAnsatz),
    ]

    for name, circuit_class in circuits:
        for n_qubits in [3, 4]:
            for depth in [1, 2]:
                circuit = circuit_class(n_qubits=n_qubits, depth=depth)

                # Verify parameter count
                expected_params = n_qubits * (depth + 1)
                assert circuit.get_parameter_count() == expected_params, \
                    f"{name} n={n_qubits} d={depth}: param count mismatch"

                # Build circuit
                params = np.random.uniform(0, 2*np.pi, circuit.get_parameter_count())
                qc = circuit.build_circuit(params)

                assert qc.num_qubits == n_qubits
                assert len(qc) > 0

                print(f"✓ {name:15s} n={n_qubits} d={depth} - {circuit.get_parameter_count()} params, {len(qc)} instructions")

    print(f"\n✓ All circuit architectures verified")


def verify_metrics():
    """Verify metric calculations."""
    print("\n" + "="*70)
    print("VERIFYING METRICS")
    print("="*70)

    # Test cases
    p_identical = np.array([0.25, 0.25, 0.25, 0.25])
    p_different = np.array([1.0, 0, 0, 0])
    p_similar = np.array([0.2, 0.3, 0.25, 0.25])

    # Identical should be perfect
    metrics = DistributionMetrics.compute_all_metrics(p_identical, p_identical)
    assert metrics["l2_distance"] < 1e-6, "Identical L2 should be ~0"
    assert metrics["fidelity"] > 0.999, "Identical fidelity should be ~1"
    print(f"✓ Identical distributions - L2={metrics['l2_distance']:.2e}, Fidelity={metrics['fidelity']:.6f}")

    # Different should be far
    metrics = DistributionMetrics.compute_all_metrics(p_identical, p_different)
    assert metrics["l2_distance"] > 0.5, "Different L2 should be large"
    assert metrics["fidelity"] < 0.5, "Different fidelity should be small"
    print(f"✓ Different distributions - L2={metrics['l2_distance']:.6f}, Fidelity={metrics['fidelity']:.6f}")

    # All metrics bounded
    metrics = DistributionMetrics.compute_all_metrics(p_identical, p_similar)
    assert 0 <= metrics["jensen_shannon"] <= 1, "JS should be in [0,1]"
    assert 0 <= metrics["ks_statistic"] <= 1, "KS should be in [0,1]"
    assert 0 <= metrics["fidelity"] <= 1, "Fidelity should be in [0,1]"
    print(f"✓ Similar distributions - L2={metrics['l2_distance']:.6f}, KL={metrics['kl_divergence']:.6f}, JS={metrics['jensen_shannon']:.6f}")

    print(f"\n✓ All metrics verified")


def verify_optimizers():
    """Verify optimizer interfaces."""
    print("\n" + "="*70)
    print("VERIFYING OPTIMIZERS")
    print("="*70)

    # Simple test function
    def test_cost(x):
        return np.sum((x - np.array([1.5, -2.0])) ** 2)

    initial = np.array([0.0, 0.0])

    # Test COBYLA
    opt_cobyla = QuantumOptimizer(
        test_cost, initial, method="COBYLA",
        options={"maxiter": 50, "tol": 1e-4}
    )
    params_cobyla, cost_cobyla, iters_cobyla = opt_cobyla.optimize()
    assert len(params_cobyla) == 2
    assert cost_cobyla < 0.1, "COBYLA should converge close to minimum"
    print(f"✓ COBYLA - final_cost={cost_cobyla:.6f}, iterations={iters_cobyla}, params={params_cobyla}")

    # Test SPSA
    opt_spsa = SPSAOptimizer(
        test_cost, initial,
        a=0.1, c=0.1, maxiter=100, random_seed=42
    )
    params_spsa, cost_spsa, iters_spsa = opt_spsa.optimize()
    assert len(params_spsa) == 2
    assert cost_spsa < 0.5, "SPSA should make progress toward minimum"
    print(f"✓ SPSA    - final_cost={cost_spsa:.6f}, iterations={iters_spsa}, params={params_spsa}")

    # Verify same interface
    stats_cobyla = opt_cobyla.get_convergence_stats()
    stats_spsa = opt_spsa.get_convergence_stats()

    for key in ["initial_cost", "final_cost", "iterations"]:
        assert key in stats_cobyla, f"COBYLA missing {key}"
        assert key in stats_spsa, f"SPSA missing {key}"

    print(f"\n✓ Both optimizers have identical interface")


def verify_integration():
    """Quick integration test: full optimization run."""
    print("\n" + "="*70)
    print("VERIFYING INTEGRATION (End-to-End)")
    print("="*70)

    np.random.seed(42)

    # Setup
    n_qubits = 3
    depth = 1
    circuit = RyCZLinearAnsatz(n_qubits=n_qubits, depth=depth)
    dist_gen = BinomialDistribution(n_qubits=n_qubits)
    target_dist = dist_gen.generate()

    # Cost function (using Qiskit 1.x Sampler)
    def cost_func(params):
        qc = circuit.build_circuit(params)
        quantum_dist = get_probability_distribution(qc, shots=100)  # Reduced for speed
        return float(DistributionMetrics.l2_distance(target_dist, quantum_dist))

    # Optimize
    initial_params = np.random.uniform(0, 2*np.pi, circuit.get_parameter_count())
    optimizer = QuantumOptimizer(
        cost_func, initial_params,
        method="COBYLA",
        options={"maxiter": 20, "tol": 1e-4}
    )

    final_params, final_cost, n_iters = optimizer.optimize()

    print(f"✓ Integration test complete:")
    print(f"  Circuit: {circuit}")
    print(f"  Target: Binomial")
    print(f"  Initial cost: {optimizer.cost_history[0]:.6f}")
    print(f"  Final cost: {final_cost:.6f}")
    print(f"  Iterations: {n_iters}")
    print(f"  Convergence: {100*(optimizer.cost_history[0]-final_cost)/optimizer.cost_history[0]:.1f}%")


def main():
    """Run all verifications."""
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  COMPREHENSIVE VERIFICATION SUITE".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)

    try:
        verify_distributions()
        verify_circuits()
        verify_metrics()
        verify_optimizers()
        verify_integration()

        print("\n" + "█"*70)
        print("█" + " "*68 + "█")
        print("█" + "  ✓ ALL VERIFICATIONS PASSED - READY FOR EXPERIMENTS".center(68) + "█")
        print("█" + " "*68 + "█")
        print("█"*70 + "\n")

        return True

    except Exception as e:
        print(f"\n✗ VERIFICATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
