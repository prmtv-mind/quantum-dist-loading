"""
Qiskit 1.x compatibility utilities.

Qiskit 1.x introduced breaking changes:
- Deprecated: qiskit.execute() and backend.run()
- New: qiskit.primitives.Sampler for measurement-based execution
- New: qiskit.primitives.Estimator for observable estimation

This module provides utilities to execute circuits and extract distributions
in the Qiskit 1.x API.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import Sampler
from qiskit.transpiler import PassManager
import logging

logger = logging.getLogger(__name__)


def get_probability_distribution(
    circuit: QuantumCircuit,
    shots: int = 1000
) -> np.ndarray:
    """
    Execute a quantum circuit and extract probability distribution using Qiskit 1.x Sampler.

    Args:
        circuit: Qiskit QuantumCircuit (without measurements - they're added internally)
        shots: Number of measurement shots

    Returns:
        Normalized probability distribution (length 2^n_qubits)

    QISKIT 1.X CHANGES:
    1. Create a measurement circuit from the state preparation circuit
    2. Use Sampler primitive (not deprecated execute())
    3. Sampler returns QuasiDistribution with measurement counts
    4. Convert to probability array

    REFERENCE:
    https://qiskit.org/documentation/guides/primitives.html
    """
    n_qubits = circuit.num_qubits
    n_outcomes = 2 ** n_qubits

    # Create circuit copy and add measurements
    meas_circuit = circuit.copy()
    meas_circuit.measure_all()

    # Use Sampler primitive (Qiskit 1.x)
    sampler = Sampler()

    try:
        # Run the circuit with sampler
        job = sampler.run(
            [meas_circuit],
            shots=shots,
        )
        result = job.result()

        # Extract quasi-distribution from result
        quasi_dist = result.quasi_dists[0]

        # Convert to numpy array
        probs = np.zeros(n_outcomes)
        for outcome_int, prob in quasi_dist.items():
            if outcome_int < n_outcomes:
                probs[outcome_int] = prob

        # Normalize (should already be normalized, but ensure)
        probs = probs / probs.sum()

        return probs

    except Exception as e:
        logger.error(f"Error in Sampler execution: {e}")
        logger.warning("Falling back to statevector simulation...")

        # Fallback: use statevector simulation
        from qiskit_aer import AerSimulator

        sim = AerSimulator(method='statevector')
        meas_circuit_transpiled = sim.transpile(meas_circuit)
        job = sim.run(meas_circuit_transpiled, shots=shots)
        result = job.result()
        counts = result.get_counts()

        probs = np.zeros(n_outcomes)
        for outcome_str, count in counts.items():
            outcome_int = int(outcome_str, 2)
            probs[outcome_int] = count / shots

        return probs


def get_statevector(circuit: QuantumCircuit) -> np.ndarray:
    """
    Get statevector from a circuit (for exact distribution).

    Args:
        circuit: Qiskit QuantumCircuit (no measurements)

    Returns:
        Complex statevector (length 2^n_qubits)
    """
    from qiskit_aer import AerSimulator

    n_qubits = circuit.num_qubits

    sim = AerSimulator(method='statevector')
    circuit_copy = circuit.copy()
    circuit_copy.save_statevector()

    job = sim.run(circuit_copy)
    result = job.result()
    statevector = result.get_statevector()

    return np.array(statevector)


def get_exact_distribution(circuit: QuantumCircuit) -> np.ndarray:
    """
    Get exact probability distribution from statevector (no shot noise).

    Args:
        circuit: Qiskit QuantumCircuit

    Returns:
        Exact probabilities |⟨i|ψ⟩|² for each basis state
    """
    statevector = get_statevector(circuit)
    probs = np.abs(statevector) ** 2
    return probs
