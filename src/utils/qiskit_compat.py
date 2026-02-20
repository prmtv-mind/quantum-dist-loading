"""
Qiskit 1.x execution utilities.

Per paper scaffold (Section 5.1):
"All experiments are conducted in simulation using Qiskit 1.2.4 and the
Qiskit-Aer statevector and shot-based simulators."

Uses only modern Qiskit 1.x APIs (Sampler primitive, no deprecated execute()).
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import Sampler
from qiskit_aer import AerSimulator
import logging

logger = logging.getLogger(__name__)


def get_probability_distribution(
    circuit: QuantumCircuit,
    shots: int = 1000
) -> np.ndarray:
    """
    Execute a quantum circuit and extract probability distribution.

    Uses Qiskit 1.x Sampler primitive (paper Section 5.1).

    Args:
        circuit: Qiskit QuantumCircuit (without measurements)
        shots: Number of measurement shots (paper: 1000)

    Returns:
        Normalized probability distribution of length 2^n_qubits
    """
    n_qubits = circuit.num_qubits
    n_outcomes = 2 ** n_qubits

    # Add measurements to circuit
    meas_circuit = circuit.copy()
    meas_circuit.measure_all()

    # Execute using Qiskit 1.x Sampler primitive
    sampler = Sampler()
    job = sampler.run([meas_circuit], shots=shots)
    result = job.result()

    # Extract quasi-distribution (probability from shots)
    quasi_dist = result.quasi_dists[0]

    # Convert to numpy array
    probs = np.zeros(n_outcomes)
    for outcome_int, prob in quasi_dist.items():
        if outcome_int < n_outcomes:
            probs[outcome_int] = prob

    # Normalize (quasi_dist should already be normalized, but ensure)
    return probs / probs.sum()


def get_exact_distribution(circuit: QuantumCircuit) -> np.ndarray:
    """
    Get exact probability distribution using statevector simulation.

    Uses Qiskit-Aer statevector simulator (no shot noise).

    Args:
        circuit: Qiskit QuantumCircuit (no measurements)

    Returns:
        Exact probabilities |⟨i|ψ⟩|² for each basis state
    """
    n_qubits = circuit.num_qubits
    n_outcomes = 2 ** n_qubits

    # Use AerSimulator with statevector method
    sim = AerSimulator(method='statevector')

    # Run circuit
    job = sim.run(circuit)
    result = job.result()

    # Get statevector
    statevector = result.get_statevector(circuit)

    # Convert to probabilities
    probs = np.abs(statevector) ** 2

    return probs


def get_statevector(circuit: QuantumCircuit) -> np.ndarray:
    """
    Get statevector from a circuit.

    Args:
        circuit: Qiskit QuantumCircuit (no measurements)

    Returns:
        Complex statevector
    """
    sim = AerSimulator(method='statevector')
    job = sim.run(circuit)
    result = job.result()
    return np.array(result.get_statevector(circuit))
