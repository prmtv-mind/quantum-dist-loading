"""
Qiskit execution utilities with cross-version compatibility.

Supports both:
- Qiskit 2.x primitives (`StatevectorSampler`)
- Older Qiskit 1.x primitives (`Sampler`) as fallback
"""

import numpy as np
from qiskit import QuantumCircuit
import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    # Qiskit 2.x path
    from qiskit.primitives import StatevectorSampler  # type: ignore
except ImportError:
    StatevectorSampler = None

try:
    # Qiskit 1.x fallback path
    from qiskit.primitives import Sampler as LegacySampler  # type: ignore
except ImportError:
    LegacySampler = None

try:
    from qiskit_aer import AerSimulator  # type: ignore
except ImportError:
    AerSimulator = None

from qiskit.quantum_info import Statevector


def _has_measurements(circuit: QuantumCircuit) -> bool:
    return any(instr.operation.name == "measure" for instr in circuit.data)


def _prepare_measured_circuit(circuit: QuantumCircuit) -> QuantumCircuit:
    meas_circuit = circuit.copy()
    if not _has_measurements(meas_circuit):
        meas_circuit.measure_all()
    return meas_circuit


def _prepare_statevector_circuit(circuit: QuantumCircuit) -> QuantumCircuit:
    state_circuit = circuit.copy()
    if _has_measurements(state_circuit):
        state_circuit.remove_final_measurements(inplace=True)
    return state_circuit


def get_probability_distribution(
    circuit: QuantumCircuit,
    shots: int = 1000
) -> np.ndarray:
    """
    Execute a quantum circuit and extract probability distribution.

    Uses the available primitive for the installed Qiskit version.

    Args:
        circuit: Qiskit QuantumCircuit (without measurements)
        shots: Number of measurement shots (paper: 1000)

    Returns:
        Normalized probability distribution of length 2^n_qubits
    """
    n_qubits = circuit.num_qubits
    n_outcomes = 2 ** n_qubits

    meas_circuit = _prepare_measured_circuit(circuit)
    probs = np.zeros(n_outcomes, dtype=float)

    if StatevectorSampler is not None:
        sampler = StatevectorSampler()
        result = sampler.run([meas_circuit], shots=shots).result()
        joined: Any = result[0].join_data()
        counts = joined.get_int_counts()
        total = sum(counts.values())
        if total == 0:
            raise ValueError("Sampler returned zero total counts.")
        for outcome_int, count in counts.items():
            if outcome_int < n_outcomes:
                probs[outcome_int] = count / total
    elif LegacySampler is not None:
        sampler = LegacySampler()
        result = sampler.run([meas_circuit], shots=shots).result()
        quasi_dist = result.quasi_dists[0]
        for outcome_int, prob in quasi_dist.items():
            if outcome_int < n_outcomes:
                probs[outcome_int] = prob
    else:
        raise ImportError(
            "No compatible Qiskit sampler primitive found "
            "(expected StatevectorSampler or Sampler)."
        )

    total_prob = probs.sum()
    if total_prob <= 0:
        raise ValueError("Probability vector is empty or non-positive.")
    return probs / total_prob


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

    statevector = get_statevector(circuit)
    probs = np.abs(statevector) ** 2

    if probs.shape[0] != n_outcomes:
        raise ValueError(
            f"Unexpected statevector length {probs.shape[0]} for {n_qubits} qubits."
        )
    return probs / probs.sum()


def get_statevector(circuit: QuantumCircuit) -> np.ndarray:
    """
    Get statevector from a circuit.

    Args:
        circuit: Qiskit QuantumCircuit (no measurements)

    Returns:
        Complex statevector
    """
    state_circuit = _prepare_statevector_circuit(circuit)

    if AerSimulator is not None:
        try:
            sim_circuit = state_circuit.copy()
            sim_circuit_any: Any = sim_circuit
            sim_circuit_any.save_statevector()
            sim = AerSimulator(method="statevector")
            result = sim.run(sim_circuit).result()
            return np.asarray(result.get_statevector(sim_circuit), dtype=complex)
        except Exception as exc:
            logger.warning(
                "Aer statevector simulation failed; falling back to "
                "qiskit.quantum_info.Statevector: %s",
                exc,
            )

    return np.asarray(Statevector.from_instruction(state_circuit).data, dtype=complex)
