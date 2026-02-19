"""
Base parameterized quantum circuit (PQC) class.

This module provides the foundational interface for all quantum circuit
architectures used in distribution loading experiments. All specific ansätze
(linear, circular, alternating) inherit from this base class.

QUANTUM CONCEPT:
A parameterized quantum circuit is a unitary operation U(θ) where θ is a
vector of parameters (rotation angles). By adjusting θ, we shape the
probability distribution of measurement outcomes in the computational basis.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
import logging

logger = logging.getLogger(__name__)


class BaseParameterizedCircuit(ABC):
    """
    Abstract base class for parameterized quantum circuits.

    This class defines the interface all circuit architectures must implement,
    ensuring consistent behavior across different ansatz types.

    DESIGN RATIONALE:
    - Abstract methods force subclasses to implement required functionality
    - Common initialization and utility methods reduce code duplication
    - Type hints enable IDE autocompletion and catch errors early
    """

    def __init__(self, n_qubits: int, depth: int):
        """
        Initialize a parameterized quantum circuit.

        Args:
            n_qubits: Number of qubits in the circuit
            depth: Number of layers/repetitions of the ansatz pattern

        IMPLEMENTATION NOTES:
        - Depth determines circuit expressibility: deeper circuits can
          represent more complex distributions, but take longer to optimize
        - Parameter count typically scales as n_qubits × (depth + 1)
        """
        self.n_qubits = n_qubits
        self.depth = depth
        self.n_params = self._calculate_n_params()

        logger.info(
            f"Initialized {self.__class__.__name__}: "
            f"n_qubits={n_qubits}, depth={depth}, n_params={self.n_params}"
        )

    @abstractmethod
    def _calculate_n_params(self) -> int:
        """Calculate number of parameters for this circuit architecture."""
        pass

    @abstractmethod
    def build_circuit(self, parameters: np.ndarray) -> QuantumCircuit:
        """
        Build the quantum circuit with given parameters.

        Args:
            parameters: 1D array of rotation angles (in radians)

        Returns:
            QuantumCircuit: Qiskit QuantumCircuit object

        IMPLEMENTATION NOTES:
        - Parameters should be in range [0, 2π] for Ry/Rz rotations
        - Circuit is built from |0⟩^⊗n initial state (no initialization needed)
        - No measurements are added here; those are handled during evaluation
        """
        pass

    def get_circuit_depth(self) -> int:
        """Return estimated circuit depth (layer count)."""
        return self.depth

    def get_parameter_count(self) -> int:
        """Return number of parameters in the circuit."""
        return self.n_params

    def get_gate_counts(self, circuit: QuantumCircuit) -> Dict[str, int]:
        """
        Count different gate types in the circuit.

        Args:
            circuit: Compiled QuantumCircuit

        Returns:
            Dictionary with gate names and counts

        WHY THIS MATTERS:
        Gate counts are important for benchmarking because:
        - Single-qubit gates (Ry, Rz) are typically fast (~nanoseconds)
        - Two-qubit gates (CZ, CNOT) are slow (~microseconds) and error-prone
        - More two-qubit gates → more circuit depth → more noise on real hardware
        """
        gate_counts = {}
        for instruction, qargs, cargs in circuit.data:
            gate_name = instruction.name
            gate_counts[gate_name] = gate_counts.get(gate_name, 0) + 1
        return gate_counts

    def validate_parameters(self, parameters: np.ndarray) -> bool:
        """
        Validate that parameter array has correct shape and values.

        Args:
            parameters: Parameter array to validate

        Returns:
            bool: True if valid, raises ValueError otherwise

        IMPLEMENTATION NOTES:
        - Catches common mistakes: wrong array size, NaN/inf values
        - Helps with debugging during optimization
        """
        if parameters.shape != (self.n_params,):
            raise ValueError(
                f"Expected parameters shape {(self.n_params,)}, "
                f"got {parameters.shape}"
            )

        if not np.isfinite(parameters).all():
            raise ValueError("Parameters contain NaN or infinite values")

        return True

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"{self.__class__.__name__}("
            f"n_qubits={self.n_qubits}, "
            f"depth={self.depth}, "
            f"n_params={self.n_params})"
        )
