"""
Ry-CZ Linear Ansatz - Hardware efficient circuit with linear connectivity.

DESIGN PHILOSOPHY:
Linear connectivity (only nearest-neighbor gates) is realistic for many
quantum hardware platforms. This ansatz is simple but expressive enough
to load moderate probability distributions on 3-4 qubits.

CIRCUIT STRUCTURE:
For depth D on n qubits:
1. Initial Ry layer: n parameters (one per qubit)
2. Repeat D times:
   - Apply CZ gates between (0,1), (1,2), ..., (n-2,n-1)
   - Apply Ry rotations: n parameters (one per qubit)

Total parameters: n × (D + 1)

QUANTUM MECHANICS INTUITION:
- Ry(θ) rotates the state around the Y-axis in the Bloch sphere
- CZ is a controlled-phase gate that entangles qubits
- The combination creates complex superpositions that can approximate
  any probability distribution (for sufficient depth)
"""

import numpy as np
from qiskit import QuantumCircuit
import logging
from src.circuits.base import BaseParameterizedCircuit

logger = logging.getLogger(__name__)


class RyCZLinearAnsatz(BaseParameterizedCircuit):
    """
    Linear ansatz using Ry single-qubit rotations and CZ two-qubit gates.

    This is one of the most commonly used hardware-efficient ansätze because:
    1. Linear connectivity matches many real quantum processors
    2. CZ gates are often faster and more accurate than CNOT
    3. The expressiveness grows with depth while maintaining simplicity
    """

    def _calculate_n_params(self) -> int:
        """
        Calculate total parameter count.

        For linear ansatz:
        - Initial Ry layer: n_qubits parameters
        - Each depth layer: n_qubits parameters (after CZ gates)
        - Total: n_qubits × (depth + 1)

        DESIGN NOTE:
        Parameter count determines optimization difficulty:
        - More parameters = more expressibility but slower optimization
        - Need enough parameters to represent target distribution
        - Too many parameters = prone to barren plateaus (flat optimization landscape)
        """
        return self.n_qubits * (self.depth + 1)

    def build_circuit(self, parameters: np.ndarray) -> QuantumCircuit:
        """
        Build the Ry-CZ linear circuit with given parameters.

        Args:
            parameters: 1D array of length n_qubits × (depth + 1)

        Returns:
            QuantumCircuit: The constructed circuit ready for simulation

        IMPLEMENTATION STRATEGY:
        We organize parameters in layers:
        - Layer 0: parameters[0:n_qubits] → initial Ry rotations
        - Layer 1: parameters[n_qubits:2*n_qubits] → Ry after first CZ layer
        - etc.

        This makes the code readable and the optimization trajectory traceable.
        """
        # Validate before building
        self.validate_parameters(parameters)

        # Create quantum circuit
        qc = QuantumCircuit(self.n_qubits, name="Ry-CZ Linear")

        # LAYER 0: Initial Ry rotations
        # These set up the initial superposition state
        param_idx = 0
        for qubit in range(self.n_qubits):
            qc.ry(parameters[param_idx], qubit)
            param_idx += 1

        # MAIN LOOP: Repeat the entangling pattern 'depth' times
        for layer in range(self.depth):
            # Add CZ gates in linear chain: (0,1), (1,2), ..., (n-2,n-1)
            #
            # WHY CZ IN LINEAR CHAIN?
            # - CZ between qubits i and i+1 creates local entanglement
            # - Linear pattern maps to hardware topology of most QPUs
            # - Minimizes circuit depth (no compilation overhead)
            for qubit in range(self.n_qubits - 1):
                qc.cz(qubit, qubit + 1)

            # Add Ry rotations in this layer
            # These act on the entangled state from CZ gates
            for qubit in range(self.n_qubits):
                qc.ry(parameters[param_idx], qubit)
                param_idx += 1

        return qc

    def __repr__(self) -> str:
        """Descriptive string for logging and debugging."""
        return (
            f"RyCZLinearAnsatz("
            f"n_qubits={self.n_qubits}, "
            f"depth={self.depth}, "
            f"n_params={self.n_params})"
        )


# Example usage and testing
if __name__ == "__main__":
    # Create a small circuit for testing
    circuit = RyCZLinearAnsatz(n_qubits=3, depth=1)
    print(f"Circuit: {circuit}")
    print(f"Parameter count: {circuit.get_parameter_count()}")

    # Generate random parameters
    params = np.random.uniform(0, 2*np.pi, circuit.get_parameter_count())

    # Build and visualize
    qc = circuit.build_circuit(params)
    print("\nCircuit:")
    print(qc)
    print("\nGate counts:")
    print(circuit.get_gate_counts(qc))
