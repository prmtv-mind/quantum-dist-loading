"""
Ry-CZ Alternating Ansatz - Brickwork-style alternating entanglement.

DESIGN DIFFERENCE FROM LINEAR:
Linear applies: CZ(0,1), CZ(1,2), CZ(2,3), ... per layer (nearest-neighbor chain)
Alternating alternates between:
  - Even sub-layer: CZ(0,1), CZ(2,3), CZ(4,5), ... (even-indexed pairs)
  - Odd sub-layer: CZ(1,2), CZ(3,4), CZ(5,6), ... (odd-indexed pairs)

Then applies a SINGLE Ry layer (not two - that was the bug).

PARAMETER COUNT: n × (D + 1) - same as linear and circular
"""

import numpy as np
from qiskit import QuantumCircuit
import logging
from src.circuits.base import BaseParameterizedCircuit

logger = logging.getLogger(__name__)


class RyCZAlternatingAnsatz(BaseParameterizedCircuit):
    """
    Alternating ansatz with brickwork-style CZ entanglement.

    Alternates between even-pair and odd-pair CZ gates to create
    a denser entanglement pattern than linear nearest-neighbor.
    """

    def _calculate_n_params(self) -> int:
        """
        Parameter count: n × (D + 1)

        - Initial Ry layer: n_qubits parameters
        - Each depth layer: n_qubits parameters (applied once, after even+odd CZ)
        - Total: n_qubits × (depth + 1)
        """
        return self.n_qubits * (self.depth + 1)

    def build_circuit(self, parameters: np.ndarray) -> QuantumCircuit:
        """
        Build the alternating brickwork circuit.

        STRUCTURE (corrected for proper parameter count):
        1. Initial Ry layer (n parameters)
        2. For each depth D:
           - Even CZ sublayer: CZ(0,1), CZ(2,3), CZ(4,5), ...
           - Odd CZ sublayer: CZ(1,2), CZ(3,4), CZ(5,6), ...
           - Single Ry layer (n parameters) applied after both sublayers

        This ensures parameter count = n_qubits × (depth + 1).

        The "alternating" refers to the CZ gate pattern (even/odd pairs),
        not to multiple Ry applications per depth.
        """
        self.validate_parameters(parameters)

        qc = QuantumCircuit(self.n_qubits, name="Ry-CZ Alternating")

        # LAYER 0: Initial Ry rotations
        param_idx = 0
        for qubit in range(self.n_qubits):
            qc.ry(parameters[param_idx], qubit)
            param_idx += 1

        # MAIN LOOP: Depth layers
        for layer in range(self.depth):
            # Even sublayer: CZ gates between even-indexed pairs
            # CZ(0,1), CZ(2,3), CZ(4,5), ...
            for qubit in range(0, self.n_qubits - 1, 2):
                qc.cz(qubit, qubit + 1)

            # Odd sublayer: CZ gates between odd-indexed pairs
            # CZ(1,2), CZ(3,4), CZ(5,6), ...
            for qubit in range(1, self.n_qubits - 1, 2):
                qc.cz(qubit, qubit + 1)

            # Apply Ry layer ONCE after both even and odd CZ sublayers
            # (not twice - that was the bug causing extra parameters)
            for qubit in range(self.n_qubits):
                qc.ry(parameters[param_idx], qubit)
                param_idx += 1

        return qc


if __name__ == "__main__":
    circuit = RyCZAlternatingAnsatz(n_qubits=4, depth=2)
    print(f"Alternating ansatz: {circuit}")
    print(f"Parameter count: {circuit.get_parameter_count()}")

    params = np.random.uniform(0, 2*np.pi, circuit.get_parameter_count())
    qc = circuit.build_circuit(params)
    print(f"Circuit instructions: {len(qc)}")
    print(f"Gate counts: {circuit.get_gate_counts(qc)}")
