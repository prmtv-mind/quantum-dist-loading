"""
Ry-CZ Alternating Ansatz - Brickwork-style alternating entanglement.

DESIGN DIFFERENCE FROM LINEAR:
Linear applies: CZ(0,1), CZ(1,2), CZ(2,3), ... per layer
Alternating alternates between:
  - Even layer: CZ(0,1), CZ(2,3), CZ(4,5), ... (only even-indexed pairs)
  - Odd layer: CZ(1,2), CZ(3,4), CZ(5,6), ... (only odd-indexed pairs)

WHY ALTERNATING?
- Creates "brickwork" entanglement pattern (used in many VQA papers)
- Every qubit becomes adjacent to every other qubit (eventually) via alternating
- Higher entanglement density per layer than linear
- May saturate expressibility at lower depth, reducing optimization difficulty
- May help bimodal distributions by enabling more complex interference patterns

PARAMETER COUNT:
n × (D + 1) (same as linear; just different gate pattern)

CIRCUIT DIAGRAM (4 qubits, depth 2 - 1 full depth cycle = 2 layers):
Initial Ry layer:
     ┌────────┐
q_0: ┤ RY(θ₀) ├
     ├────────┤
q_1: ┤ RY(θ₁) ├
     ├────────┤
q_2: ┤ RY(θ₂) ├
     ├────────┤
q_3: ┤ RY(θ₃) ├
     └────────┘

Even sub-layer (CZ(0,1), CZ(2,3)):
q_0: ──■──
      ┌┴─┐
q_1: ┤ Z ├
     └───┘
q_2: ──■──
      ┌┴─┐
q_3: ┤ Z ├
     └───┘

Ry after even:
     ┌────────┐
q_0: ┤ RY(θ₄) ├
     ├────────┤
q_1: ┤ RY(θ₅) ├
     ├────────┤
q_2: ┤ RY(θ₆) ├
     ├────────┤
q_3: ┤ RY(θ₇) ├
     └────────┘

Odd sub-layer (CZ(1,2), CZ(3,?) ):
q_0: ──────
q_1: ──■──
     ┌┴─┐
q_2: ┤ Z ├
     └───┘
q_3: ──■──  (if n=4, only CZ(1,2) applied; qubit 3 has no odd partner)
"""

import numpy as np
from qiskit import QuantumCircuit
import logging
from src.circuits.base import BaseParameterizedCircuit

logger = logging.getLogger(__name__)


class RyCZAlternatingAnsatz(BaseParameterizedCircuit):
    """
    Alternating ansatz with brickwork-style CZ entanglement.

    Alternates between even-pair CZ gates and odd-pair CZ gates to create
    a denser entanglement pattern than linear nearest-neighbor.
    """

    def _calculate_n_params(self) -> int:
        """Parameter count: same as linear."""
        return self.n_qubits * (self.depth + 1)

    def build_circuit(self, parameters: np.ndarray) -> QuantumCircuit:
        """
        Build the alternating brickwork circuit.

        STRUCTURE:
        1. Initial Ry layer
        2. For each depth layer (count as 1 depth unit = 1 even + 1 odd sublayer):
           - Even sublayer: CZ(0,1), CZ(2,3), CZ(4,5), ...
           - Ry rotations
           - Odd sublayer: CZ(1,2), CZ(3,4), CZ(5,6), ...
           - Ry rotations

        DESIGN DECISION:
        Each "depth" D corresponds to D full cycles of (even + odd + Ry).
        This means alternating at depth 1 uses more CZ gates than linear at depth 1,
        but parameter count is identical (only Ry gates are parameterized).

        This creates a fairer comparison: same parameters, different connectivity.
        """
        self.validate_parameters(parameters)

        qc = QuantumCircuit(self.n_qubits, name="Ry-CZ Alternating")

        # LAYER 0: Initial Ry
        param_idx = 0
        for qubit in range(self.n_qubits):
            qc.ry(parameters[param_idx], qubit)
            param_idx += 1

        # MAIN LOOP: Depth layers
        for layer in range(self.depth):
            # Even sublayer: CZ(0,1), CZ(2,3), CZ(4,5), ...
            for qubit in range(0, self.n_qubits - 1, 2):
                qc.cz(qubit, qubit + 1)

            # Ry after even sublayer
            for qubit in range(self.n_qubits):
                qc.ry(parameters[param_idx], qubit)
                param_idx += 1

            # Odd sublayer: CZ(1,2), CZ(3,4), CZ(5,6), ...
            for qubit in range(1, self.n_qubits - 1, 2):
                qc.cz(qubit, qubit + 1)

            # Ry after odd sublayer
            for qubit in range(self.n_qubits):
                qc.ry(parameters[param_idx], qubit)
                param_idx += 1

        return qc


if __name__ == "__main__":
    circuit = RyCZAlternatingAnsatz(n_qubits=4, depth=1)
    params = np.random.uniform(0, 2*np.pi, circuit.get_parameter_count())
    qc = circuit.build_circuit(params)
    print(f"Alternating ansatz: {qc.num_qubits} qubits, {len(qc)} instructions")
    print(f"Gate counts: {circuit.get_gate_counts(qc)}")
    print(qc)
