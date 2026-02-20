"""
Ry-CZ Circular Ansatz - Hardware efficient with cyclic connectivity.

DESIGN DIFFERENCE FROM LINEAR:
Linear creates nearest-neighbor chain: 0-1-2-...-n-1
Circular adds wrap-around: 0-1-2-...-n-1-0 (closing the ring)

WHY CIRCULAR?
- Periodic boundary conditions provide rotational symmetry
- May help symmetric distributions (binomial, uniform) that benefit from symmetry
- For bimodal, symmetric connectivity might constrain the circuit (harder to break symmetry)
- Topology matches some quantum processors (IBM devices sometimes arranged in rings)

PARAMETER COUNT:
Same as linear: n × (D + 1)
The extra CZ gates don't add parameters; they just change connectivity.

CIRCUIT DIAGRAM (3 qubits, depth 1):
     ┌────────┐     ┌────────┐
q_0: ┤ RY(θ₀) ├──■──┤ RY(θ₃) ├──■──
     ├────────┤┌─┴─┐├────────┤┌─┴─┐
q_1: ┤ RY(θ₁) ├┤ Z ├┤ RY(θ₄) ├┤ Z ├──
     ├────────┤└───┘├────────┤└───┘
q_2: ┤ RY(θ₂) ├─────┤ RY(θ₅) ├──■──  ← wrap-around CZ
     └────────┘     └────────┘
     Layer 0      CZ(0,1)   Layer 1   Wrap CZ(2,0)
                  CZ(1,2)
"""

import numpy as np
from qiskit import QuantumCircuit
import logging
from src.circuits.base import BaseParameterizedCircuit

logger = logging.getLogger(__name__)


class RyCZCircularAnsatz(BaseParameterizedCircuit):
    """
    Circular ansatz using Ry rotations and CZ gates with wrap-around connectivity.

    Adds one additional CZ gate per layer (connecting last qubit back to first),
    creating a ring topology. Suitable for testing whether circular symmetry helps
    for symmetric target distributions.
    """

    def _calculate_n_params(self) -> int:
        """Parameter count: same as linear (Ry gates unchanged)."""
        return self.n_qubits * (self.depth + 1)

    def build_circuit(self, parameters: np.ndarray) -> QuantumCircuit:
        """
        Build the Ry-CZ circular circuit.

        STRUCTURE:
        1. Initial Ry layer (n parameters)
        2. For each depth layer:
           - CZ gates in linear chain: (0,1), (1,2), ..., (n-2,n-1)
           - CZ wrap-around: (n-1, 0)
           - Ry rotations (n parameters)

        WHY WRAP-AROUND?
        Closing the chain into a ring makes the connectivity symmetric:
        each qubit has exactly 2 neighbors (except in linear, end qubits have 1).
        This may help the circuit "circulate" correlations more efficiently.
        """
        self.validate_parameters(parameters)

        qc = QuantumCircuit(self.n_qubits, name="Ry-CZ Circular")

        # LAYER 0: Initial Ry
        param_idx = 0
        for qubit in range(self.n_qubits):
            qc.ry(parameters[param_idx], qubit)
            param_idx += 1

        # MAIN LOOP: Depth layers
        for layer in range(self.depth):
            # Linear CZ chain: (0,1), (1,2), ..., (n-2, n-1)
            for qubit in range(self.n_qubits - 1):
                qc.cz(qubit, qubit + 1)

            # CIRCULAR WRAP-AROUND: CZ between last and first qubit
            # This is the key difference from linear ansatz
            qc.cz(self.n_qubits - 1, 0)

            # Ry rotations
            for qubit in range(self.n_qubits):
                qc.ry(parameters[param_idx], qubit)
                param_idx += 1

        return qc


if __name__ == "__main__":
    circuit = RyCZCircularAnsatz(n_qubits=3, depth=1)
    params = np.random.uniform(0, 2*np.pi, circuit.get_parameter_count())
    qc = circuit.build_circuit(params)
    print(f"Circular ansatz: {qc.num_qubits} qubits, {len(qc)} instructions")
    print(qc)
