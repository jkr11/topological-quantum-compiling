from functools import lru_cache

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import random_unitary

from exact_synthesis.synthesis import synthesize_unitary


class QiskitToAnyonTranspiler:
  def __init__(self, accuracy: float):
    self.accuracy = accuracy

  def _preprocess(self, circuit: QuantumCircuit) -> QuantumCircuit:
    n = circuit.num_qubits
    line_coupling = [[i, i + 1] for i in range(n - 1)] + [[i + 1, i] for i in range(n - 1)]
    return transpile(
      circuit,
      basis_gates=["cx", "u"],
      coupling_map=line_coupling or None,
      initial_layout=list(range(n)),
      optimization_level=3,
    )

  def _classify_instruction(self, circuit: QuantumCircuit, instruction):
    operation = instruction.operation
    qubits = [circuit.find_bit(q).index for q in instruction.qubits]
    if operation.name == "u":
      return "u", qubits, operation.to_matrix()
    if operation.name == "cx":
      return "cx", qubits, None
    raise ValueError(f"Unsupported gate '{operation.name}': circuit must be in {{cx, u}} normal form.")

  @lru_cache
  def approximate_cnot_gate(self):
    raise NotImplementedError

  def compile(self, circuit: QuantumCircuit) -> list[list]:
    qc = self._preprocess(circuit)
    output = [[] for _ in range(qc.num_qubits)]
    for instruction in qc.data:
      kind, qubits, unitary = self._classify_instruction(qc, instruction)
      if kind == "u":
        output[qubits[0]].append(synthesize_unitary(unitary, self.accuracy))
      else:
        control, target = qubits
        output[control].append("control")
        output[target].append("target")
    return output


if __name__ == "__main__":
  import mpmath
  mpmath.mp.dps = 400
  n = 4
  U_matrix = random_unitary(4, seed=42).data
  qc = QuantumCircuit(n)
  qc.unitary(U_matrix, [0, 1])
  qc.unitary(U_matrix, [0, 2])
  qta = QiskitToAnyonTranspiler(1e-20)

  normal_form = qta._preprocess(qc)
  for instruction in normal_form.data:
    print(instruction.operation.name)
  print(qta.compile(qc))