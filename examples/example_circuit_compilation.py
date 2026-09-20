from qiskit.circuit.library import UnitaryGate, UGate
from qiskit import transpile, QuantumCircuit
from qiskit.circuit import CircuitInstruction
from qiskit.quantum_info import random_unitary
import qiskit
from functools import lru_cache
from exact_synthesis.synthesis import synthesize_unitary


class QiskitToAnyonTranspiler:
  def __init__(self, accuracy: float):
    self.accuracy = accuracy

  def _preprocess(self, circuit: QuantumCircuit) -> QuantumCircuit:
    n = circuit.num_qubits
    line_coupling = [[i, i + 1] for i in range(n - 1)] + [[i + 1, i] for i in range(n - 1)]  # LNN graph
    return qiskit.transpile(circuit, basis_gates=["cx", "u"], coupling_map=line_coupling, optimization_level=3)

  def _classify_instruction(self, operation):
    #assert isinstance(operation, CircuitInstruction)
    print(operation)
    op_name = operation.name
    qubits = [q._index for q in operation.qubits]
    if op_name == "u":
      target_qubit = qubits[0]
      params = operation.params
      unitary = UGate(*params).to_matrix()  # Hmm i might need to write a pass after all that decomposes into a sort of euler angle decomp
      return "u", unitary, target_qubit
    elif op_name == "cx":
      control_qubit = qubits[0]
      target_qubit = qubits[1]
      return "cx", target_qubit, control_qubit
    else: 
      print(op_name)
      raise Exception("This must be a normal form circuit.")

  @lru_cache
  def approximate_cnot_gate(self):
    pass

  def compile(self, circuit: QuantumCircuit):
    qc = self._preprocess(circuit)
    n = circuit.num_qubits
    output = [[] * n]
    for operation in qc.data:
      res = self._classify_instruction(operation)
      if res[0] == "u":
        approx = synthesize_unitary(res[1], self.accuracy)
        output[res[2]].append(approx)
      elif res[0] == "cx":
        output[res[1]].append("target")
        output[res[2]].append("control")
    return output


if __name__ == "__main__":
  n = 4
  U_matrix = random_unitary(4, seed=42).data
  qc = QuantumCircuit(n)
  qc.append(UnitaryGate(U_matrix), [0, 1])
  qc.append(UnitaryGate(U_matrix), [0, 2])
  qta = QiskitToAnyonTranspiler(1e-10)

  #line_coupling = [[i, i + 1] for i in range(n - 1)] + [[i + 1, i] for i in range(n - 1)]  # LNN graph
  #normal_form = transpile(qc, basis_gates=["cx", "u"], coupling_map=line_coupling, optimization_level=3)
  normal_form = qta._preprocess(qc)
  for operation in normal_form.data:
    print(operation.name)
  ll = qta.compile(qc)
  print(ll)
