from single_qubit.exact_synthesis.exactUnitary import ExactUnitary
from single_qubit.exact_synthesis.synthesis import Gate, FGate, TGate, Sigma1, Sigma2, WIGate
import mpmath
from typing import Union, List
from qiskit.circuit.library import UnitaryGate
from qiskit import QuantumCircuit
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pyzx import Circuit as xzCircuit
from qiskit.quantum_info import Operator


type mpmatrix = mpmath.matrices.matrices.matrix


class Fibonacci:
  @classmethod
  def sigma1(self, backend: str = "np") -> Union[np.ndarray, mpmatrix]:
    if backend == "np":
      return ExactUnitary.sigma1().to_numpy()
    elif backend == "mpmath":
      return ExactUnitary.to_matrix
    else:
      raise ValueError(f"Unknown backend: {backend!r}. Expected 'np' or 'mpmath'.")

  @classmethod
  def sigma2(self, backend: str = "np") -> Union[np.ndarray, mpmatrix]:
    if backend == "np":
      return ExactUnitary.sigma2().to_numpy()
    elif backend == "mpmath":
      return ExactUnitary.to_matrix
    else:
      raise ValueError(f"Unknown backend: {backend!r}. Expected 'np' or 'mpmath'.")


type operator = Union[np.ndarray, mpmatrix]


@dataclass
class Unitary:
  name: str
  power: int
  matrix: operator


def gate_to_operator(gates: List[Gate]) -> List[Unitary]:
  output = []
  for gate in gates:
    if isinstance(gate, WIGate):
      output.append(Unitary("w", gate.power, ExactUnitary.I().omega_mul(gate.power).to_numpy()))
    elif isinstance(gate, FGate):
      output.append(Unitary("F", gate.power, ExactUnitary.F().to_numpy()))
    elif isinstance(gate, TGate):
      output.append(Unitary("T", gate.power, ExactUnitary.T().to_numpy()))

  return output


type SingleCircuit = List[Unitary]


class SingleQbitTranspiler(ABC):
  @abstractmethod
  def translate(self, circuit: SingleCircuit):
    pass


class QiskitTranspiler(SingleQbitTranspiler):
  @classmethod
  def translate(self, circuit) -> QuantumCircuit:
    qc = QuantumCircuit(1)  # TODO: n_qubits
    circuit = gate_to_operator(circuit)
    for gate in circuit:
      u_gate = UnitaryGate(np.asarray(gate.matrix), label=gate.name)
      for i in range(gate.power):
        qc.append(u_gate, [0])
    return qc


class XZTranspiler(SingleQbitTranspiler):
  def translate(self, circuit) -> xzCircuit:
    pass


if __name__ == "__main__":
  from single_qubit.exact_synthesis.synthesis import ExactFibonacciSynthesizer, evaluate_gate_sequence, norm

  mpmath.mp.dps = 200
  phi = 4 * mpmath.pi / 1000
  epsilon = 1e-20

  EU = ExactFibonacciSynthesizer.synthesize_z_rotation(phi, epsilon)
  gates = ExactFibonacciSynthesizer._exact_synthesize(EU)
  ugates = gate_to_operator(gates)
  qc = QiskitTranspiler.translate(ugates)
  print(qc.draw())
  unit = Operator(qc).data
  print(unit)
  from single_qubit.gates import Gates
  from single_qubit.exact_synthesis.synthesis import d_z

  print(evaluate_gate_sequence(gates).to_numpy())
  print(Gates.Rz(float(phi)))
  # print(d_z(phi, EU))
