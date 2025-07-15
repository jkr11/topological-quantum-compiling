from single_qubit.exact_synthesis.exactUnitary import ExactUnitary
import mpmath
from typing import Union, List
from qiskit.circuit.library import UnitaryGate
from qiskit import QuantumCircuit
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pyzx import Circuit as xzCircuit


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
class Gate:
  name: str
  matrix: operator


type Circuit = List[Gate]


class TranslationEngine(ABC):
  @abstractmethod
  def translate(self, circuit: Circuit):
    pass


class QiskitTranspiler(TranslationEngine):
  def translate(self, circuit) -> QuantumCircuit:
    qc = QuantumCircuit(1)  # TODO: n_qubits

    for gate in circuit:
      u_gate = UnitaryGate(gate.matrix, lable=gate.name)
      qc.append(u_gate)
    return qc


class XZTranspiler(TranslationEngine):
  def translate(self, circuit) -> xzCircuit:
    pass


@dataclass(frozen=True)
class Sigma1:
  power: int


@dataclass(frozen=True)
class Sigma2:
  power: int


s1 = Fibonacci.sigma1("np")
s2 = Fibonacci.sigma2("np")


def braid_matrix(braid: List):
  U = np.eye(2, dtype=complex)
  for gate in braid:
    if isinstance(gate, Sigma1):
      M = np.linalg.matrix_power(s1, gate.power)
    elif isinstance(gate, Sigma2):
      M = np.linalg.matrix_power(s2, gate.power)
    else:
      raise ValueError(f"Unknown gate: {gate}")
    U = U @ M
  return U


def invert_braid(braid: List[Gate]) -> List[Gate]:
  inverted = []
  for gate in reversed(braid):
    if isinstance(gate, Sigma1):
      inverted.append(Sigma1(-gate.power))
    elif isinstance(gate, Sigma2):
      inverted.append(Sigma2(-gate.power))
    else:
      raise ValueError(f"Unknown gate type: {gate}")
  return inverted


INJECTION_WEAVE = [
  Sigma2(3),
  Sigma1(2),
  Sigma2(-4),
  Sigma1(2),
  Sigma2(2),
  Sigma1(-2),
  Sigma2(-2),
  Sigma1(-2),
  Sigma2(2),
  Sigma1(2),
  Sigma2(2),
  Sigma1(-2),
  Sigma2(2),
  Sigma1(-2),
  Sigma2(4),
  Sigma1(-2),
  Sigma2(2),
  Sigma1(4),
  Sigma2(2),
  Sigma1(-2),
  Sigma2(1),
]

IW = [
  Sigma2(3),
  Sigma1(-2),
  Sigma2(-4),
  Sigma1(2),
  Sigma2(4),
  Sigma1(2),
  Sigma2(-2),
  Sigma1(-2),
  Sigma2(-4),
  Sigma1(-4),
  Sigma2(-2),
  Sigma1(4),
  Sigma2(2),
  Sigma1(-2),
  Sigma2(2),
  Sigma1(2),
  Sigma2(-2),
  Sigma1(3)
]


if __name__ == "__main__":
  A = braid_matrix(IW)
  B = braid_matrix(invert_braid(IW))



  print(A)
  print(B)
  print(A @ B)
