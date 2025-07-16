from dataclasses import dataclass
from typing import List
from single_qubit._transpiler import Fibonacci, Unitary
import numpy as np
from abc import abstractmethod
import abc


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


def invert_braid(braid: List[Unitary]) -> List[Unitary]:
  inverted = []
  for gate in reversed(braid):
    if isinstance(gate, Sigma1):
      inverted.append(Sigma1(-gate.power))
    elif isinstance(gate, Sigma2):
      inverted.append(Sigma2(-gate.power))
    else:
      raise ValueError(f"Unknown gate type: {gate}")
  return inverted


# TODO: move to multi_qubit
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
  Sigma1(3),
]


@dataclass
class Gate:
  matrix: Unitary
  nqbits: int


@dataclass
class Circuit:
  gates: List[Gate]
  nqubits: int


class AbstractCircuitTranspiler(abc.ABC):
  @abstractmethod
  def translate(self, circuit: Circuit):
    pass


if __name__ == "__main__":
  A = braid_matrix(IW)
  B = braid_matrix(invert_braid(IW))

  print(A)
  print(B)
  print(A @ B)
