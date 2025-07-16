from single_qubit.exact_synthesis.exactUnitary import ExactUnitary
import mpmath
from typing import Union, List
from qiskit.circuit.library import UnitaryGate
from qiskit import QuantumCircuit
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pyzx import Circuit as xzCircuit
from scipy.optimize import minimize, minimize_scalar
from .gates import Gates

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
  matrix: operator


type Circuit = List[Unitary]


class SingleQbitTranspiler(ABC):
  @abstractmethod
  def translate(self, circuit: Circuit):
    pass


class QiskitTranspiler(SingleQbitTranspiler):
  def translate(self, circuit) -> QuantumCircuit:
    qc = QuantumCircuit(1)  # TODO: n_qubits

    for gate in circuit:
      u_gate = UnitaryGate(gate.matrix, lable=gate.name)
      qc.append(u_gate)
    return qc


class XZTranspiler(SingleQbitTranspiler):
  def translate(self, circuit) -> xzCircuit:
    pass
