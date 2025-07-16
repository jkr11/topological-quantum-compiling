import numpy as np
from single_qubit.exact_synthesis.util import trace_norm
from abc import ABC, abstractmethod

class AnyonModel(ABC):
  """Abstract base class for any anyon model."""

  def fusion_rules(self):
    raise NotImplementedError

  def R_matrix(self, a, b, c):
    """Braiding matrix for a × b → c"""
    raise NotImplementedError

  def F_matrix(self, a, b, c, d, e):
    """F-symbol for ((a×b)×c) → (a×(b×c))"""
    raise NotImplementedError

  def get_hilbert_space(self, n: int, total_charge):
    """Construct the Hilbert space of n anyons with total charge."""
    raise NotImplementedError

  @abstractmethod
  def braid_operator(self, i: int, n: int):
    pass


class FibonacciModel(AnyonModel):
  particles = ["1", "τ"]

  def fusion_rules(self):
    return {("τ", "τ"): ["1", "τ"], ("τ", "1"): ["τ"], ("1", "τ"): ["τ"]}

  def R_matrix(self, a, b, c):
    if (a, b, c) == ("τ", "τ", "1"):
      return np.exp(-4j * np.pi / 5)
    elif (a, b, c) == ("τ", "τ", "τ"):
      return np.exp(3j * np.pi / 5)
    else:
      return 1.0

  def F_matrix(self, a, b, c, d, e):
    φ = (1 + np.sqrt(5)) / 2  # golden ratio
    inv_φ = 1 / φ
    if all(x == "τ" for x in [a, b, c, d, e]):
      return np.array([[inv_φ, np.sqrt(inv_φ)], [np.sqrt(inv_φ), -inv_φ]])
    else:
      return np.array([[1]])

  def R(self):
    return np.diag([self.R_matrix("τ", "τ", "1"), self.R_matrix("τ", "τ", "τ")])

  def F(self):
    φ = (1 + np.sqrt(5)) / 2  # golden ratio
    inv_φ = 1 / φ
    return np.array([[inv_φ, np.sqrt(inv_φ)], [np.sqrt(inv_φ), -inv_φ]])


  @classmethod
  def braid_operator(self, i, n):
    if n == 3:
      if i == 1:
        return self.R
      elif i == 2:
        return self.F @ self.R @ self.F



class PfaffianBasis:
  @classmethod
  def basis_12(self):
    return np.array([[1, 0], [0, 1j]], dtype=complex)

  @classmethod
  def basis_23(self):
    return 1 / np.sqrt(2) * np.exp(1j * np.pi / 4) * np.array([[1, -1j], [-1j, 1]])

  @classmethod
  def basis_34(self):
    return self.basis_12()

  @classmethod
  def braid_operator(self, k: int, n: int):
    if n == 6:
      if k == 1:
        return np.kron(self.basis_12(), np.eye(2))
      elif k == 2:
        return np.kron(self.basis_23(), np.eye(2))
      elif k == 3:
        return np.diag([1, 1j, 1j, 1])
      elif k == 4:
        return np.kron(np.eye(2), self.basis_23())
      elif k == 5:
        return np.kron(np.eye(2), self.basis_34())


if __name__ == "__main__":
  print(PfaffianBasis.basis(3,6))
