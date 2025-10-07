from functools import cached_property
from single_qubit.exact_synthesis.rings import Cyclotomic10, ZTau, N_i
import numpy as np
from typing import List
import mpmath


class ExactUnitary:
  def __init__(self, u: Cyclotomic10, v: Cyclotomic10, k: int):
    self.u: Cyclotomic10 = u
    self.v: Cyclotomic10 = v
    self.k: int = k % 10
    self.validate()

  def validate(self) -> None:
    norm_u_sq = N_i(self.u)
    norm_v_sq = N_i(self.v)
    left = norm_u_sq + (ZTau.Tau() * norm_v_sq)
    if left != ZTau.One():
      raise ValueError(f"Invalid exact unitary: |u|² + τ|v|² ≠ 1, was {left.evaluate()} != 1")

  @classmethod
  def T(self) -> "ExactUnitary":
    return ExactUnitary(Cyclotomic10.One(), Cyclotomic10.Zero(), 6)

  @classmethod
  def F(self) -> "ExactUnitary":
    return ExactUnitary(Cyclotomic10.Tau(), Cyclotomic10.One(), 0)

  @classmethod
  def I(self) -> "ExactUnitary":
    return ExactUnitary(Cyclotomic10.One(), Cyclotomic10.Zero(), 5)

  def __mul__(self, other) -> "ExactUnitary":
    if other == self.I():
      return self
    elif self == self.I():
      return other
    elif other == self.T():
      return ExactUnitary(self.u, self.v, (self.k + 1) % 10)
    elif self == self.T():
      return ExactUnitary(other.u, other.v * Cyclotomic10.Omega(), other.k + 1 % 10)
    u3 = self.u * other.u + self.v.conjugate() * other.v * Cyclotomic10.Tau() * Cyclotomic10.Omega_(self.k)
    v3 = self.v * other.u - self.u.conjugate() * other.v * Cyclotomic10.Omega_(self.k)
    k = (other.k + self.k) % 10
    return ExactUnitary(u3, v3, k + 5 % 10)

  def __matmul__(self, other) -> "ExactUnitary":
    if not isinstance(other, ExactUnitary):
      raise ValueError(f"Can only @ with ExactUnitary, got {type(other)}")
    return self.__mul__(other)

  # def __rmul__(self, scalar):
  #   if isinstance(scalar, Cyclotomic10):
  #     return self.scalar_mul_left(scalar)
  #   else:
  #     raise TypeError("Left multiplication only supports Cyclotomic10 scalars")

  # def _left_mul_scalar(self, scalar: Cyclotomic10):
  #   return ExactUnitary(self.u * scalar, self.v * scalar, self.k)

  def omega_mul(self, k: int):
    """Multiply this unitary by ω^k according to w^s * U[u,v,k] = U[uw^s,vw^s,k + 2s]"""
    omega_k = Cyclotomic10.Omega_(k)
    u_new = self.u * omega_k
    v_new = self.v * omega_k
    new_k = (self.k + 2 * k) % 10
    return ExactUnitary(u_new, v_new, new_k)

  def __hash__(self):
    return hash((self.u, self.v, self.k))

  def to_numpy(self, threshold: float = None) -> np.ndarray:
    matrix = np.matrix(self.to_matrix.tolist(), dtype=np.complex256).reshape(2, 2)
    if threshold:
      matrix[np.abs(matrix) < threshold] = 0
    return matrix

  @cached_property
  def to_matrix(self):
    tau = (mpmath.sqrt(5) - 1) / 2
    sqrt_tau = mpmath.sqrt(tau)

    omega_k = Cyclotomic10.Omega() ** self.k

    u_val = self.u.evaluate()
    v_val = self.v.evaluate()
    v_conj_val = self.v.conjugate().evaluate()
    u_conj_val = self.u.conjugate().evaluate()

    entry11 = u_val
    entry12 = v_conj_val * sqrt_tau * omega_k.evaluate()
    entry21 = v_val * sqrt_tau
    entry22 = -u_conj_val * omega_k.evaluate()

    return mpmath.matrix([[entry11, entry12], [entry21, entry22]], dtype=complex)

  def __pow__(self, k: int) -> "ExactUnitary":
    if not isinstance(k, int):
      raise ValueError(f"exponent has to be integer, was {type(k)}")
    if k < 0:
      raise ValueError(f"exponent has to be positive integer, was {k}")
    if k == 0:
      return ExactUnitary.I()
    basis = self
    rhs = self
    for i in range(1, k):
      basis = basis * rhs
    return basis

  def __eq__(self, other) -> bool:
    if isinstance(other, ExactUnitary):
      return self.u == other.u and self.v == other.v and self.k == other.k % 10
    else:
      raise ValueError(f"Can only compare ExactUnitary to ExactUnitary, but was {type(other)}")

  def gauss_complexity(self) -> int:
    return self.u.gauss_complexity().evaluate()

  def conjugate(self):
    return ExactUnitary(self.u.conjugate(), self.v.conjugate(), -self.k % 10)

  def transpose(self):
    return ExactUnitary(self.u, self.v * Cyclotomic10.Omega_(-self.k % 10), self.k)

  @classmethod
  def from_gates_string(self, gates: List[str]):
    unitary = self.I()
    for g in reversed(gates):
      if g == "T":
        unitary = unitary * self.T()
      elif g == "F":
        unitary = unitary * self.F()
      elif g == "W":
        unitary = unitary.omega_mul(1)
    return unitary

  def __repr__(self):
    return f"U{str(self.u), str(self.v), self.k}"

  @classmethod
  def sigma1(self):
    """
    Returns sigma1 as an exact unitary from the <F,T> circuit (wI)^6 T^7
    """
    wI6 = self.I().omega_mul(6)
    return wI6 @ (self.T() ** 7)

  @classmethod
  def sigma2(self):
    """
    Returns sigma2 as an exact unitary (wI)^6 F T^7 F
    """
    wI6 = self.I().omega_mul(6)
    return wI6 @ self.F() @ (self.T() ** 7) @ self.F()
