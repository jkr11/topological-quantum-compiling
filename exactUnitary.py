from dataclasses import dataclass
from rings import Cyclotomic10, np, cached_property, RealCyclotomic10, N_i, N
import cmath
from typing import List, Tuple


class ExactUnitary:
  def __init__(self, u: Cyclotomic10, v: Cyclotomic10, k: int):
    self.u: Cyclotomic10 = u
    self.v: Cyclotomic10 = v
    self.k: int = k
    self.validate()

  def validate(self):
    norm_u_sq = N_i(self.u)
    norm_v_sq = N_i(self.v)
    left = (norm_u_sq.to_cycl() + (Cyclotomic10.Tau() * norm_v_sq)).evaluate()
    if left != 1:
      raise ValueError(f"Invalid exact unitary: |u|² + τ|v|² ≠ 1, was {left}")

  # @property
  # def u(self):
  #  return self.u

  # @property
  # def v(self):
  #  return self.v

  # @property
  # def k(self):
  #  return self.k

  @classmethod
  def T(self):
    return ExactUnitary(Cyclotomic10.One(), Cyclotomic10.Zero(), 6)

  @classmethod
  def F(self):
    return ExactUnitary(Cyclotomic10.Tau(), Cyclotomic10.One(), 0)

  @classmethod
  def I(self):
    return ExactUnitary(Cyclotomic10.One(), Cyclotomic10.Zero(), 5)

  def __mul__(self, other):
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

  def __rmul__(self, scalar):
    if isinstance(scalar, Cyclotomic10):
      return self.scalar_mul_left(scalar)
    else:
      raise TypeError("Left multiplication only supports Cyclotomic10 scalars")

  def _left_mul_scalar(self, scalar: Cyclotomic10):
    return ExactUnitary(self.u * scalar, self.v * scalar, self.k)

  def omega_mul(self, k: int):
    """Multiply this unitary by ω^k according to w^s * U[u,v,k] = U[uw^s,vw^s,k + 2s]"""
    omega_k = Cyclotomic10.Omega_(k)
    u_new = self.u * omega_k
    v_new = self.v * omega_k
    new_k = (self.k + 2 * k) % 10
    return ExactUnitary(u_new, v_new, new_k)

  # def to_matrix(self) -> List[List[Cyclotomic10]]:
  # TODO: need rmul for float? since we need sqrt(tau)
  #  return [[
  #      self.u, self.v * Cyclotomic10.Tau() * Cyclotomic10.Omega_(self.k)
  #  ]]

  @cached_property
  def to_numpy(self) -> np.ndarray:
    tau_val = (cmath.sqrt(5) - 1) / 2  # τ ≈ 0.618 less error than w^2 - w^3, generally not nice symbolically
    sqrt_tau = cmath.sqrt(tau_val)  # √τ ≈ 0.786 not representable by Cyclotomic10

    # Compute ω^k: ω = e^(iπ/5), ω^k = e^(iπk/5) symbolically
    omega_k = Cyclotomic10.Omega() ** self.k

    u_val = self.u.evaluate()
    v_val = self.v.evaluate()
    v_conj_val = self.v.conjugate().evaluate()
    u_conj_val = self.u.conjugate().evaluate()

    entry11 = u_val
    entry12 = v_conj_val * sqrt_tau * omega_k.evaluate()
    entry21 = v_val * sqrt_tau
    entry22 = -u_conj_val * omega_k.evaluate()

    return np.array([[entry11, entry12], [entry21, entry22]], dtype=complex)

  def __pow__(self, k: int):
    if k == 0:
      return ExactUnitary.I()
    basis = self
    rhs = self
    for i in range(1, k):
      basis = basis * rhs
    return basis

  def __eq__(self, other):
    if isinstance(other, ExactUnitary):
      return self.u == other.u and self.v == other.v and self.k == other.k % 10
    return False

  def gauss_complexity(self) -> int:
    return int(self.u.gauss_complexity().evaluate().real)

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
    return f"U{str(self.u), str(self.v), self.k}\n {self.to_numpy}"
