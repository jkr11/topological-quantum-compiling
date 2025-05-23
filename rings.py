import math
import numpy as np
import cmath
from functools import cached_property, total_ordering
from fractions import Fraction
from typing import List


# @total_ordering
class Cyclotomic10:
  def __init__(self, c0, c1, c2, c3):
    self.c0: int = c0
    self.c1: int = c1
    self.c2: int = c2
    self.c3: int = c3

  def coeffs(self) -> List[int]:
    return [self.c0, self.c1, self.c2, self.c3]

  def __mul__(self, other):
    exponents_coeffs = [
      (1, 0, 0, 0),  # ζ^0
      (0, 1, 0, 0),  # ζ^1
      (0, 0, 1, 0),  # ζ^2
      (0, 0, 0, 1),  # ζ^3
      (-1, 1, -1, 1),  # ζ^4
      (-1, 0, 0, 0),  # ζ^5
      (0, -1, 0, 0),  # ζ^6
    ]
    result = [0, 0, 0, 0]
    a = self.coeffs()
    b = other.coeffs() if isinstance(other, Cyclotomic10) else other
    for i in range(4):
      for j in range(4):
        k = i + j
        coeffs = exponents_coeffs[k] if k < 7 else (0, 0, 0, 0)
        term = a[i] * b[j]
        result[0] += term * coeffs[0]
        result[1] += term * coeffs[1]
        result[2] += term * coeffs[2]
        result[3] += term * coeffs[3]
    return Cyclotomic10(*result)

  def __rmul__(self, other):
    return self.__mul__(other)

  def __pow__(self, exponent):
    if isinstance(exponent, int):
      if self == Cyclotomic10.One():
        return Cyclotomic10.One()
      elif self == Cyclotomic10.Omega():
        if exponent % 10 == 5:
          return -self.One()
        elif exponent % 10 == 0:
          return Cyclotomic10.One()
      result = self.One()
      for _ in range(exponent):
        result = result * self
      return result
    else:
      return NotImplemented

  @classmethod
  def from_int(self, other: int):
    return self(other, 0, 0, 0)

  @classmethod
  def One(self):
    return self.from_int(1)

  @classmethod
  def Zero(self):
    return self.from_int(0)

  @classmethod
  def Tau(self):
    return Cyclotomic10(0, 0, 1, -1)

  @classmethod
  def Omega(self):
    return Cyclotomic10(0, 1, 0, 0)

  @classmethod
  def Omega_(self, k: int):
    return self.Omega() ** k

  # @cached_property
  def conjugate(self):
    c0, c1, c2, c3 = self.coeffs()
    return Cyclotomic10(c0 + c1, -c1, c1 - c3, -c1 - c2)

  # @cached_property
  def automorphism(self):
    c0, c1, c2, c3 = self.coeffs()
    return Cyclotomic10(c0 + c3, -c2 - c3, c3, c1 - c3)

  def galois_automorphism(self, k: int):
    if k == 1:
      return self
    elif k == 3:
      return self.automorphism()
    elif k == 7:
      c0, c1, c2, c3 = self.coeffs()
      return Cyclotomic10(c0, c3, -c1, 0) + self.from_omega_4(c2)
    elif k == 9:
      return self.automorphism().automorphism()
    else:
      raise NotImplementedError

  def norm_squared(self):
    product = self * self.conjugate()
    c0, c1, c2, c3 = product.coeffs()
    # return RealCyclotomic10(c0, c2)
    return product

  def inv(self):
    conjs = self.galois_automorphism(3) * self.galois_automorphism(7) * self.galois_automorphism(9)
    a, b, c, d = conjs.coeffs()
    N = self.galois_norm()
    return Cyclotomic10(a // N, b // N, c // N, d // N)

  def pseudo_inv(self):
    conjs = self.galois_automorphism(3) * self.galois_automorphism(7) * self.galois_automorphism(9)
    return Cyclotomic10(*conjs.coeffs())

  def galois_automorphism_product(self):
    return self.galois_automorphism(3) * self.galois_automorphism(7) * self.galois_automorphism(9)

  def __divmod__(self, other):
    def rounddiv(x: int, y: int):
      return (x + y // 2) // y if y > 0 else (x - (-y) // 2) // y

    if isinstance(other, int):
      other = self.from_int(other)
    elif isinstance(other, self.__class__):
      p = self * other.galois_automorphism_product()

      k = other.galois_norm()

      q_coeffs = [rounddiv(c, k) for c in p.coeffs()]
      q = self.__class__(*q_coeffs)

      r = self - other * q
      return q, r
    else:
      return NotImplemented

  def galois_norm(self) -> int:
    norm = self * self.galois_automorphism(3) * self.galois_automorphism(7) * self.galois_automorphism(9)
    return norm.to_subring().to_int()

  def norm_i(self):
    return self * self.conjugate()

  def norm_tau(self):
    return self * self.automorphism()

  def N(self):
    return self.norm_i().norm_tau()

  def integer_remainder_mod_one_plus_omega(self):
    a, b, c, d = self.coeffs()
    return a - b + c - d

  @classmethod
  def from_omega_4(self, k: int):
    return self(-k, k, -k, k)

  def is_unit(self):
    return self.galois_norm() == 1

  def mod_one_plus_omega(self) -> int:
    """
    Compute eta mod (1 + ω) by substituting ω = -1.
    The result is an integer (ℤ), in the range {0, ±1, ±2}.
    """
    c0, c1, c2, c3 = self.coeffs()
    result = c0 - c1 + c2 - c3
    return result % 5  # modulo N(1 + ω)

  def divides_by_one_plus_omega(self) -> bool:
    return self.galois_norm() % 5 == 0 and self.mod_one_plus_omega() == 0

  def evaluate(self):
    theta = math.pi / 5  # ω = e^(πi/5)
    omega = complex(math.cos(theta), math.sin(theta))
    total = 0j
    for i in range(4):
      total += self.coeffs()[i] * (omega**i)
    return total

  def to_subring(self):
    a, b, c, d = self.coeffs()
    if b == 0 and c == -d:
      return RealCyclotomic10(a, c)
    else:
      raise ValueError(f"Tau is not represented in {self}")

  def gauss_complexity(self):
    u1 = self.norm_squared()
    u2 = self.automorphism().norm_squared()
    return u1 + u2

  def div_by_one_plus_omega(self):
    one_plus_omega = Cyclotomic10(1, 1, 0, 0)
    if self.galois_norm() % 5 == 0:
      inter = self * one_plus_omega.pseudo_inv()
      print("Intermediate result in div by 1 + w: ", inter)
      a, b, c, d = inter.coeffs()
      a1 = a // 5
      b1 = b // 5
      c1 = c // 5
      d1 = d // 5
      return Cyclotomic10(a1, b1, c1, d1)
    else:
      raise ValueError("Norm not divable by 5")

  def __eq__(self, other):
    return self.coeffs() == other.coeffs()

  def __add__(self, other):
    return Cyclotomic10(
      self.coeffs()[0] + other.coeffs()[0],
      self.coeffs()[1] + other.coeffs()[1],
      self.coeffs()[2] + other.coeffs()[2],
      self.coeffs()[3] + other.coeffs()[3],
    )

  def __neg__(self):
    return self.__class__(-self.c0, -self.c1, -self.c2, -self.c3)

  def __sub__(self, other):
    if isinstance(other, self.__class__):
      return self + (-other)
    elif isinstance(other, int):
      return self + (-self.__class__.from_int(other))

  def __str__(self):
    labels = ["", "ω", "ω²", "ω³"]
    return " + ".join(f"{c}{l}" for c, l in zip(self.coeffs(), labels) if c != 0) or "0"

  def __repr__(self):
    return f"Cyclotomic10{self.coeffs()}"


TAU = (math.sqrt(5) - 1) / 2
PHI = TAU + 1


class RealCyclotomic10:
  def __init__(self, a: int, b: int):
    self.a: int = a
    self.b: int = b

  def __mul__(self, other):
    # (a + bτ)(c + dτ) = (ac + bd) + (ad + bc - bd)τ
    a, b = self.a, self.b
    c, d = other.a, other.b

    real_part = a * c + b * d
    tau_part = a * d + b * c - b * d

    return RealCyclotomic10(real_part, tau_part)

  def evaluate(self):
    phi_conjugate = (math.sqrt(5) - 1) / 2
    return self.a + self.b * phi_conjugate

  def automorphism(self):
    """applies the automorphism w -> w^3 to 'self', on tau = w^2 - w^3. aut(tau) = -phi, where phi = 1 + tau. Thus aut(a + btau) = a - (1 + tau)b"""
    return RealCyclotomic10(self.a - self.b, -self.b)

  def norm(self) -> int:
    return self * self.automorphism()

  def conjugate(self):
    return self.automorphism().automorphism()

  def from_int(self, n: int):
    return RealCyclotomic10(n, 0)

  def to_cycl(self) -> Cyclotomic10:
    return Cyclotomic10(self.a, 0, self.b, -self.b)

  def to_int(self) -> int:
    if self.b == 0:
      return self.a
    else:
      raise NotImplementedError

  def div_by_two_minus_tau(self):
    num = self * RealCyclotomic10(2, -1)
    if num.a % 5 != 0 or num.b % 5 != 0:
      raise ValueError("Division not closed in Z[tau]")
    print("dividing")
    return RealCyclotomic10(num.a // 5, num.b // 5)

  def __add__(self, other):
    return RealCyclotomic10(self.a + other.a, self.b + other.b)

  def __sub__(self, other):
    return RealCyclotomic10(self.a - other.a, self.b - other.b)

  def __eq__(self, other):
    if isinstance(other, int):
      return self == Cyclotomic10.from_int(other)
    return self.a == other.a and self.b == other.b

  def __repr__(self):
    if self.b >= 0:
      if self.b == 1:
        return f"{self.a} + τ"
      return f"{self.a} + {self.b}τ"
    else:
      if self.b == -1:
        return f"{self.a} - τ"
      return f"{self.a} - {-self.b}τ"


# A7
def N_tau(xi: RealCyclotomic10) -> int:
  return (xi * xi.automorphism()).to_int()


# A8
def N_i(eta: Cyclotomic10) -> RealCyclotomic10:
  return (eta * eta.conjugate()).to_subring()


# A9
def N(eta: Cyclotomic10) -> int:
  return N_tau(N_i(eta))


# A10
def gauss_complexity_measure(eta: Cyclotomic10) -> int:
  return (N_i(eta) + N_i(eta).automorphism()).to_int()


def find_unit_inverse_mod_one_plus_omega(x: Cyclotomic10) -> Cyclotomic10:
  if x.divides_by_one_plus_omega():
    raise ValueError("Input divisible by (1 + ω)")

  xmod = x.mod_one_plus_omega()
  inv_mod = pow(xmod, -1, 5)

  # Precomputed units
  units = [
    Cyclotomic10.One(),
    Cyclotomic10.Omega(),
    Cyclotomic10.Omega() ** 2,
    Cyclotomic10.Omega() ** 3,
    Cyclotomic10.Omega() ** 4,
    Cyclotomic10.Tau(),
    Cyclotomic10.Tau() * Cyclotomic10.Omega(),
    Cyclotomic10.Tau() * Cyclotomic10.Omega() ** 2,
    Cyclotomic10.Tau() * Cyclotomic10.Omega() ** 3,
    Cyclotomic10.Tau() * Cyclotomic10.Omega() ** 4,
    # Add more units as needed
  ]

  for u in units:
    if u.mod_one_plus_omega() == inv_mod and u.is_unit():
      return u

  raise RuntimeError(f"No unit inverse found for residue {xmod}")


if __name__ == "__main__":
  x = RealCyclotomic10(2, -1)
  print(x * RealCyclotomic10(3, 1))
  print(x.div_by_two_minus_tau())

  y = Cyclotomic10(1, 1, 1, 1)
  aut_expected = 1 + Cyclotomic10.Omega_(3).evaluate() + Cyclotomic10.Omega_(6).evaluate() + Cyclotomic10.Omega_(9).evaluate()
  print(y.automorphism().evaluate())
  print(aut_expected)
  aut7_expected = 1 + Cyclotomic10.Omega_(7).evaluate() + Cyclotomic10.Omega_(4).evaluate() + Cyclotomic10.Omega_(1).evaluate()
  print(aut7_expected)
  print(y.galois_automorphism(7).evaluate())

  aut9_expected = 1 + Cyclotomic10.Omega_(9).evaluate() + Cyclotomic10.Omega_(8).evaluate() + Cyclotomic10.Omega_(7).evaluate()
  print(aut9_expected)
  print(y.automorphism().automorphism().evaluate())

  print((y * y.conjugate()).to_subring())

  yy = y.inv()
  print(y.norm_squared().evaluate())
  print(yy)
  print(y * yy)
  print(N(yy))
  print(yy.galois_norm())

  fact = Cyclotomic10(1, 1, 0, 0)
  fact_conj = Cyclotomic10(1, -1, 0, 0)
  print("N(1+w) =", N(fact))
  print("(1+w)⁻¹ =", fact.pseudo_inv())
  conjs = fact.galois_automorphism(3) * fact.galois_automorphism(7) * fact.galois_automorphism(9)
  print("conj * fact: ", fact * conjs)
  print("fact.divide_by_one_plus_omega: ", fact.div_by_one_plus_omega())
  print("(1-w).is_unit(): ", fact_conj.is_unit())
  print(Cyclotomic10.Omega().is_unit())
  print("-1//5: ", -1 // 5)
  test = Cyclotomic10(15, 0, -8, 8)
  print(test.mod_one_plus_omega())
  u = find_unit_inverse_mod_one_plus_omega(test)
  print(u)
  print((test * u).mod_one_plus_omega())
  print(find_unit_inverse_mod_one_plus_omega(-Cyclotomic10.One()))
  print(Cyclotomic10.Tau().is_unit())

  a = Cyclotomic10(7, 3, -2, 1)
  b = Cyclotomic10(2, -1, 1, 0)
  q, r = divmod(a, b)
  # Check: a == b*q + r
  recomposed = b * q + r
  assert recomposed == a, f"Failed: b*q + r = {recomposed}, expected {a}"
  # Norm of remainder should be less than norm of divisor (heuristic)
  norm_r = r.galois_norm()
  norm_b = b.galois_norm()
  assert norm_r < norm_b, f"Remainder norm {norm_r} not smaller than divisor norm {norm_b}"
  print(f"Test passed:\n a = {a}\n b = {b}\n q = {q}\n r = {r}")
  print(f"Norm of b: {norm_b}, Norm of r: {norm_r}")
