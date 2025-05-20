import math
import numpy as np
import cmath
from functools import cached_property, total_ordering
from fractions import Fraction


#@total_ordering
class Cyclotomic10:

  def __init__(self, c0, c1, c2, c3):
    self.c0 : int = c0
    self.c1 : int = c1
    self.c2 : int = c2
    self.c3 : int = c3

  def coeffs(self):
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
          return Cyclotomic10(-1, 0, 0, 0)
        elif exponent & 10 == 0:
          return Cyclotomic10.One()
      result = Cyclotomic10(1, 0, 0, 0)  # Identity element (1)
      for _ in range(exponent):
        result = result * self
      return result
    else:
      return NotImplemented

  @classmethod
  def from_int(self, other: int):
    return Cyclotomic10(other, 0, 0, 0)

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
    return self.Omega()**k

  #@cached_property
  def conjugate(self):
    c0, c1, c2, c3 = self.coeffs()
    return Cyclotomic10(c0 + c1, -c1, c1 - c3, -c1 - c2)

  #@cached_property
  def automorphism(self):
    c0, c1, c2, c3 = self.coeffs()
    return Cyclotomic10(c0 + c3, -c2 - c3, c3, c1 - c3)

  def norm_squared(self):
    product = self * self.conjugate()
    c0, c1, c2, c3 = product.coeffs()
    # return RealCyclotomic10(c0, c2)
    return product

  def mod_one_plus_omega(self) -> int:
    """
    Compute eta mod (1 + ω) by substituting ω = -1.
    The result is an integer (ℤ), in the range {0, ±1, ±2}.
    """
    c0, c1, c2, c3 = self.coeffs()
    result = c0 - c1 + c2 - c3
    return result % 5  # modulo N(1 + ω)

  def divides_by_one_plus_omega(self) -> bool:
    return self.mod_one_plus_omega(self) == 0

  def evaluate(self):
    theta = math.pi / 5  # ω = e^(πi/5)
    omega = complex(math.cos(theta), math.sin(theta))
    total = 0j
    for i in range(4):
      total += self.coeffs()[i] * (omega**i)
    return total

  def gauss_complexity(self):
    u1 = self.norm_squared()
    u2 = self.automorphism().norm_squared()
    return u1 + u2

  def __eq__(self, other):
    return self.coeffs() == other.coeffs()

  def __add__(self, other):
    return Cyclotomic10(
        self.coeffs()[0] + other.coeffs()[0],
        self.coeffs()[1] + other.coeffs()[1],
        self.coeffs()[2] + other.coeffs()[2],
        self.coeffs()[3] + other.coeffs()[3],
    )

  def __sub__(self, other):
    return Cyclotomic10(
        self.coeffs()[0] - other.coeffs()[0],
        self.coeffs()[1] - other.coeffs()[1],
        self.coeffs()[2] - other.coeffs()[2],
        self.coeffs()[3] - other.coeffs()[3],
    )

  def __str__(self):
    labels = ["", "ω", "ω²", "ω³"]
    return " + ".join(f"{c}{l}"
                      for c, l in zip(self.coeffs(), labels) if c != 0) or "0"

  def __repr__(self):
    return f"Cyclotomic10{self.coeffs()}"


TAU = (math.sqrt(5) - 1) / 2
PHI = TAU + 1


class RealCyclotomic10:

  def __init__(self, a: int, b: int):
    self.a = a
    self.b = b

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
    return (self * self.automorphism())

  def conjugate(self):
    return self.automorphism().automorphism()

  def to_cycl(self) -> Cyclotomic10:
    return Cyclotomic10(self.a, 0, self.b, -self.b)

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
    return self.a == other.a and self.b == other.b

  def __repr__(self):
    if self.b >= 0:
      if self.b == 1:
        return f"{self.a} + τ"
      return f"{self.a} + {self.b}τ"
    else:
      if self.b == -1:
        return f"{self.a} - τ"
      return f"{self.a} - {- self.b}τ"


if __name__ == "__main__":
  x = RealCyclotomic10(2, -1)
  print(x * RealCyclotomic10(3, 1))
  print(x.div_by_two_minus_tau())
  