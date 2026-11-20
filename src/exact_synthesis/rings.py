from __future__ import annotations

import mpmath
from mpmath import mp
from dataclasses import dataclass


def _rounddiv(x: int, y: int) -> int:
  return (x + y // 2) // y if y > 0 else (x - (-y) // 2) // y


class ZOmega:
  r"""Ring of cyclotomic integers of order 10.

  Represents elements in :math:`\mathbb{Z}[\zeta_{10}]`, where
  :math:`\zeta_{10} = e^{2\pi i / 10}` is a primitive 10th root of unity.
  Any element of this ring has the unique representation

  .. math::

      c_0 + c_1 \zeta_{10} + c_2 \zeta_{10}^2 + c_3 \zeta_{10}^3 \quad (c_i \in \mathbb{Z})

    where the minimal polynomial of :math:`\zeta_{10}` is

  .. math::

      \Phi_{10}(x) = x^4 - x^3 + x^2 - x + 1
  """

  def __init__(self, c0: int, c1: int, c2: int, c3: int) -> None:
    self.c0: int = c0
    self.c1: int = c1
    self.c2: int = c2
    self.c3: int = c3

  # @property TODO: make dataclass frozen
  def coeffs(self) -> list[int]:
    return [self.c0, self.c1, self.c2, self.c3]

  def __mul__(self, other: int | ZTau | ZOmega) -> ZOmega:
    if isinstance(other, int):
      return self.__mul__(ZOmega.from_int(other))
    elif isinstance(other, ZTau):
      return self.__mul__(other.to_cycl())
    a, b = self.coeffs(), other.coeffs()
    p = [0] * 7
    for i, x in enumerate(a):
      for j, y in enumerate(b):
        p[i + j] += x * y
    p[1] -= p[6]
    p[0] -= p[5]
    p[3] += p[4]
    p[2] -= p[4]
    p[1] += p[4]
    p[0] -= p[4]
    return ZOmega(*p[:4])

  def __rmul__(self, other: int | ZTau | ZOmega):
    return self.__mul__(other)

  def __pow__(self, exp: int) -> ZOmega:
    if not isinstance(exp, int):
      return NotImplemented
    result = self.One()
    base, exp = (self, exp) if exp >= 0 else (self.inv(), -exp)

    while exp:
      if exp & 1:
        result *= base
      base *= base
      exp >>= 1
    return result

  @classmethod
  def from_int(cls, other: int):
    return cls(other, 0, 0, 0)

  @classmethod
  def One(cls) -> ZOmega:
    return cls.from_int(1)

  @classmethod
  def Zero(cls):
    return cls.from_int(0)

  @classmethod
  def Tau(cls):
    return ZOmega(0, 0, 1, -1)

  @classmethod
  def Phi(cls):
    return ZOmega(1, 0, 1, -1)

  @classmethod
  def Omega(cls):
    return ZOmega(0, 1, 0, 0)

  @classmethod
  def Omega_(cls, k: int):
    return cls.Omega() ** k

  def conjugate(self):
    a, b, c, d = self.coeffs()
    return ZOmega(a + b, -b, b - d, -b - c)

  def automorphism(self):
    c0, c1, c2, c3 = self.coeffs()
    return ZOmega(c0 + c3, -c2 - c3, c3, c1 - c3)

  def galois_automorphism(self, k: int) -> ZOmega:
    if k == 1:
      return self
    elif k == 3:
      return self.automorphism()
    elif k == 7:
      c0, c1, c2, c3 = self.coeffs()
      return ZOmega(c0 - c2, c3 + c2, -c1 - c2, c2)
    elif k == 9:
      return self.automorphism().automorphism()
    else:
      raise NotImplementedError(f"galois_automorphism(k : int) is only available for k = 1,3,7,9 but was {k}.")

  def inv(self) -> ZOmega:
    conjs = self.galois_automorphism(3) * self.galois_automorphism(7) * self.galois_automorphism(9)
    a, b, c, d = conjs.coeffs()
    N = self.galois_norm()
    return ZOmega(a // N, b // N, c // N, d // N)

  def pseudo_inv(self):
    conjs = self.galois_automorphism(3) * self.galois_automorphism(7) * self.galois_automorphism(9)
    return ZOmega(*conjs.coeffs())

  def galois_automorphism_product(self):
    return self.galois_automorphism(3) * self.galois_automorphism(7) * self.galois_automorphism(9)

  # TODO fix
  def __divmod__(self, other) -> tuple[ZOmega, ZOmega]:
    if isinstance(other, int):
      other = self.from_int(other)
    if isinstance(other, self.__class__):
      p = self * other.galois_automorphism_product()
      k = N(other)
      q_coeffs = [_rounddiv(c, k) for c in p.coeffs()]

      q = self.__class__(*q_coeffs)
      r = self - other * q
      return q, r
    else:
      raise NotImplementedError(f"__divmod__ is not implement for {type(self)} and {type(other)}.")

  def __floordiv__(self, other):
    q, _ = divmod(self, other)
    return q

  def __truediv__(self, other):
    raise NotImplementedError("Only __floordiv__ available as ZOmega only supports Integer coefficients.")

  def galois_norm(self) -> int:
    norm = self * self.galois_automorphism(3) * self.galois_automorphism(7) * self.galois_automorphism(9)
    return norm.to_subring().to_int()

  def norm_i(self):
    return self * self.conjugate()

  def integer_remainder_mod_one_plus_omega(self) -> int:
    a, b, c, d = self.coeffs()
    return a - b + c - d

  @classmethod
  def from_omega_4(cls, k: int) -> ZOmega:
    return ZOmega(-k, k, -k, k)

  def is_unit(self) -> bool:
    return self.galois_norm() == 1

  def mod_one_plus_omega(self) -> int:
    c0, c1, c2, c3 = self.coeffs()
    result = c0 - c1 + c2 - c3
    return ((result + 2) % 5) - 2  # modulo N(1 + ω)

  def divides_by_one_plus_omega(self) -> bool:
    return self.galois_norm() % 5 == 0 and self.mod_one_plus_omega() == 0

  def evaluate(self):
    omega = mpmath.exp(mpmath.mpc(0, mpmath.pi) / 5)
    return self.c0 + self.c1 * omega + self.c2 * omega**2 + self.c3 * omega**3

  def to_subring(self):
    a, b, c, d = self.coeffs()
    assert b == 0 and c == -d, f"Tau is not represented in {self}"
    return ZTau(a, c)

  def gauss_complexity(self):
    u1 = N_i(self)
    u2 = N_i(self.automorphism())
    return u1 + u2

  def __eq__(self, other):
    return self.coeffs() == other.coeffs()

  def __add__(self, other: ZOmega) -> ZOmega:
    if isinstance(other, ZOmega):
      return ZOmega(*(a + b for a, b in zip(self.coeffs(), other.coeffs())))
    elif isinstance(other, int):
      a, b, c, d = self.coeffs()
      return ZOmega(a + other, b, c, d)
    else:
      raise ValueError(f"__add__ for ZOmega does not support {other}.")

  def __neg__(self):
    return self.__class__(-self.c0, -self.c1, -self.c2, -self.c3)

  def __sub__(self, other: ZOmega) -> ZOmega:
    return self + (-other)

  def __str__(self):
    labels = ["", "ω", "ω²", "ω³"]
    return " + ".join(f"{c}{l}" for c, l in zip(self.coeffs(), labels) if c != 0) or "0"

  def __repr__(self):
    return f"Cyclotomic10{self.coeffs()}"

  def __hash__(self):
    return hash(tuple(self.coeffs()))


class ZTau:
  """
  Real subring of ZOmega := Z [(sqrt(5) - 1) / 2]
  """

  def __init__(self, a: int, b: int):
    self.a: int = a
    self.b: int = b

  def __mul__(self, other: int | ZTau) -> ZTau:
    if isinstance(other, int):
      other = ZTau.from_int(other)
    a, b = self.a, self.b
    c, d = other.a, other.b

    return ZTau(a * c + b * d, a * d + b * c - (b * d))

  def to_tuple(self) -> tuple[int, int]:
    return self.a, self.b

  def evaluate(self):
    tau = mp.mpf((mp.sqrt(5) - 1) / 2)
    return self.a + self.b * tau

  def automorphism(self):
    return ZTau(self.a - self.b, -self.b)

  def norm(self) -> int:
    x = self * self.automorphism()
    assert x.b == 0  # TODO: what is this?
    return x.a

  def conjugate(self):
    return self.automorphism().automorphism()

  @classmethod
  def from_int(cls, n: int):
    return ZTau(n, 0)

  def to_cycl(self) -> ZOmega:
    return ZOmega(self.a, 0, self.b, -self.b)

  def to_int(self) -> int:
    if self.b == 0:
      return self.a
    else:
      raise NotImplementedError

  def div_by_two_minus_tau(self):
    num = self * ZTau(2, -1).automorphism()  # multiply by (2+τ)

    if num.a % 5 != 0 or num.b % 5 != 0:
      raise ValueError(f"{self} is not divisible by (2-τ)")

    return ZTau(num.a // 5, num.b // 5)

  def gen_div(self, other):
    n = N_tau(other)
    num = self * other.automorphism()
    if num.a % n != 0 or num.b % n != 0:
      raise ValueError(f"{self} is not divisible by {other}")
    return ZTau(num.a // n, num.b // n)

  def __div__(self, other):
    return self.gen_div(other)

  def __rmul__(self, other):
    if isinstance(other, int):
      return ZTau(self.a * other, self.b * other)
    return self.__mul__(other)

  def __pow__(self, exponent: int) -> ZTau:
    if not isinstance(exponent, int):
      raise ValueError(f"exponent must be integer, was {type(exponent)}")
    base, exp = (self, exponent) if exponent >= 0 else (self.inv(), -exponent)
    result = ZTau(1, 0)

    while exp:
      if exp & 1:
        result *= base
      base *= base
      exp >>= 1

    return result

  def __neg__(self):
    return ZTau(-self.a, -self.b)

  def __add__(self, other):
    if isinstance(other, int):
      return ZTau(self.a + other, self.b)
    elif isinstance(other, ZTau):
      return ZTau(self.a + other.a, self.b + other.b)
    else:
      raise ValueError(f"Can only add ZTau with ZTau but was {type(other)}")

  def __sub__(self, other) -> ZTau:
    return self.__add__(-other)

  def __eq__(self, other):
    if isinstance(other, int):
      return self.a == other and self.b == 0
    elif isinstance(other, ZTau):
      return self.a == other.a and self.b == other.b
    else:
      raise NotImplementedError(f"Types not matching {other.__class__}")

  def inv(self):
    # TODO: fix with normal arithmetic
    inv = self.to_cycl().inv()
    return inv.to_subring()

  @classmethod
  def One(cls):
    return ZTau(1, 0)

  @classmethod
  def Tau(cls):
    return ZTau(0, 1)

  @classmethod
  def Phi(cls) -> ZTau:
    return ZTau(1, 1)

  def __repr__(self) -> str:
    return f"ZTau({self.a}, {self.b})"

  def __str__(self) -> str:
    if self.b >= 0:
      if self.b == 1:
        return f"{self.a} + τ"
      return f"{self.a} + {self.b}τ"
    else:
      if self.b == -1:
        return f"{self.a} - τ"
      return f"{self.a} - {-self.b}τ"

  def __divmod__(self, other: ZTau | int) -> tuple[ZTau, ZTau]:
    if isinstance(other, int):
      other = ZTau.from_int(other)
    if not isinstance(other, ZTau):
      return NotImplemented
    k = N_tau(other)
    if k == 0:
      raise ZeroDivisionError("division by zero in ZTau")
    p = self * other.automorphism()
    q = ZTau(_rounddiv(p.a, k), _rounddiv(p.b, k))
    return q, self - other * q

  def __floordiv__(self, other: ZTau) -> ZTau:
    q, _ = divmod(self, other)
    return q

  def __mod__(self, other: ZTau) -> ZTau:
    _, r = self.__divmod__(other)
    return r


# A7
def N_tau(xi: ZTau) -> int:
  return (xi * xi.automorphism()).to_int()


# A8
def N_i(eta: ZOmega) -> ZTau:
  return (eta * eta.conjugate()).to_subring()


# A9
def N(eta: ZOmega) -> int:
  return N_tau(N_i(eta))


# A10
def gauss_complexity_measure(eta: ZOmega) -> int:
  """
  N_i(eta) + N_i(eta^aut) evaluates to a ZTau, but since this equation is idempotent over autimorphisms, we can just cast to int.
  """
  return (N_i(eta) + N_i(eta).automorphism()).to_int()


SQRT_TAU_MINUS_2 = ZOmega(-1, 2, -1, 1)


if __name__ == "__main__":
  import random

  # Move to test
  for _ in range(10_000):
    a = ZTau(random.randint(-50, 50), random.randint(-50, 50))
    b = ZTau(random.randint(-50, 50), random.randint(-50, 50))
    if N_tau(b) == 0:
      continue
    q, r = divmod(a, b)
    assert a == b * q + r
    assert abs(N_tau(r)) < abs(N_tau(b))
