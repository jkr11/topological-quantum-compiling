import mpmath
from mpmath import mp
from typing import List, Tuple


class Cyclotomic10:
  def __init__(self, c0: int, c1: int, c2: int, c3: int) -> None:
    self.c0: int = c0
    self.c1: int = c1
    self.c2: int = c2
    self.c3: int = c3

  def coeffs(self) -> List[int]:
    return [self.c0, self.c1, self.c2, self.c3]

  def __mul__(self, other) -> "Cyclotomic10":
    if isinstance(other, int):
      return self.__mul__(Cyclotomic10.from_int(other))
    elif isinstance(other, ZTau):
      return self.__mul__(other.to_cycl())
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

  def __pow__(self, exponent: int):
    if not isinstance(exponent, int):
      return NotImplemented
    if exponent == 0:
      return self.One()
    if exponent > 0:
      result = self.One()
      for _ in range(exponent):
        result = result * self
      return result
    else:
      inv = self.inv()
      result = self.One()
      for _ in range(-exponent):
        result = result * inv
      return result

  @classmethod
  def from_int(cls, other: int):
    return cls(other, 0, 0, 0)

  @classmethod
  def One(cls):
    return cls.from_int(1)

  @classmethod
  def Zero(cls):
    return cls.from_int(0)

  @classmethod
  def Tau(cls):
    return Cyclotomic10(0, 0, 1, -1)

  @classmethod
  def Omega(cls):
    return Cyclotomic10(0, 1, 0, 0)

  @classmethod
  def Omega_(cls, k: int):
    return cls.Omega() ** k

  def conjugate(self):
    c0, c1, c2, c3 = self.coeffs()
    return Cyclotomic10(c0 + c1, -c1, c1 - c3, -c1 - c2)

  def automorphism(self):
    c0, c1, c2, c3 = self.coeffs()
    return Cyclotomic10(c0 + c3, -c2 - c3, c3, c1 - c3)

  def galois_automorphism(self, k: int) -> "Cyclotomic10":
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
      raise NotImplementedError(f"galois_automorphism(k : int) is only available for k = 1,3,7,9 but was {k}.")

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

  def __divmod__(self, other) -> tuple["Cyclotomic10", "Cyclotomic10"]:
    def rounddiv(x: int, y: int):
      return (x + y // 2) // y if y > 0 else (x - (-y) // 2) // y

    if isinstance(other, int):
      other = self.from_int(other)
    elif isinstance(other, self.__class__):
      p = self * other.galois_automorphism_product()

      k = N(other)

      q_coeffs = [rounddiv(c, k) for c in p.coeffs()]
      q = self.__class__(*q_coeffs)

      r = self - other * q
      return q, r
    else:
      raise NotImplementedError(f"__divmod__ is not implement for {type(self)} and {type(other)}.")

  def __floordiv__(self, other):
    q, _ = divmod(self, other)
    return q

  def __div__(self, other):
    raise NotImplementedError("Only __floordiv__ available as we implement over Integers")

  def galois_norm(self) -> int:
    norm = self * self.galois_automorphism(3) * self.galois_automorphism(7) * self.galois_automorphism(9)
    return norm.to_subring().to_int()

  def norm_i(self):
    return self * self.conjugate()

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
    omega = mpmath.exp(2 * mpmath.pi * 1j / 10)
    return self.c0 + self.c1 * omega + self.c2 * omega**2 + self.c3 * omega**3

  def to_subring(self):
    a, b, c, d = self.coeffs()
    if b == 0 and c == -d:
      return ZTau(a, c)
    else:
      raise ValueError(f"Tau is not represented in {self}")

  def gauss_complexity(self):
    u1 = N_i(self)
    u2 = N_i(self.automorphism())
    return u1 + u2

  def __eq__(self, other):
    return self.coeffs() == other.coeffs()

  def __add__(self, other):
    if isinstance(other, Cyclotomic10):
      return Cyclotomic10(
        self.coeffs()[0] + other.coeffs()[0],
        self.coeffs()[1] + other.coeffs()[1],
        self.coeffs()[2] + other.coeffs()[2],
        self.coeffs()[3] + other.coeffs()[3],
      )
    elif isinstance(other, int):
      a, b, c, d = self.coeffs()
      return Cyclotomic10(a + other, b, c, d)

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

  def __hash__(self):
    return hash(tuple(self.coeffs()))

  # def __divmod__(self, other) -> Tuple:
  #   def rounddiv(x : int, y : int) -> int:
  #     return (x + y // 2) // y if y > 0 else (x - (-y) // 2) // y
  #   if not isinstance(other, Cyclotomic10):
  #     raise TypeError(f"Unsupported operand type for divmod {type(other)}")
  #   p = self * other
  #   k = N(other)
  #   q_coeffs = [rounddiv(c, k) for c in p.coeffs()]
  #   q = Cyclotomic10(*q_coeffs)
  #   r = self - (other * q)
  #   return q, r


class ZTau:
  def __init__(self, a: int, b: int):
    self.a: int = a
    self.b: int = b

  def __mul__(self, other):
    if not isinstance(other, ZTau):
      other = ZTau.from_int(other)
    # (a + bτ)(c + dτ) = (ac + bd) + (ad + bc - bd)τ
    a, b = self.a, self.b
    c, d = other.a, other.b

    real_part = a * c + b * d
    tau_part = a * d + b * c - (b * d)

    return ZTau(real_part, tau_part)

  def evaluate(self):
    tau = mp.mpf((mp.sqrt(5) - 1) / 2)
    return self.a + self.b * tau

  def automorphism(self):
    """applies the automorphism w -> w^3 to 'self', on tau = w^2 - w^3. aut(tau) = -phi, where phi = 1 + tau. Thus aut(a + btau) = a - (1 + tau)b"""
    return ZTau(self.a - self.b, -self.b)

  def norm(self) -> int:
    return self * self.automorphism()

  def conjugate(self):
    return self.automorphism().automorphism()

  @classmethod
  def from_int(self, n: int):
    return ZTau(n, 0)

  def to_cycl(self) -> Cyclotomic10:
    return Cyclotomic10(self.a, 0, self.b, -self.b)

  def to_int(self) -> int:
    if self.b == 0:
      return self.a
    else:
      raise NotImplementedError

  def div_by_two_minus_tau(self):
    """Divide by (2-τ) using the fact that (2-τ)(3+τ) = 5"""
    # First multiply by (3+τ)
    num = self * ZTau(2, -1).automorphism()  # multiply by (2+τ)

    # Check if result is divisible by 5
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
    """Right multiplication - allows integer * RealCyclotomic10"""
    if isinstance(other, int):
      return ZTau(self.a * other, self.b * other)
    return self.__mul__(other)

  def __pow__(self, exponent: int):
    if not isinstance(exponent, int):
      return ValueError(f"exponent must be integer, was {type(exponent)}")

    if exponent == 0:
      return ZTau(1, 0)

    if exponent < 0:
      raise ValueError(f"exponent must be positive integer, was {exponent}")

    result = ZTau(1, 0)
    for _ in range(exponent):
      result = result * self

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

  def __sub__(self, other):
    return self.__add__(-other)

  def __eq__(self, other):
    if isinstance(other, int):
      return self == ZTau.from_int(other)
    return self.a == other.a and self.b == other.b

  def inv(self):
    n = N_tau(self)
    ns = self.conjugate()
    if ns.a % n != 0 or ns.b % n != 0:
      raise ValueError(f"{self} is not invertible in Z[τ]")
    return ZTau(ns.a // n, ns.b // n)

  @classmethod
  def One(self):
    return ZTau(1, 0)

  @classmethod
  def Tau(self):
    return ZTau(0, 1)

  @classmethod
  def Phi(self) -> "ZTau":
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

  def __divmod__(self, other: "ZTau") -> Tuple["ZTau", "ZTau"]:
    if not isinstance(other, ZTau):
      raise TypeError(f"divmod does not support {type(self)} and {type(other)}")

    def rounddiv(x: int, y: int) -> int:
      return (x + y // 2) // y if y > 0 else (x - (-y) // 2) // y

    if isinstance(other, int):
      other = ZTau(other, 0)

    a_cyclo = self.to_cycl()
    b_cyclo = other.to_cycl()

    p = a_cyclo * b_cyclo.galois_automorphism_product()
    k = N(other.to_cycl())  # TODO: check why this is different when casting

    q_coeffs = [rounddiv(c, k) for c in p.coeffs()]
    q_cyclo = Cyclotomic10(*q_coeffs)

    q = q_cyclo.to_subring()

    r = self - other * q
    return q, r

  def __floordiv__(self, other):
    q, _ = divmod(self, other)
    return q

  def __mod__(self, other):
    _, r = divmod(self, other)
    return r


# A7
def N_tau(xi: ZTau) -> int:
  return (xi * xi.automorphism()).to_int()


# A8
def N_i(eta: Cyclotomic10) -> ZTau:
  return (eta * eta.conjugate()).to_subring()


# A9
def N(eta: Cyclotomic10) -> int:
  return N_tau(N_i(eta))


# A10
def gauss_complexity_measure(eta: Cyclotomic10) -> int:
  """
  N_i(eta) + N_i(eta^aut) evaluates to a ZTau, but since this equation is idempotent over autimorphisms, we can just cast to int.
  """
  return (N_i(eta) + N_i(eta).automorphism()).to_int()
