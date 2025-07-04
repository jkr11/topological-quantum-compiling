import mpmath

mpmath.mp.dps = 100  # Set default precision; adjust as needed


class RealCyclotomic10MP:
  """
  Represents elements of Z[τ] with arbitrary precision using mpmath.
  τ = (sqrt(5) - 1)/2
  """

  TAU = (mpmath.sqrt(5) - 1) / 2

  def __init__(self, a, b):
    self.a = mpmath.mpf(a)
    self.b = mpmath.mpf(b)

  def __add__(self, other):
    if isinstance(other, (int, float, mpmath.mpf)):
      return RealCyclotomic10MP(self.a + other, self.b)
    return RealCyclotomic10MP(self.a + other.a, self.b + other.b)

  def __sub__(self, other):
    if isinstance(other, (int, float, mpmath.mpf)):
      return RealCyclotomic10MP(self.a - other, self.b)
    return RealCyclotomic10MP(self.a - other.a, self.b - other.b)

  def __mul__(self, other):
    if isinstance(other, (int, float, mpmath.mpf)):
      return RealCyclotomic10MP(self.a * other, self.b * other)
    a, b = self.a, self.b
    c, d = other.a, other.b
    # (a + bτ)(c + dτ) = ac + (ad + bc + bdτ)τ
    real = a * c + b * d * self.TAU
    tau = a * d + b * c + b * d * self.TAU
    return RealCyclotomic10MP(real, tau)

  def __pow__(self, exponent: int):
    if exponent == 0:
      return RealCyclotomic10MP(1, 0)
    result = RealCyclotomic10MP(1, 0)
    if exponent > 0:
      for _ in range(exponent):
        result = result * self
    else:
      inv = self.inv()
      for _ in range(-exponent):
        result = result * inv
    return result

  def __neg__(self):
    return RealCyclotomic10MP(-self.a, -self.b)

  def __eq__(self, other):
    return mpmath.almosteq(self.a, other.a) and mpmath.almosteq(self.b, other.b)

  def __repr__(self):
    return f"{self.a} + {self.b}τ"

  def automorphism(self):
    # Galois automorphism: τ ↦ -1 - τ
    return RealCyclotomic10MP(self.a - self.b, -self.b)

  def norm(self):
    # N_τ(a + bτ) = a^2 + ab - b^2
    return self.a**2 + self.a * self.b - self.b**2

  def inv(self):
    N = self.norm()
    if N == 0:
      raise ZeroDivisionError("Cannot invert zero in RealCyclotomic10MP")
    conj = self.automorphism()
    return RealCyclotomic10MP(conj.a / N, conj.b / N)

  def evaluate(self):
    return self.a + self.b * self.TAU

  @classmethod
  def from_int(cls, n):
    return cls(n, 0)

  @classmethod
  def One(cls):
    return cls(1, 0)

  @classmethod
  def Zero(cls):
    return cls(0, 0)


class Cyclotomic10MP:
  """
  Represents elements of Z[ω] with arbitrary precision using mpmath.
  ω = exp(2πi/10)
  """

  def __init__(self, c0, c1, c2, c3):
    self.c0 = mpmath.mpf(c0)
    self.c1 = mpmath.mpf(c1)
    self.c2 = mpmath.mpf(c2)
    self.c3 = mpmath.mpf(c3)

  def coeffs(self):
    return [self.c0, self.c1, self.c2, self.c3]

  def __add__(self, other):
    return Cyclotomic10MP(self.c0 + other.c0, self.c1 + other.c1, self.c2 + other.c2, self.c3 + other.c3)

  def __sub__(self, other):
    return Cyclotomic10MP(self.c0 - other.c0, self.c1 - other.c1, self.c2 - other.c2, self.c3 - other.c3)

  def __neg__(self):
    return Cyclotomic10MP(-self.c0, -self.c1, -self.c2, -self.c3)

  def __eq__(self, other):
    return all(mpmath.almosteq(x, y) for x, y in zip(self.coeffs(), other.coeffs()))

  def __repr__(self):
    return f"{self.c0} + {self.c1}ω + {self.c2}ω² + {self.c3}ω³"

  def __mul__(self, other):
    if isinstance(other, int):
      return self.__mul__(Cyclotomic10MP.from_int(other))
    a = self.coeffs()
    b = other.coeffs() if isinstance(other, Cyclotomic10MP) else other
    exponents_coeffs = [
      (1, 0, 0, 0),  # ζ^0
      (0, 1, 0, 0),  # ζ^1
      (0, 0, 1, 0),  # ζ^2
      (0, 0, 0, 1),  # ζ^3
      (-1, 1, -1, 1),  # ζ^4
      (-1, 0, 0, 0),  # ζ^5
      (0, -1, 0, 0),  # ζ^6
    ]
    result = [mpmath.mpf(0)] * 4
    for i in range(4):
      for j in range(4):
        k = i + j
        coeffs = exponents_coeffs[k] if k < 7 else (0, 0, 0, 0)
        term = a[i] * b[j]
        result[0] += term * coeffs[0]
        result[1] += term * coeffs[1]
        result[2] += term * coeffs[2]
        result[3] += term * coeffs[3]
    return Cyclotomic10MP(*result)

  def __pow__(self, exponent: int):
    if exponent == 0:
      return Cyclotomic10MP(1, 0, 0, 0)
    result = Cyclotomic10MP(1, 0, 0, 0)
    if exponent > 0:
      for _ in range(exponent):
        result = result * self
    else:
      inv = self.inv()
      for _ in range(-exponent):
        result = result * inv
    return result

  def inv(self):
    # Inverse in Q(ω): Use norm and Galois conjugate
    N = self.norm()
    if N == 0:
      raise ZeroDivisionError("Cannot invert zero in Cyclotomic10MP")
    conj = self.galois_conjugate()
    return Cyclotomic10MP(conj.c0 / N, conj.c1 / N, conj.c2 / N, conj.c3 / N)

  def galois_conjugate(self):
    # Placeholder: implement the correct Galois conjugate for ω
    # For ω = exp(2πi/10), the complex conjugate is ω̄ = exp(-2πi/10)
    # For now, just return self (not correct for general use)
    return Cyclotomic10MP(self.c0, -self.c1, self.c2, -self.c3)

  def norm(self):
    # Placeholder: implement the correct norm for Z[ω]
    # For now, just use sum of squares (not correct for general use)
    return sum(x * x for x in self.coeffs())

  def evaluate(self):
    # Evaluate as a complex number
    omega = mpmath.exp(2 * mpmath.pi * 1j / 10)
    return self.c0 + self.c1 * omega + self.c2 * omega**2 + self.c3 * omega**3

  @classmethod
  def from_int(cls, n):
    return cls(n, 0, 0, 0)

  @classmethod
  def One(cls):
    return cls(1, 0, 0, 0)

  @classmethod
  def Zero(cls):
    return cls(0, 0, 0, 0)

  def conjugate(self):
    c0, c1, c2, c3 = self.coeffs()
    return Cyclotomic10MP(c0 + c1, -c1, c1 - c3, -c1 - c2)

  def automorphism(self):
    c0, c1, c2, c3 = self.coeffs()
    return Cyclotomic10MP(c0 + c3, -c2 - c3, c3, c1 - c3)

  def galois_automorphism(self, k: int):
    if k == 1:
      return self
    elif k == 3:
      return self.automorphism()
    elif k == 7:
      c0, c1, c2, c3 = self.coeffs()
      return Cyclotomic10MP(c0, c3, -c1, mpmath.mpf(0)) + Cyclotomic10MP.from_omega_4(c2)
    elif k == 9:
      return self.automorphism().automorphism()
    else:
      raise NotImplementedError

  def norm_squared(self):
    product = self * self.conjugate()
    return product

  def pseudo_inv(self):
    conjs = self.galois_automorphism(3) * self.galois_automorphism(7) * self.galois_automorphism(9)
    return Cyclotomic10MP(*conjs.coeffs())

  def galois_automorphism_product(self):
    return self.galois_automorphism(3) * self.galois_automorphism(7) * self.galois_automorphism(9)

  def __divmod__(self, other):
    def rounddiv(x, y):
      return mpmath.nint(x / y)

    if isinstance(other, int):
      other = Cyclotomic10MP.from_int(other)
    elif isinstance(other, Cyclotomic10MP):
      p = self * other.galois_automorphism_product()
      k = other.galois_norm()
      q_coeffs = [rounddiv(c, k) for c in p.coeffs()]
      q = Cyclotomic10MP(*q_coeffs)
      r = self - other * q
      return q, r
    else:
      return NotImplemented

  def __floordiv__(self, other):
    q, _ = divmod(self, other)
    return q

  def galois_norm(self):
    norm = self * self.galois_automorphism(3) * self.galois_automorphism(7) * self.galois_automorphism(9)
    # For the norm, you may want to convert to a RealCyclotomic10MP and then to int if needed
    return norm.coeffs()[0]  # or implement to_subring if needed

  def integer_remainder_mod_one_plus_omega(self):
    a, b, c, d = self.coeffs()
    return a - b + c - d

  @classmethod
  def from_omega_4(cls, k):
    return cls(-k, k, -k, k)

  def is_unit(self):
    return self.galois_norm() == 1

  def mod_one_plus_omega(self):
    c0, c1, c2, c3 = self.coeffs()
    result = c0 - c1 + c2 - c3
    return int(result) % 5

  def divides_by_one_plus_omega(self):
    return self.galois_norm() % 5 == 0 and self.mod_one_plus_omega() == 0

  def to_subring(self):
    a, b, c, d = self.coeffs()
    if b == 0 and c == -d:
      return RealCyclotomic10MP(a, c)
    else:
      raise ValueError(f"Tau is not represented in {self}")

  def __str__(self):
    labels = ["", "ω", "ω²", "ω³"]
    return " + ".join(f"{c}{l}" for c, l in zip(self.coeffs(), labels) if c != 0) or "0"

    # Add any other methods from Cyclotomic10 as needed, using mpmath types.
