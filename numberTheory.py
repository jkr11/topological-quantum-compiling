import math
import random
from rings import Cyclotomic10, RealCyclotomic10, N, find_unit_inverse_mod_one_plus_omega, N_tau, N_i
from typing import List, Tuple, Union
from itertools import product
from util import CONSTANTS


def fibonacci(n: int) -> int:
  assert n >= 0
  a, b = 0, 1
  for _ in range(n):
    a, b = b, a + b
  return a


def extended_fib_coefficients(n: int) -> Tuple[int, int]:
  assert n >= 0
  Fn = fibonacci(n)
  Fn_minus_1 = fibonacci(n - 1)
  u = ((-1) ** (n + 1)) * Fn
  v = (-1) ** n * Fn_minus_1
  return u, v


# APPROX-REAL procedure
def APPROX_REAL(x: float, n: int) -> RealCyclotomic10:
  print(f"APPROX_REAL with x={x}, n={n}")
  PREC = CONSTANTS.TAU ** (n - 1) * (1 - CONSTANTS.TAU**n)

  p = fibonacci(n)
  q = fibonacci(n + 1)
  u = (-1) ** (n + 1) * p
  v = (-1) ** n * fibonacci(n - 1)

  assert u * p + v * q == 1

  c = round(x * q)
  a = c * v + p * round((c * u) / q)
  b = c * u - q * round((c * u) / q)

  approx = a + b * CONSTANTS.TAU
  abs_diff = abs(x - approx)
  abs_b = abs(b)
  phi_pow_n = CONSTANTS.PHI**n
  assert (abs_diff <= PREC) and (abs_b <= phi_pow_n), f"Failed: abs(x - approx)={abs_diff} (PREC={PREC}), abs(b)={abs_b} (PHI^n={phi_pow_n})"
  return RealCyclotomic10(a, b)


# RANDOM-SAMPLE procedure
def RANDOM_SAMPLE(theta: float, epsilon: float, r: float) -> Cyclotomic10:
  assert r >= 1, f"r needs to be >= 1 but was {r}."
  PHI = CONSTANTS.PHI
  TAU = CONSTANTS.TAU
  # print(f"RANDOM_SAMPLE with {theta}, {epsilon}, {r}")
  C: float = math.sqrt(CONSTANTS.PHI / (4 * r))
  # print(f"C: {C}")
  m: int = math.ceil(math.log(C * epsilon * r, CONSTANTS.TAU)) + 1
  # print(f"math.log(C * epsilon * r), {math.log(C * epsilon * r, TAU)}")
  # print(f"m : {m}")
  N: int = math.ceil(CONSTANTS.PHI**m)

  sin_theta: float = math.sin(theta)
  cos_theta: float = math.cos(theta)

  sqrt_expr: float = math.sqrt(4 - epsilon**2)
  ymin: float = r * CONSTANTS.PHI**m * (sin_theta - epsilon * (sqrt_expr * cos_theta + epsilon * sin_theta) / 2)
  ymax: float = r * PHI**m * (sin_theta + epsilon * (sqrt_expr * cos_theta - epsilon * sin_theta) / 2)

  xmax: float = r * PHI**m * ((1 - epsilon**2 / 2) * cos_theta - epsilon * math.sqrt(1 - epsilon**2 / 4) * sin_theta)
  xc: float = xmax - (r * epsilon**2 * PHI**m) / (4 * cos_theta)

  # Random sampling
  j: int = random.randint(1, N - 1)
  y: float = ymin + j * (ymax - ymin) / N

  # Step 11: Approximate y'
  y_prime: float = y / math.sqrt(2 - TAU)
  yy = APPROX_REAL(y_prime, m)

  # Step 12: x calculation
  y_approx = (yy.evaluate()) * math.sqrt(2 - TAU)
  x = xc - (y_approx - ymin) * math.tan(theta)

  # Step 13: Approximate x
  xx: RealCyclotomic10 = APPROX_REAL(x, m)
  # ax, bx = xx.a, xx.b
  # Step 14: Final return
  part1 = xx
  print(f"Part 1 : {part1}")
  part2 = yy * RealCyclotomic10(2, -1)

  return (part1 + part2).to_cycl()


def is_square(n: int) -> Union[int, None]:
  if n < 0:
    return None
  root = math.isqrt(n)
  if root * root == n:
    return root
  return None


def IS_PRIME(p: int) -> bool:
  if p < 2:
    return False
  for i in range(2, int(p**0.5) + 1):
    if p % i == 0:
      return False
  return True


def EASY_SOLVABLE(fl: List[Tuple[RealCyclotomic10, int]]) -> bool:
  for i in range(0, len(fl)):
    # print("Fl: ", fl[i])
    xi, k = fl[i]
    if k % 2 == 1:
      if xi != RealCyclotomic10(5, 0):
        # print("Xi: ", xi)
        p: int = N_tau(xi)
        # print("P: ", p)
        r = p % 5
        # print("R: ", r)
        # print("Is prime: ", IS_PRIME(p))
        # print("R not in [0, 1]: ", r not in [0, 1])
        if not IS_PRIME(p) or r not in [0, 1]:
          # print("Not solvable")
          return False
  return True


# TODO: fix
def EASY_FACTOR(xi: RealCyclotomic10) -> List[Tuple[RealCyclotomic10, int]]:
  # print("EASY_FACTOR: ", xi)
  if isinstance(xi, int):
    xi = RealCyclotomic10(xi, 0)
  a, b = xi.a, xi.b
  c = math.gcd(a, b)
  a1 = a // c
  b1 = b // c
  xi1 = RealCyclotomic10(a1, b1)
  ret = []
  d: int = is_square(c)
  if d is not None:
    ret = [(d, 2)]
  else:
    d = is_square(c // 5) if c % 5 == 0 else None
    if d is not None:
      ret = [(d, 2), (5, 1)]
    else:
      print("Equations is not going to be solvable")
      return [(xi1, 1)]
  n = N_tau(xi1)
  # print("Xi1: ", xi1)
  if n % 5 == 0:
    xi2 = xi1.div_by_two_minus_tau()
    # print("Xi2: ", xi2)
    ret.append((RealCyclotomic10(2, -1), 1))
    ret.append((xi2, 1))  # this is not in the description, but it is in the example
    return ret
  else:
    # print("Xi1: ", xi1)
    ret.append((xi1, 1))
    return ret


def UNIT_DLOG(u: RealCyclotomic10) -> Tuple[int, int]:
  """
  Finds discrete logarithm of a unit u in Z[τ]
  Returns (s,k) such that u = s * τ^k where s = ±1
  """

  a, b = u.a, u.b
  s, k = 1, 0

  if a < 0:
    a, b = -a, -b
    s = -s
  if b == 0 and a == 1:
    return (1, 0)
  elif b == 0 and a == -1:
    return (-1, 0)
  elif a == 0 and b == 1:
    return (s, 1)
  elif a == 0 and b == -1:
    return (-s, 1)
  mu = a * b
  while abs(mu) > 1:
    if mu > 1:
      a, b = b, a - b
      k = k - 1
    else:
      a, b = a, a - b
      k = k + 1
    mu = a * b
    print(f"Current state: a={a}, b={b}, mu={mu}, k={k}")
  assert abs(mu) == 1, "Failed to reduce unit to base units"

  # Complete set of base units in Z[τ]
  print("a, b: ", a, b)
  match (a, b):
    case (1, 0):  # 1
      pass
    case (-1, 0):  # -1
      s *= -1
    case (0, 1):  # τ
      k += 1
    case (0, -1):  # -τ
      s *= -1
      k += 1
    case (1, 1):  # τ + 1 = τ^-1
      k -= 1
    case (-1, -1):  # −(τ + 1) = −τ^-1
      s *= -1
      k -= 1
    case (1, -1):  # τ - 1 = τ^2
      k += 2
    case (-1, 1):  # −(τ - 1) = −τ^(2)
      s *= -1
      k += 2
    # Other edge cases can be added here
    case _:
      pass  # Already in correct form
  return (s, k)


def legendre_symbol(a: int, p: int) -> int:
  """Compute the Legendre symbol (a | p)."""
  return pow(a, (p - 1) // 2, p)


def tonelli_shanks(n: int, p: int) -> Tuple[int, int]:
  """
  Solves for r such that r^2 = n (mod p), where p is an odd prime and n is a quadratic residue mod p.

  Returns a tuple (r, p - r).
  """
  assert p > 2 and legendre_symbol(n, p) == 1, "n must be a quadratic residue"

  # represent p-1 as q * 2 ^s with q odd
  q: int = p - 1
  s: int = 0
  while q % 2 == 0:
    q //= 2
    s += 1
  print(s)
  if s == 1:
    r: int = pow(n, ((p + 1) // 4), p)
    return (r, p - r)

  # By randomized trial, find a quadratic residue z.
  z: int = 2
  while legendre_symbol(z, p) != p - 1:
    z += 1

  c: int = pow(z, q, p)
  r: int = pow(n, (q + 1) // 2, p)
  t: int = pow(n, q, p)
  m: int = s
  while t != 1:
    i: int = 1
    temp: int = pow(t, 2, p)
    while temp != 1 and i < m:
      temp = pow(temp, 2, p)
      i += 1
    if i == m:
      raise ValueError("Failed to find i such that t^(2^i) ≡ 1 mod p")

    exponent: int = 2 ** (m - i - 1)
    b: int = pow(c, exponent, p)
    r = (r * b) % p
    t = (t * pow(b, 2, p)) % p
    c = pow(b, 2, p)
    m = i
  return (r, p - r)


# TODO: not sure if this works
def splitting_root(xi: RealCyclotomic10):
  def tau_norm_(s: RealCyclotomic10):
    a, b = s.a, s.b
    return a**2 - a * b - b**2

  def _mod_inv(b: int, p: int) -> int:
    return pow(b, -1, p)

  p: int = N_tau(xi)
  assert p % 2 == 1 and p % 5 == 1, "p = N_tau (xi) must be an odd prime = 1 "
  assert xi.b % p != 0
  b1 = _mod_inv(xi.b, p)
  n: int = (-xi.a * b1 - 2) % p
  return tonelli_shanks(n, p)


def _brute_force_unit() -> List[Cyclotomic10]:
  """
  Generates all units in Z[omega] of norm 1 and -1 using the Galois norm.
  """
  units = []
  for a, b, c, d in product(range(-10, 11), repeat=4):
    cyclo = Cyclotomic10(a, b, c, d)
    norm = cyclo.galois_norm()
    if abs(norm) == 1:  # Changed to check for both 1 and -1
      units.append(cyclo)
  return units


def binary_gcd(a: Union[Cyclotomic10, RealCyclotomic10], b: Union[Cyclotomic10, RealCyclotomic10]) -> Cyclotomic10:
  print("A: ", a)
  print("B: ", b)
  if isinstance(a, RealCyclotomic10):
    a = a.to_cycl()
  if isinstance(b, RealCyclotomic10):
    b = b.to_cycl()

  if a == Cyclotomic10.Zero():
    return b
  if b == Cyclotomic10.Zero():
    return a

  if a.divides_by_one_plus_omega() and b.divides_by_one_plus_omega():
    a1 = a.div_by_one_plus_omega()
    b1 = b.div_by_one_plus_omega()
    gcd_inner = binary_gcd(a1, b1)
    one_plus_omega = Cyclotomic10(1, 1, 0, 0)
    return one_plus_omega * gcd_inner

  u = inverse_mod_one_plus_omega(a) if not a.divides_by_one_plus_omega() else Cyclotomic10.One()
  v = inverse_mod_one_plus_omega(b) if not b.divides_by_one_plus_omega() else Cyclotomic10.One()

  c = a if N(a) <= N(b) else b

  next_input = u * a - v * b

  return binary_gcd(c, next_input)


def gcd(a: Cyclotomic10, b: Cyclotomic10) -> Cyclotomic10:
  while b != Cyclotomic10.Zero():
    q, r = divmod(a, b)
    assert a == b * q + r, f"Verification failed: a = {a}, b*q + r = {b * q + r}"
    a, b = b, r
  return a


def solve_norm_equation(xi: RealCyclotomic10) -> Union[Cyclotomic10, str]:
  """
  Outputs x \in Z[\omega] such that |x|² = xi \in Z[\tau]
  """
  if xi.evaluate() < 0 or xi.automorphism().evaluate() < 0:
    return "Unsolved"
  fl = EASY_FACTOR(xi)
  if not EASY_SOLVABLE(fl):
    return "Unsolved"
  x: Cyclotomic10 = Cyclotomic10.One()
  for i in range(len(fl)):
    xii: RealCyclotomic10 = fl[i][0]
    if isinstance(xii, int):
      xii = RealCyclotomic10(xii, 0)
    m: int = fl[i][1]
    x = x * (xii ** (m // 2))  # TODO: implement rmul and pow for ZTAU
    if m % 2 == 1:
      if xii.a == 5 and xii.b == 0:
        x = x * (RealCyclotomic10(1, 2))
      else:
        if xii == RealCyclotomic10(2, -1):
          x = x * (Cyclotomic10.Omega() + Cyclotomic10.Omega_(4))
        else:
          M: Tuple[int, int] = splitting_root(xii)
          #          assert (RealCyclotomic10.from_int(M[0]) - (RealCyclotomic10(2, -1))) % xi == RealCyclotomic10.Zero(), "M^2 != 2 - tau % xi"
          y: Cyclotomic10 = gcd(xii.to_cycl(), Cyclotomic10.from_int(M[0]) - (Cyclotomic10.Omega() + Cyclotomic10.Omega_(4)))
          u = xii.to_cycl() // N_i(y).to_cycl()  # TODO:
          s, m = UNIT_DLOG(u.to_subring())
          print("S, M: ", s, m)
          assert s == 1 and m % 2 == 0, "Unit DLOG failed for unit: " + str(u)
          print("X before mul: , ", x)
          print(f"X = {x} * tau^{m // 2} * {y}")
          x = x * (Cyclotomic10.Tau() ** (m // 2)) * y
          print("X in norm equation: ", x)
  return x


def inverse_mod_one_plus_omega(a: Cyclotomic10) -> Cyclotomic10:
  """
  Finds a unit u such that u * a ≡ 1 (mod 1 + ω)
  Returns None if no such unit exists.
  """
  if a.divides_by_one_plus_omega():
    raise ValueError("Element divisible by 1 + ω has no inverse modulo 1 + ω")

  # Use the brute force units to find the inverse
  units = _brute_force_unit()
  one_plus_omega = Cyclotomic10(1, 1, 0, 0)

  # Try each unit to find one that works
  for u in units:
    product = u * a
    remainder = product.integer_remainder_mod_one_plus_omega() % 5
    if remainder == 1:
      return u

  raise ValueError(f"No unit inverse found for {a} modulo 1 + ω")


def main2():
  # θ between 0 and π/5
  theta = math.pi / 10
  epsilon = 0.01
  r = 2

  result = RANDOM_SAMPLE(theta, epsilon, r)
  print("Sampled value:", result)

  real_test = 3.14141414
  n = 20
  c = APPROX_REAL(real_test, n)
  PREC = CONSTANTS.TAU ** (n - 1) * (1 - CONSTANTS.TAU**n)

  print(f"Prec: {PREC}")
  print(f"Approx : {c.evaluate()}")

  test = RealCyclotomic10(2, -1)
  print(test.div_by_two_minus_tau())

  XI_Test = RealCyclotomic10(760, -780)
  print(EASY_FACTOR(XI_Test))

  # u = RealCyclotomic10(2, -1)  # should be τ^-1
  # s, k = UNIT_DLOG(u)
  # print(f"u = {s} * τ^{k}")  # should output -1 * τ^-1
  n = 10
  p = 13
  root1, root2 = tonelli_shanks(n, p)
  print(f"Square roots of {n} mod {p} are {root1} and {root2}")

  print("Splitting root of (15 - 8t): ", splitting_root(RealCyclotomic10(15, -8)))

  x = RealCyclotomic10(15, -8).to_cycl()
  print(x)
  z = Cyclotomic10(15, 0, -8, 8)
  print(z)
  part = Cyclotomic10.Omega() + Cyclotomic10.Omega_(4)
  print(part)
  print(part**2)
  print("Part^2 = 2 - tau: ", part**2 == RealCyclotomic10(2, -1).to_cycl())
  print("2-tau: ", RealCyclotomic10(2, -1).to_cycl())
  y = Cyclotomic10(63, 0, 0, 0) - part
  print("Y: ", y)

  q, r = divmod(x, y)
  print("q =", q)
  print("r =", r)
  print("y * q + r =", y * q + r)
  print("x =", x)
  print("Success:", y * q + r == x)
  print("Units for x: ", find_unit_inverse_mod_one_plus_omega(x))
  # exit (0)
  # print("Gcd: ", binary_gcd(y, z))
  a = Cyclotomic10(1, 0, 0, 0)
  b = Cyclotomic10(2, 1, 0, 0)
  import sys

  sys.setrecursionlimit(20)
  print("GCD++++++++++++++++++++++++++++")
  g = gcd(y, z)
  print("GCD:", g)
  print("Is unit: ", (g // Cyclotomic10(3, 2, -7, 7)).is_unit())
  print(solve_norm_equation(XI_Test))
  print("---------------------------")
  print(EASY_SOLVABLE([(2, 2), (5, 1), (RealCyclotomic10(2, -1), 1), (RealCyclotomic10(15, -8), 1)]))
  # for i in range(10):
  #  for j in range(10):
  #    for k in range(10):
  #      for l in range(10):
  #        c = Cyclotomic10(i, j, k, l)
  #        n = N(c)
  #        if n == 1:
  #          print(i, j, k, l)
  #          print(c.divides_by_one_plus_omega())
  #          print("Remainder: ",
  #                (c * z).integer_remainder_mod_one_plus_omega() % 5)
  #        c2 = -c
  #        n = N(c)
  #        if n == 1:
  #          print("-", i, j, k, l)
  #          print(c.divides_by_one_plus_omega())
  #          print("Remainder: ",
  #                (c2 * z).integer_remainder_mod_one_plus_omega() % 5)


def mu(u: RealCyclotomic10) -> int:
  a, b = u.a, u.b
  return a * b


if __name__ == "__main__":
  x = [(2, 2), (5, 1), (RealCyclotomic10(2, -1), 1), (RealCyclotomic10(15, -8), 1)]
  XX = x
  xi = RealCyclotomic10(760, -780)
  EF = EASY_FACTOR(xi)
  print("EASY_FACTOR: ", EF)
  print("EASY_SOLVABLE: ", EASY_SOLVABLE(EF))
  x = solve_norm_equation(xi)
  print("Solve norm equation: ", x)
  ref = 2 * (RealCyclotomic10(4, 3).to_cycl()) * (Cyclotomic10(12, -20, 15, -3))
  print("X^2: ", N_i(x).evaluate())
  print("Reference: ", N_i(ref).evaluate())
  print("XI: ", xi.evaluate())
  print("XI / X^2", xi.evaluate() / N_i(x).evaluate())
  exit(0)
  print("Unit DLOG: ", UNIT_DLOG(RealCyclotomic10(1, 0)))

  xii = XX[3][0]  # RealCyclotomic10(15, -8)
  M = splitting_root(xii)
  print("Splitting root M: ", M[0], M[1])
  print("M^2: ", M[0] ** 2)
  tau_minus_2 = RealCyclotomic10(-2, 1)

  # Compute M^2 as an integer
  M_squared = M[0] ** 2
  xi = xii
  # Compute the difference as a RealCyclotomic10
  diff = RealCyclotomic10(M_squared, 0) - tau_minus_2

  # Now check if diff is divisible by xi
  q, r = divmod(diff.to_cycl(), xi.to_cycl())
  print(f"Quotient: {q}, Remainder: {r}")
  y: Cyclotomic10 = gcd(xii.to_cycl(), Cyclotomic10.from_int(M[0]) - (Cyclotomic10.Omega() + Cyclotomic10.Omega_(4)))
  print("Y: ", y)
  y = Cyclotomic10(3, 2, -7, 7)
  print("Y: ", y)
  absysquared = N_i(y)
  print("Abs(y)^2: ", absysquared)
  inv = absysquared.to_cycl().inv()
  print("Inverse of Abs(y)^2: ", inv)
  print("XII: ", xii.to_cycl())
  u = xii.to_cycl() // absysquared.to_cycl()
  print("Unit u: ", xii.to_cycl() // absysquared.to_cycl())
  s, m = UNIT_DLOG(u.to_subring())
  print("S, M: ", s, m)
  c = s * (RealCyclotomic10.Tau() ** m)
  print("Check: ", c)
