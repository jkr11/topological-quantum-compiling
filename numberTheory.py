import math
import random
from rings import Cyclotomic10, RealCyclotomic10, N
from typing import List, Tuple, Union

SQRT_5 = math.sqrt(5)
TAU = (SQRT_5 - 1) / 2  # Here we use this directly, no need to factor down from the rings
PHI = TAU + 1


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
  PREC = TAU ** (n - 1) * (1 - TAU**n)

  p = fibonacci(n)
  q = fibonacci(n + 1)
  u = (-1) ** (n + 1) * p
  v = (-1) ** n * fibonacci(n - 1)

  assert u * p + v * q == 1

  c = round(x * q)
  a = c * v + p * round((c * u) / q)
  b = c * u - q * round((c * u) / q)

  assert (abs(x - (a + b * TAU)) <= PREC) and (abs(b) <= PHI**n)
  return RealCyclotomic10(a, b)


# RANDOM-SAMPLE procedure
def RANDOM_SAMPLE(theta: float, epsilon: float, r: float) -> Cyclotomic10:
  assert r >= 1, f"r needs to be >= 1 but was {r}."
  # print(f"RANDOM_SAMPLE with {theta}, {epsilon}, {r}")
  C: float = math.sqrt(PHI / (4 * r))
  # print(f"C: {C}")
  m: int = math.ceil(math.log(C * epsilon * r, TAU)) + 1
  # print(f"math.log(C * epsilon * r), {math.log(C * epsilon * r, TAU)}")
  # print(f"m : {m}")
  N: int = math.ceil(PHI**m)

  sin_theta: float = math.sin(theta)
  cos_theta: float = math.cos(theta)

  sqrt_expr: float = math.sqrt(4 - epsilon**2)
  ymin: float = r * PHI**m * (sin_theta - epsilon * (sqrt_expr * cos_theta + epsilon * sin_theta) / 2)
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
  print("Part 1 : {part1}")
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
  if n < 2:
    return False
  for i in range(2, int(n**0.5) + 1):
    if n % i == 0:
      return False
  return True


def EASY_SOLVABLE(fl: List[Tuple[RealCyclotomic10, int]]) -> bool:
  for i in range(0, len(fl)):
    xi, k = fl[i]
    if k % 2 == 1:
      if xi != RealCyclotomic10(5, 0):
        p = xi.norm()
        r = p % 5
        if not IS_PRIME(p) or r not in [0, 1]:
          return False
  return True


# TODO: fix
def EASY_FACTOR(xi: RealCyclotomic10) -> List[Tuple[RealCyclotomic10, int]]:
  a, b = xi.a, xi.b
  c = math.gcd(a, b)
  a1 = a / c
  b1 = b / c
  xi1 = RealCyclotomic10(a1, b1)
  ret = None
  d = is_square(c)
  if d is not None:
    ret = [(d, 2)]
  else:
    d = is_square(c // 5) if c % 5 == 0 else None
    if d is not None:
      ret = [(d, 2), (5, 1)]
    else:
      return [(xi1, 1)]
  n = xi1.norm().evaluate()
  if n % 5 == 0:
    print(xi1)
    xi2 = xi1.div_by_two_minus_tau()
    ret.append([(RealCyclotomic10(2, -1), 1), (xi2, 1)])
    return ret
  else:
    ret.append([xi1, 1])
    return ret


def UNIT_DLOG(u: RealCyclotomic10) -> Tuple[int, int]:
  assert abs(u.norm().evaluate()) == 1
  a, b = u.a, u.b
  s, k = 1, 0
  if a < 0:
    a, b = -a, -b
    s = -s

  mu = a * b
  while abs(mu) > 1:
    if mu > 1:
      a, b = b, a - b
      k = k - 1
    else:
      a, b = a, a - b
      k = k + 1
    mu = a * b

  def tau_power(n: int) -> Tuple[int, int]:
    a, b = 1, 0
    if n == 0:
      return (a, b)
    elif n > 0:
      a, b = 0, 1
      for _ in range(1, n):
        a, b = b, a - b
      return (a, b)
    else:
      a, b = 0, 1
      for _ in range(-1, -n, 1):
        a, b = b - a, a
      return (a, b)

  for i in range(-3, 4):
    ta, tb = tau_power(i)
    if (a, b) == (ta, tb):
      k += i
      return (s, k)

  raise ValueError("Failed to find final match")


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

  p: int = tau_norm_(xi)
  assert p % 2 == 1 and p % 5 == 1, "p = N_tau (xi) must be an odd prime = 1 "
  assert xi.b % p != 0
  b1 = _mod_inv(xi.b, p)
  n: int = (-xi.a * b1 - 2) % p
  return tonelli_shanks(n, p)


def _get_unit_for_residue(r: int) -> Cyclotomic10:
  print("Residue ", r)
  if r == 1:
    return Cyclotomic10(1, 0, 0, 0)
  elif r == 4:
    return Cyclotomic10(-1, 0, 0, 0)
  elif r == 2:
    return Cyclotomic10(1, 0, -1, -1)
  elif r == 3:
    return Cyclotomic10(1, 0, 1, 0)
  else:
    raise ValueError("Invalid residue")


def binary_gcd(a: Union[Cyclotomic10, RealCyclotomic10], b: Union[Cyclotomic10, RealCyclotomic10]) -> Cyclotomic10:
  if isinstance(a, RealCyclotomic10):
    a = a.to_cycl()
  if isinstance(b, RealCyclotomic10):
    b = b.to_cycl()
  if a == Cyclotomic10.Zero():
    return b
  elif b == Cyclotomic10.Zero():
    return a
  print("A: ", a)
  print("B: ", b)
  a_res = a.integer_remainder_mod_one_plus_omega() % 5
  b_res = b.integer_remainder_mod_one_plus_omega() % 5
  print("A residue ", a_res)
  print("B residue ", b_res)
  if a_res == 0 and b_res == 0:
    print("a and b are divisible by one plus omega")
    print(f"N(a) = {N(a)}, N(b) = {N(b)}")
    a1 = a.div_by_one_plus_omega()
    b1 = b.div_by_one_plus_omega()
    return (Cyclotomic10.One() + Cyclotomic10.Omega()) * binary_gcd(a1, b1)

  u = v = Cyclotomic10.One()
  if a_res != 0 and b_res != 0:
    u = _get_unit_for_residue(a_res)
    assert (a * u).integer_remainder_mod_one_plus_omega() % 5 == 1
    print(a * u)
    v = _get_unit_for_residue(b_res)
    print("b:", b)
    print("v:", v)
    result = b * v
    print("b * v:", result)
    print("b * v mod (1 + ω):", result.integer_remainder_mod_one_plus_omega())

    assert (b * v).integer_remainder_mod_one_plus_omega() % 5 == 1

  if a.galois_norm() <= b.galois_norm():
    c = a
    difference = (u * a) - (v * b)
  else:
    c = b
    difference = (v * b) - (u * a)

  if difference == Cyclotomic10.Zero():
    return c
  print("Diff: ", difference)
  return binary_gcd(c, difference)


def solve_norm_equation(xi: RealCyclotomic10) -> Union[Cyclotomic10, str]:
  """
  Outputs x \in Z[\omega] such that |x|² = xi \in Z[\tau]
  """
  raise NotImplementedError
  if xi.evaluate() < 0 or xi.automorphism().evaluate() < 0:
    return "Unsolved"
  fl = EASY_FACTOR(xi)
  if not EASY_SOLVABLE(fl):
    return "Unsolved"
  x: RealCyclotomic10 = RealCyclotomic10(1, 0)
  for i in range(len(fl) - 1):
    xii: RealCyclotomic10 = fl[i][0]
    m: int = fl[i][1]
    x = x * xii ** (m // 2)  # TODO: implement rmul and pow for ZTAU
    if m % 2 == 1:
      if xii.a == 5 and xii.b == 0:
        x = x * (RealCyclotomic10(1, 2))
      else:
        if xii.a == 2 and xii.b == -1:
          x = x * Cyclotomic10(-1, 2, -1, 1)  # w + w^4
        else:
          M: Tuple[int, int] = splitting_root(xii)
          # TODO: think about how to do efficient divisions


if __name__ == "__main__":
  # θ between 0 and π/5
  theta = math.pi / 10
  epsilon = 0.01
  r = 2

  result = RANDOM_SAMPLE(theta, epsilon, r)
  print("Sampled value:", result)

  real_test = 3.14141414
  n = 20
  c = APPROX_REAL(real_test, n)
  PREC = TAU ** (n - 1) * (1 - TAU**n)

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
  y = Cyclotomic10(63, -1, 0, 0) + Cyclotomic10.from_omega_4(-1)
  print("Y: ", y.integer_remainder_mod_one_plus_omega() % 5)
  print(binary_gcd(x, y))

  z = Cyclotomic10(15, 0, -8, 8)

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
