import math
import random
from single_qubit.exact_synthesis.rings import Cyclotomic10, ZTau, N_tau, N_i
from typing import List, Tuple, Union
from single_qubit.exact_synthesis.util import CONSTANTS
from single_qubit.exact_synthesis.prec import APPROX_REAL, RANDOM_SAMPLE


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


def is_square(n: int) -> Union[int, None]:
  if n < 0:
    return None
  root = math.isqrt(n)
  if root * root == n:
    return root
  return None


def IS_PRIME(p: int) -> bool:
  return miller_rabin(p)


def miller_rabin(n: int, k: int = 10) -> bool:
  if n < 2:
    return False

  small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
  for sp in small_primes:
    if n == sp:
      return True
    if n % sp == 0 and n != sp:
      return False

  r, d = 0, n - 1
  while d % 2 == 0:
    d >>= 1
    r += 1

  for _ in range(k):
    a = random.randrange(2, n - 1)
    x = pow(a, d, n)
    if x == 1 or x == n - 1:
      continue
    for __ in range(r - 1):
      x = pow(x, 2, n)
      if x == n - 1:
        break
    else:
      return False
  return True


def easy_solvable_predicate(xi: ZTau):
  Nt = N_tau(xi)
  is_positive = xi > 0 and xi.automorphism() > 0
  is_prime = IS_PRIME(Nt) and (Nt % 5 == 1)
  return is_positive and is_prime


def EASY_SOLVABLE(fl: List[Tuple[ZTau, int]]) -> bool:
  for i in range(0, len(fl)):
    xi, k = fl[i]
    if k % 2 == 1:
      if xi != ZTau(5, 0):
        p: int = N_tau(xi)
        r = p % 5
        if not IS_PRIME(p) or r not in [0, 1]:
          return False
  return True


def EASY_FACTOR(xi: ZTau) -> List[Tuple[ZTau, int]]:
  if isinstance(xi, int):
    xi = ZTau(xi, 0)
  a, b = xi.a, xi.b
  c = math.gcd(a, b)
  a1 = a // c
  b1 = b // c
  xi1 = ZTau(a1, b1)
  ret = []
  d: int = is_square(c)
  if d is not None:
    ret = [(d, 2)]
  else:
    d = is_square(c // 5) if c % 5 == 0 else None
    if d is not None:
      ret = [(d, 2), (5, 1)]
    else:
      return [(xi1, 1)]
  n = N_tau(xi1)
  if n % 5 == 0:
    xi2 = xi1.div_by_two_minus_tau()
    ret.append((ZTau(2, -1), 1))
    ret.append((xi2, 1))
    return ret
  else:
    ret.append((xi1, 1))
    return ret


def UNIT_DLOG(u: ZTau) -> Tuple[int, int]:
  """
  Finds discrete logarithm of a unit u in Z[τ]
  Returns (s,k) such that u = s * τ^k where s = ±1
  """
  # TODO: assert unit
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
  assert abs(mu) == 1, "Failed to reduce unit to base units"

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
    case _:
      pass
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


def splitting_root(xi: ZTau) -> Tuple[int, int]:
  def _mod_inv(b: int, p: int) -> int:
    return pow(b, -1, p)

  p: int = N_tau(xi)
  assert p % 2 == 1 and p % 5 == 1, "p = N_tau (xi) must be an odd prime = 1 "
  assert xi.b % p != 0
  b1 = _mod_inv(xi.b, p)
  n: int = (-xi.a * b1 - 2) % p
  return tonelli_shanks(n, p)


# Returns the same as binary_gcd up to a multiplicative unit
def gcd(a: Cyclotomic10, b: Cyclotomic10) -> Cyclotomic10:
  while b != Cyclotomic10.Zero():
    q, r = divmod(a, b)
    a, b = b, r
  return a


def solve_norm_equation(xi: ZTau) -> Union[Cyclotomic10, str]:
  """
  Outputs x in Z[omega] such that |x|² = xi in Z[tau]
  """
  if xi.evaluate() < 0 or xi.automorphism().evaluate() < 0:
    return "Unsolved"
  fl = EASY_FACTOR(xi)
  if not EASY_SOLVABLE(fl):
    return "Unsolved"
  x: Cyclotomic10 = Cyclotomic10.One()
  for i in range(len(fl)):
    xii: ZTau = fl[i][0]
    if isinstance(xii, int):
      xii = ZTau(xii, 0)
    m: int = fl[i][1]
    x = x * (xii ** (m // 2))
    if m % 2 == 1:
      if xii.a == 5 and xii.b == 0:
        x = x * (ZTau(1, 2))
      else:
        if xii == ZTau(2, -1):
          x = x * (Cyclotomic10.Omega() + Cyclotomic10.Omega_(4))
        else:
          M: Tuple[int, int] = splitting_root(xii)

          y: Cyclotomic10 = gcd(
            xii.to_cycl(),
            Cyclotomic10.from_int(M[0]) - (Cyclotomic10.Omega() + Cyclotomic10.Omega_(4)),
          )
          u = xii.to_cycl() // N_i(y).to_cycl()  # TODO:
          s, m = UNIT_DLOG(u.to_subring())
          assert s == 1 and m % 2 == 0, "Unit DLOG failed for unit: " + str(u)
          x = x * (Cyclotomic10.Tau() ** (m // 2)) * y
  return x


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

  test = ZTau(2, -1)
  print(test.div_by_two_minus_tau())

  XI_Test = ZTau(760, -780)
  print(EASY_FACTOR(XI_Test))

  # u = RealCyclotomic10(2, -1)  # should be τ^-1
  # s, k = UNIT_DLOG(u)
  # print(f"u = {s} * τ^{k}")  # should output -1 * τ^-1
  n = 10
  p = 13
  root1, root2 = tonelli_shanks(n, p)
  print(f"Square roots of {n} mod {p} are {root1} and {root2}")

  print("Splitting root of (15 - 8t): ", splitting_root(ZTau(15, -8)))

  x = ZTau(15, -8).to_cycl()
  print(x)
  z = Cyclotomic10(15, 0, -8, 8)
  print(z)
  part = Cyclotomic10.Omega() + Cyclotomic10.Omega_(4)
  print(part)
  print(part**2)
  print("Part^2 = 2 - tau: ", part**2 == ZTau(2, -1).to_cycl())
  print("2-tau: ", ZTau(2, -1).to_cycl())
  y = Cyclotomic10(63, 0, 0, 0) - part
  print("Y: ", y)

  q, r = divmod(x, y)
  print("q =", q)
  print("r =", r)
  print("y * q + r =", y * q + r)
  print("x =", x)
  print("Success:", y * q + r == x)
  # exit (0)
  # print("Gcd: ", binary_gcd(y, z))
  print("GCD++++++++++++++++++++++++++++")
  g = gcd(y, z)
  print("GCD:", g)
  print("Is unit: ", (g // Cyclotomic10(3, 2, -7, 7)).is_unit())
  print(solve_norm_equation(XI_Test))
  print("---------------------------")
  print(EASY_SOLVABLE([(2, 2), (5, 1), (ZTau(2, -1), 1), (ZTau(15, -8), 1)]))
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


def mu(u: ZTau) -> int:
  a, b = u.a, u.b
  return a * b


if __name__ == "__main__":
  x = [(2, 2), (5, 1), (ZTau(2, -1), 1), (ZTau(15, -8), 1)]
  XX = x
  xi = ZTau(760, -780)
  EF = EASY_FACTOR(xi)
  print("EASY_FACTOR: ", EF)
  print("EASY_SOLVABLE: ", EASY_SOLVABLE(EF))
  x = solve_norm_equation(xi)
  print("Solve norm equation: ", x)
  ref = 2 * (ZTau(4, 3).to_cycl()) * (Cyclotomic10(12, -20, 15, -3))
  print("X^2: ", N_i(x).evaluate())
  print("Reference: ", N_i(ref).evaluate())
  print("XI: ", xi.evaluate())
  print("XI / X^2", xi.evaluate() / N_i(x).evaluate())
  # exit(0)
  print("Unit DLOG: ", UNIT_DLOG(ZTau(1, 0)))

  xii = XX[3][0]  # RealCyclotomic10(15, -8)
  M = splitting_root(xii)
  print("Splitting root M: ", M[0], M[1])
  print("M^2: ", M[0] ** 2)
  tau_minus_2 = ZTau(-2, 1)

  # Compute M^2 as an integer
  M_squared = M[0] ** 2
  xi = xii
  # Compute the difference as a RealCyclotomic10
  diff = ZTau(M_squared, 0) - tau_minus_2

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
  # c = s * (RealCyclotomic10.Tau() ** m)
  # print("Check: ", c)
