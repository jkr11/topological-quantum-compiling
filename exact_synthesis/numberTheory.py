import math
import random
from exact_synthesis.rings import ZOmega, ZTau, N_tau, N_i
from typing import List, Tuple, Union


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


def is_prime(p: int) -> bool:
  return miller_rabin(p)


def miller_rabin(n: int, k: int = 7) -> bool:
  """Use Miller-Rabin primality test to check if n is prime.
  k is number of accuracy rounds.
  """
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
  isprime = is_prime(Nt) and (Nt % 5 == 1)
  return is_positive and isprime


def easy_solvable(fl: List[Tuple[ZTau, int]]) -> bool:
  for i in range(0, len(fl)):
    xi, k = fl[i]
    if k % 2 == 0:
      pass
    elif xi != ZTau(5, 0):
      p: int = N_tau(xi)
      r = p % 5
      if not is_prime(p) or r not in [0, 1]:
        return False
  return True


def easy_factor(xi: ZTau) -> List[Tuple[ZTau, int]]:
  if isinstance(xi, int):
    xi = ZTau(xi, 0)
  a, b = xi.a, xi.b
  c = math.gcd(a, b)
  a1 = a // c
  b1 = b // c
  xi1 = ZTau(a1, b1)
  ret: List[Tuple[ZTau, int]] = []
  d: int = is_square(c)
  if d is not None:
    ret = [(ZTau(d, 0), 2)]
  else:
    d = is_square(c // 5) if c % 5 == 0 else None
    if d is not None:
      ret = [(ZTau(d, 0), 2), (ZTau(5, 0), 1)]
    else:
      # this means unsolvable
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


def unit_dlog(u: ZTau) -> Tuple[int, int]:
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


def splitting_root(xi: ZTau) -> Tuple[int, int]:
  def _mod_inv(b: int, p: int) -> int:
    return pow(b, -1, p)

  p: int = N_tau(xi)
  assert p % 2 == 1 and p % 5 == 1, "p = N_tau (xi) must be an odd prime = 1 "
  assert xi.b % p != 0
  b1 = _mod_inv(xi.b, p)
  n: int = (-xi.a * b1 - 2) % p
  return tonelli_shanks(n, p)


def gcd(a: ZOmega, b: ZOmega) -> ZOmega:
  while b != ZOmega.Zero():
    q, r = divmod(a, b)
    assert a == b * q + r, f"Verification failed: a = {a}, b*q + r = {b * q + r}"
    a, b = b, r
  return a


def solve_norm_equation(xi: ZTau) -> ZOmega:
  """
  Outputs x in Z[omega] such that |x|² = xi in Z[tau]
  """
  if xi.evaluate() < 0 or xi.automorphism().evaluate() < 0:
    raise ValueError(f"{xi.evaluate} or its aut is < 0.")
  fl = easy_factor(xi)
  if not easy_solvable(fl):
    raise ValueError(f"{xi} is not easy solvable.")
  x: ZOmega = ZOmega.One()
  for i in range(len(fl)):
    xii: ZTau = fl[i][0]
    if isinstance(xii, int):
      xii = ZTau(xii, 0)
    m: int = fl[i][1]
    x = x * (xii ** (m // 2))
    if m % 2 == 1:
      if xii == 5:
        x = x * (ZTau(1, 2))
      else:
        if xii == ZTau(2, -1):
          x = x * (ZOmega.Omega() + ZOmega.Omega_(4))
        else:
          M: Tuple[int, int] = splitting_root(xii)

          y: ZOmega = gcd(xii.to_cycl(), ZOmega.from_int(M[0]) - (ZOmega.Omega() + ZOmega.Omega_(4)))  # here, we have to cast up for representing sqrt(tau - 2), which is not in ZTau
          print(type(xii))
          u = xii // N_i(y)
          s, m = unit_dlog(u)
          assert s == 1 and m % 2 == 0, "Unit DLOG failed for unit: " + str(u)
          x = x * (ZOmega.Tau() ** (m // 2)) * y
  return x


def epsilon_region(r: float, theta: float):
  pass
