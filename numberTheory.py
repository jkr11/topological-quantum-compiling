import math
import random
from rings import *
import cmath
from typing import List, Tuple, Union

SQRT_5 = math.sqrt(5)
TAU = (
    SQRT_5 -
    1) / 2  # Here we use this directly, no need to factor down from the rings
PHI = TAU + 1


def fibonacci(n):
  assert n >= 0
  a, b = 0, 1
  for _ in range(n):
    a, b = b, a + b
  return a


def extended_fib_coefficients(n):
  assert n >= 0
  Fn = fibonacci(n)
  Fn_minus_1 = fibonacci(n - 1)
  u = (-1)**(n + 1) * Fn
  v = (-1)**n * Fn_minus_1
  return u, v


# APPROX-REAL procedure
# TODO: make this return Z[\tau] object for consistency with paper
def APPROX_REAL(x, n) -> Tuple[int, int]:
  PREC = TAU**(n - 1) * (1 - TAU**n)

  p = fibonacci(n)
  q = fibonacci(n + 1)
  u = (-1)**(n + 1) * p
  v = (-1)**n * fibonacci(n - 1)

  assert (u * p + v * q == 1)

  c = round(x * q)
  a = c * v + p * round((c * u) / q)
  b = c * u - q * round((c * u) / q)

  assert ((abs(x - (a + b * TAU)) <= PREC) and (abs(b) <= PHI**n))
  return a, b


# RANDOM-SAMPLE procedure
def RANDOM_SAMPLE(theta: float, epsilon: float, r: float):
  assert (r >= 1, f"r needs to be >= 1 but was {r}.")
  print(f"RANDOM_SAMPLE with {theta}, {epsilon}, {r}")
  C = math.sqrt(PHI / (4 * r))
  print(f"C: {C}")
  m = math.ceil(math.log(C * epsilon * r, TAU)) + 1
  print(f"math.log(C * epsilon * r), {math.log(C * epsilon * r, TAU)}")
  print(f"m : {m}")
  N = math.ceil(PHI**m)

  sin_theta = math.sin(theta)
  cos_theta = math.cos(theta)

  sqrt_expr = math.sqrt(4 - epsilon**2)
  ymin = r * PHI**m * (sin_theta - epsilon *
                       (sqrt_expr * cos_theta + epsilon * sin_theta) / 2)
  ymax = r * PHI**m * (sin_theta + epsilon *
                       (sqrt_expr * cos_theta - epsilon * sin_theta) / 2)

  xmax = r * PHI**m * ((1 - epsilon**2 / 2) * cos_theta -
                       epsilon * math.sqrt(1 - epsilon**2 / 4) * sin_theta)
  xc = xmax - (r * epsilon**2 * PHI**m) / (4 * cos_theta)

  # Random sampling
  j = random.randint(1, N - 1)
  y = ymin + j * (ymax - ymin) / N

  # Step 11: Approximate y'
  y_prime = y / math.sqrt(2 - TAU)
  ay, by = APPROX_REAL(y_prime, m)

  # Step 12: x calculation
  y_approx = (ay + by * TAU) * math.sqrt(2 - TAU)
  x = xc - (y_approx - ymin) * math.tan(theta)

  # Step 13: Approximate x
  ax, bx = APPROX_REAL(x, m)

  # Step 14: Final return
  part1 = ax + bx * TAU
  print("Part 1 : {part1}")
  part2 = (ay + by * TAU) * cmath.sqrt(TAU - 2)

  return part1 + part2


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
    xi2 = xi1.div_by_two_minus_tau()
    return ret.append([(RealCyclotomic10(2, -1), 1), (xi2, 1)])
  else:
    return ret.append([xi1, 1])


if __name__ == "__main__":
  # θ between 0 and π/5
  theta = math.pi / 10
  epsilon = 0.01
  r = 2

  result = RANDOM_SAMPLE(theta, epsilon, r)
  print("Sampled value:", result)

  real_test = 3.14141414
  n = 20
  a, b = APPROX_REAL(real_test, n)
  PREC = TAU**(n - 1) * (1 - TAU**n)

  print(f"Prec: {PREC}")
  print(f"Approx : {a + b * TAU}")

  XI_Test = RealCyclotomic10(760, -780)
  print(EASY_FACTOR(XI_Test))
