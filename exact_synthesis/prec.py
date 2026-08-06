import random
from mpmath import mp

from exact_synthesis.rings import ZOmega, ZTau

SQRT_TAU_MINUS_2 = ZOmega(-1, 2, -1, 1)


def _fibonacci(n: int) -> int:
  if n >= 0:
    a, b = 0, 1
    for _ in range(n):
      a, b = b, a + b
    return a
  k = -n
  fk = _fibonacci(k)
  return fk if k % 2 == 1 else -fk


def approx_real(x, n: int) -> ZTau:
  x = mp.mpf(x)

  p = _fibonacci(n)
  q = _fibonacci(n + 1)
  u = (-1) ** (n + 1) * _fibonacci(n)
  v = (-1) ** n * _fibonacci(n - 1)
  assert u * p + v * q == 1, "Invariant invalid"

  if q == 0:
    raise ValueError(f"approx_real: F_(n+1) is zero for n={n}; choose n >= 0.")

  c = int(mp.nint(x * q))
  cu = c * u
  floor_cu_over_q = cu // q

  a = c * v + p * floor_cu_over_q
  b = c * u - q * floor_cu_over_q

  return ZTau(a, b)


def random_sample(theta, epsilon, r=1) -> ZOmega:
  theta = mp.mpf(theta)
  epsilon = mp.mpf(epsilon)
  r = mp.mpf(r)

  phi = (1 + mp.sqrt(5)) / 2
  tau = phi - 1  # = 1/phi

  C = mp.sqrt(phi / (4 * r))
  m = int(mp.ceil(mp.log(C * epsilon * r) / mp.log(tau))) + 1
  N = int(mp.ceil(phi**m))

  if N < 2:
    raise ValueError(f"random_sample: N={N} leaves no valid range for j; epsilon={epsilon} is too large for r={r} (increase precision or shrink epsilon).")

  sin_t, cos_t = mp.sin(theta), mp.cos(theta)
  sqrt4_eps2 = mp.sqrt(4 - epsilon**2)

  y_min = r * phi**m * (sin_t - epsilon * (sqrt4_eps2 * cos_t + epsilon * sin_t) / 2)
  y_max = r * phi**m * (sin_t + epsilon * (sqrt4_eps2 * cos_t - epsilon * sin_t) / 2)
  x_max = r * phi**m * ((1 - epsilon**2 / 2) * cos_t - epsilon * mp.sqrt(1 - epsilon**2 / 4) * sin_t)
  x_c = x_max - r * epsilon**2 * phi**m / (4 * cos_t)

  j = random.randint(1, N - 1)
  y = y_min + j * (y_max - y_min) / N

  sqrt_2_minus_tau = mp.sqrt(2 - tau)

  ay_by = approx_real(y / sqrt_2_minus_tau, m)
  a_y, b_y = ay_by.a, ay_by.b

  x = x_c - ((a_y + b_y * tau) * sqrt_2_minus_tau - y_min) * mp.tan(theta)

  ax_bx = approx_real(x, m)
  a_x, b_x = ax_bx.a, ax_bx.b

  real_part = ZTau(a_x, b_x).to_cycl()
  imag_part = SQRT_TAU_MINUS_2 * ZTau(a_y, b_y).to_cycl()

  return real_part + imag_part


def _random_sample(theta, epsilon, r=1):
  theta = mp.mpf(theta)
  epsilon = mp.mpf(epsilon)
  r = mp.mpf(r)

  phi = (1 + mp.sqrt(5)) / 2
  tau = phi - 1  # = 1/phi

  C = mp.sqrt(phi / (4 * r))
  m = int(mp.ceil(mp.log(C * epsilon * r) / mp.log(tau))) + 1
  N = int(mp.ceil(phi**m))

  if N < 2:
    raise ValueError(f"random_sample: N={N} leaves no valid range for j; epsilon={epsilon} is too large for r={r} (increase precision or shrink epsilon).")

  sin_t, cos_t = mp.sin(theta), mp.cos(theta)
  sqrt4_eps2 = mp.sqrt(4 - epsilon**2)

  y_min = r * phi**m * (sin_t - epsilon * (sqrt4_eps2 * cos_t + epsilon * sin_t) / 2)
  y_max = r * phi**m * (sin_t + epsilon * (sqrt4_eps2 * cos_t - epsilon * sin_t) / 2)
  x_max = r * phi**m * ((1 - epsilon**2 / 2) * cos_t - epsilon * mp.sqrt(1 - epsilon**2 / 4) * sin_t)
  x_c = x_max - r * epsilon**2 * phi**m / (4 * cos_t)

  j = random.randint(1, N - 1)
  y = y_min + j * (y_max - y_min) / N

  sqrt_2_minus_tau = mp.sqrt(2 - tau)

  ay_by = approx_real(y / sqrt_2_minus_tau, m)
  a_y, b_y = ay_by.a, ay_by.b

  x = x_c - ((a_y + b_y * tau) * sqrt_2_minus_tau - y_min) * mp.tan(theta)

  ax_bx = approx_real(x, m)
  a_x, b_x = ax_bx.a, ax_bx.b

  real_part = ZTau(a_x, b_x).to_cycl()
  imag_part = SQRT_TAU_MINUS_2 * ZTau(a_y, b_y).to_cycl()

  return real_part + imag_part, ay_by, y
