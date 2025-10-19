from exact_synthesis.rings import ZOmega, ZTau
import mpmath
import random


def approx_real(x, n) -> ZTau:
  TAU = (mpmath.sqrt(5) - 1) / 2
  PHI = TAU + 1
  x = mpmath.mpf(x)
  tau = mpmath.mpf(TAU)
  phi = mpmath.mpf(PHI)
  PREC = tau ** (n - 1) * (1 - tau**n)

  o = int(mpmath.fibonacci(n - 1))
  p = int(mpmath.fibonacci(n))
  q = int(mpmath.fibonacci(n + 1))
  u = ((-1) ** (n + 1)) * p
  v = ((-1) ** n) * o

  assert u * p + v * q == 1, f"Failed: u * p + v * q = {u * p + v * q}, expected 1"

  c = mpmath.nint(x * q)
  cuq = c * u / q
  nint_cuq = int(mpmath.nint(cuq))
  a = c * v + p * nint_cuq
  b = c * u - q * nint_cuq

  approx = a + b * tau
  abs_diff = mpmath.fabs(x - approx)
  abs_b = mpmath.fabs(b)
  phi_pow_n = phi**n
  assert (abs_diff <= PREC) and (abs_b <= phi_pow_n), f"Failed: abs(x - approx)={abs_diff} (PREC={PREC}), abs(b)={abs_b} (PHI^n={phi_pow_n})"
  return ZTau(int(a), int(b))


def random_sample(theta, epsilon, r) -> ZOmega:
  TAU = (mpmath.sqrt(5) - 1) / 2
  PHI = TAU + 1
  tau = mpmath.mpf(TAU)
  phi = mpmath.mpf(PHI)
  theta = mpmath.mpf(theta)
  assert 0 <= theta < mpmath.pi / 5, f"was {theta} instead"
  epsilon = mpmath.mpf(epsilon)
  r = mpmath.mpf(r)

  C = mpmath.sqrt(phi / (4 * r))
  m = int(mpmath.ceil(mpmath.log(C * epsilon * r) / mpmath.log(tau))) + 1
  N = int(mpmath.ceil(phi**m))

  sin_theta = mpmath.sin(theta)
  cos_theta = mpmath.cos(theta)
  sqrt_expr = mpmath.sqrt(4 - epsilon**2)

  ymin = r * phi**m * (sin_theta - epsilon * (sqrt_expr * cos_theta + epsilon * sin_theta) / 2)
  ymax = r * phi**m * (sin_theta + epsilon * (sqrt_expr * cos_theta - epsilon * sin_theta) / 2)

  xmax = r * phi**m * ((1 - epsilon**2 / 2) * cos_theta - epsilon * mpmath.sqrt(1 - epsilon**2 / 4) * sin_theta)
  xc = xmax - (r * epsilon**2 * phi**m) / (4 * cos_theta)

  j = random.randint(1, N - 1)
  y = ymin + j * (ymax - ymin) / N

  y_prime = y / mpmath.sqrt(2 - tau)
  yy = approx_real(y_prime, m)

  y_approx = (yy.a + yy.b * tau) * mpmath.sqrt(2 - tau)
  x = xc - (y_approx - ymin) * mpmath.tan(theta)

  xx = approx_real(x, m)

  part1 = xx
  part2 = yy.to_cycl() * (ZOmega.Omega() + ZOmega.Omega_(4))
  result = part1.to_cycl() + part2

  return result
