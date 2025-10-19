from exact_synthesis.rings import ZOmega, ZTau
import mpmath
import random
import matplotlib.pyplot as plt
import numpy as np


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


def RANDOM_SAMPLE_DEBUG(theta, epsilon, r):
  TAU = (mpmath.sqrt(5) - 1) / 2
  PHI = TAU + 1
  tau = mpmath.mpf(TAU)
  phi = mpmath.mpf(PHI)
  theta = mpmath.mpf(theta)
  assert 0 <= theta < mpmath.pi / 5, f"was {theta} instead"
  assert r >= 1, f"r must be >= 1, was {r}"
  epsilon = mpmath.mpf(epsilon)
  r = mpmath.mpf(r)

  C = mpmath.sqrt(phi / (4 * r))
  m = int(mpmath.ceil(mpmath.log(C * epsilon * r, tau))) + 1
  N = int(mpmath.ceil(phi**m))

  sin_theta = mpmath.sin(theta)
  cos_theta = mpmath.cos(theta)
  sqrt_expr = mpmath.sqrt(4 - epsilon**2)
  rpm = r * phi**m

  ymin = rpm * (sin_theta - epsilon * (sqrt_expr * cos_theta + epsilon * sin_theta) / 2)
  ymax = rpm * (sin_theta + epsilon * (sqrt_expr * cos_theta - epsilon * sin_theta) / 2)

  xmax = rpm * ((1 - epsilon**2 / 2) * cos_theta - epsilon * mpmath.sqrt(1 - epsilon**2 / 4) * sin_theta)
  xc = xmax - (rpm * epsilon**2) / (4 * cos_theta)

  j = random.randint(1, N - 1)
  y = ymin + j * (ymax - ymin) / N

  y_prime = y / mpmath.sqrt(2 - tau)
  print("M : ", m)
  yy = approx_real(y_prime, m)

  print("YPRIME: ", y_prime)
  print("YPRIMEAPPROX: ", yy.evaluate())
  # TODO: handle y_prime completely with Cyclotomic10
  y_approx = (yy.evaluate()) * mpmath.sqrt(2 - tau)
  x = xc - (y_approx - ymin) * mpmath.tan(theta)

  xx = approx_real(x, m)

  part1 = xx
  part2 = yy.to_cycl() * (ZOmega.Omega() + ZOmega.Omega_(4))
  result = part1.to_cycl() + part2

  return result, ymin, ymax, xc, yy, xx, x, xmax, m


def plot_random_samples(theta, epsilon, r, num_samples=200):
  mpmath.mp.dps = 200  # Set precision

  res, ymin, ymax, xc, yy, xx, x, xmax, m = RANDOM_SAMPLE_DEBUG(theta, epsilon, 1.0)
  # xmin_f = float(xmin)
  xmax_f = float(xmax)
  ymin_f = float(ymin)
  ymax_f = float(ymax)
  xleft = 2 * float(xc) - xmax_f
  # Define the corners of the parallelogram
  corners_bx = [xleft, xmax, float(xc)]
  corners_by = [ymin, ymin, ymin]

  xmin = xmax - (ymax - ymin) * mpmath.tan(theta)

  print(mpmath.sqrt((xmax_f - xmin) ** 2 + (ymax_f - ymin_f)))

  TAU = (mpmath.sqrt(5) - 1) / 2
  PHI = TAU + 1
  radius = r * PHI**m
  print(f"Radius = {r * PHI**m}")
  print(f"Distance (xmin, ymax): {mpmath.sqrt(ymax**2 + xmin**2)}")
  print(f"Width = {0.5 * r * epsilon**2 * PHI**m}")
  print(r * epsilon * PHI**m)

  topleft = xmin - (xmax - xleft)
  corners_tx = [xmin, topleft]
  corners_ty = [ymax_f, ymax]

  theta_line_length = float(ymax)  # or set to 1.5 or something custom

  plt.figure(figsize=(6, 6))

  plt.scatter([xleft], [ymin], color="orange")
  plt.scatter([xleft], [ymin], color="red", alpha=0.2)
  plt.scatter(corners_bx, corners_by, color="red", alpha=0.2)
  plt.scatter(corners_tx, corners_ty, color="blue", alpha=0.2)
  plt.scatter(float(x), float(yy.evaluate()), color="green")
  circle = plt.Circle((0, 0), radius, color="blue", fill=False, linestyle="--", alpha=0.5, label=f"Circle r·φ^m")
  plt.gca().add_patch(circle)
  margin = 0.1 * radius

  plt.xlim(float(xmin) - 3, float(xmax) + 3)
  plt.ylim(float(ymin) - 3, float(ymax) + 3)

  plt.gca().set_aspect("equal", adjustable="box")
  # Fill parallelogram
  # plt.fill(corners_x, corners_y, alpha=0.2, color='orange', label='ε-region')
  # Optional: Plot the center line
  # plt.plot([float(xc)], [ymin_f], 'rx', label='xc (base center)')
  # Optional: Plot the sample point
  # sample_point = complex(res.evaluate())
  # plt.plot([sample_point.real], [sample_point.imag], 'bo', markersize=8, label='Sample')
  # plt.gca().set_aspect('equal', adjustable='box')
  for _ in range(num_samples - 1):
    _, _, _, _, yy_i, _, x_i, _, _ = RANDOM_SAMPLE_DEBUG(theta, epsilon, r)
    plt.scatter(float(x_i), float(yy_i.evaluate()), color="green", s=10, alpha=0.5)

  plt.grid(True)
  plt.xlabel("Re")
  plt.ylabel("Im")
  plt.title("ε-region and Sample in Complex Plane")
  plt.legend()
  plt.show()


# xx = RANDOM_SAMPLE_DEBUG(1/10, 1e-10, 1.0)
# plot_random_samples(1/10, 1e-10, 1.0, 10)
