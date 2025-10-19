from exact_synthesis.rings import ZOmega, ZTau
import mpmath
import random
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle


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
  # print(f"Approx real of {x} with prec={PREC} and {abs_diff}")
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
  print("Prec: ", mpmath.mp.dps)
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
  # print("M : ", m)
  yy = approx_real(y_prime, m)

  # print("YPRIME: ", y_prime)
  # print("YPRIMEAPPROX: ", yy.evaluate())
  # TODO: handle y_prime completely with Cyclotomic10
  y_approx = (yy.evaluate()) * mpmath.sqrt(2 - tau)
  x = xc - (y_approx - ymin) * mpmath.tan(theta)

  print(f"X Real: {x}")
  xx = approx_real(x, m)
  print(f"X approx : {xx.evaluate()}")
  part1 = xx
  part2 = yy.to_cycl() * (ZOmega.Omega() + ZOmega.Omega_(4))
  result = part1.to_cycl() + part2

  return result, ymin, ymax, xc, yy, xx, x, xmax, m


def plot_random_samples(theta, epsilon, r, num_samples=200):
  # Get values from your debug sampler
  res, ymin, ymax, xc, yy, xx, x, xmax, m = RANDOM_SAMPLE_DEBUG(theta, epsilon, r)

  # Ensure all mpmath numbers are converted to floats
  ymin_f = float(ymin)
  ymax_f = float(ymax)
  xmax_f = float(xmax)
  xc_f = float(xc)
  x_f = float(x)
  yy_val = float(yy.evaluate())

  # Compute other geometry values
  xleft = 2 * xc_f - xmax_f
  xmin = xmax - (ymax - ymin) * mpmath.tan(theta)
  xmin_f = float(xmin)
  topleft = xmin - (xmax - xleft)
  topleft_f = float(topleft)

  # Compute constants
  TAU = (mpmath.sqrt(5) - 1) / 2
  PHI = TAU + 1
  radius = float(r * PHI**m)

  # Print debug information
  print("Diagonal length:", float(mpmath.sqrt((xmax_f - xmin_f) ** 2 + (ymax_f - ymin_f) ** 2)))
  print("Radius =", radius)
  print("Distance (xmin, ymax):", float(mpmath.sqrt(ymax**2 + xmin**2)))
  print("Width =", float(0.5 * r * epsilon**2 * PHI**m))
  print("Height =", float(r * epsilon * PHI**m))
  print("XMin: ", xmin_f)
  print("YMax: ", ymax_f)

  radius = float(r * PHI**m)

  plt.figure(figsize=(6, 6))
  # Scatter plots
  plt.scatter([xleft, xmax_f, xc_f], [ymin_f] * 3, color="red", alpha=1)
  plt.scatter([xmin_f, topleft_f], [ymax_f, ymax_f], color="blue", alpha=1)
  plt.scatter(x_f, yy_val, color="green", label="Sample")
  for i in range(1, num_samples):
    res, ymin_, ymax_, xc_, yy_i, xx, x_i, xmax_, m_ = RANDOM_SAMPLE_DEBUG(theta, epsilon, r)
    x_i_f = float(x_i)
    yy_i_f = float(yy_i.evaluate()) * mpmath.sqrt(2 - TAU)
    plt.scatter(x_i_f, yy_i_f, color="green", s=10, alpha=0.5)
  # Set limits explicitly (adjust padding as needed)
  x_min_view = xmin_f - 1
  x_max_view = xmax_f + 1
  y_min_view = ymin_f - 1
  y_max_view = ymax_f + 1
  plt.xlim(x_min_view, x_max_view)
  plt.ylim(y_min_view, y_max_view)
  # Add the circle with center at (xc_f, some_y), clipped by axes limits automatically
  circle_center = (0, 0)
  circle = Circle(circle_center, radius, fill=False, color="orange", linestyle="--", linewidth=2, label="Radius circle")
  ax = plt.gca()
  ax.add_patch(circle)
  # Set aspect and labels
  ax.set_aspect("equal", adjustable="box")
  plt.grid(True)
  plt.xlabel("Re")
  plt.ylabel("Im")
  plt.title("ε-region and Sample in Complex Plane")
  plt.legend()
  plt.tight_layout()
  plt.show()

#mpmath.mp.dps = 100
#print("MPMATH accuracy: ", mpmath.mp.dps)

# xx = RANDOM_SAMPLE_DEBUG(1/10, 1e-10, 1.0)
# plot_random_samples(mpmath.pi / 100, 1e-3, 2.0, 100)
