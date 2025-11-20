import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import mpmath
from exact_synthesis.prec import approx_real
from exact_synthesis.rings import ZOmega
import random


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
  m = int(mpmath.ceil(mpmath.log(C * epsilon * r, tau), prec=0)) + 1

  N = int(mpmath.ceil(phi**m))

  sin_theta = mpmath.sin(theta)
  cos_theta = mpmath.cos(theta)
  sqrt_expr = mpmath.sqrt(4 - epsilon**2)
  rpm = r * phi**m

  ymin = rpm * (sin_theta - epsilon * (sqrt_expr * cos_theta + epsilon * sin_theta) / 2)
  ymax = rpm * (sin_theta + epsilon * (sqrt_expr * cos_theta - epsilon * sin_theta) / 2)
  # print(f"Radius : ", rpm)
  # print("Norm at this point: ", rpm * abs(sin_theta + epsilon * (sqrt_expr * cos_theta - epsilon * sin_theta) / 2))

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
  x = xc - (y - ymin) * sin_theta / cos_theta

  xx = approx_real(x, m)

  # assert abs(xx.evaluate() - (xc - (y_approx - ymin) * sin_theta)) < (PHI**m * r * epsilon**2) / 4, f"was {xx.evaluate() - xc - (y_approx - ymin) * sin_theta} shoulde be {(phi**m * r * epsilon**2 / 4)}"

  part1 = xx
  part2 = yy.to_cycl() * (ZOmega.Omega() + ZOmega.Omega_(4))
  result = part1.to_cycl() + part2

  return result, ymin, ymax, xc, yy, xx, x, xmax, m


def plot_epsilon_region(theta, epsilon, r, num_samples=200):
  # Get values from your debug sampler
  res, ymin, ymax, xc, yy, xx, x, xmax, m = RANDOM_SAMPLE_DEBUG(theta, epsilon, r)

  # Ensure all mpmath numbers are converted to floats
  ymin_f = float(ymin)
  ymax_f = float(ymax)
  xmax_f = float(xmax)
  xc_f = float(xc)
  x_f = float(x)
  yy_val = float(yy.evaluate())

  xleft = 2 * xc_f - xmax_f
  xmin = xmax - (ymax - ymin) * mpmath.tan(theta)
  xmin_f = float(xmin)
  topleft = xmin - (xmax - xleft)
  topleft_f = float(topleft)

  TAU = (mpmath.sqrt(5) - 1) / 2
  PHI = TAU + 1
  radius = float(r * PHI**m)

  print("Diagonal length:", float(mpmath.sqrt((xmax_f - xmin_f) ** 2 + (ymax_f - ymin_f) ** 2)))
  print("Radius =", radius)
  print("Distance (xmin, ymax):", float(mpmath.sqrt(ymax**2 + xmin**2)))
  print("Width =", float(0.5 * r * epsilon**2 * PHI**m))
  print("Height =", float(r * epsilon * PHI**m))
  print("XMin: ", xmin_f)
  print("YMax: ", ymax_f)

  print("Distance long side : ", float(mpmath.sqrt((ymax - ymin) ** 2 + (xmax - xmin) ** 2)))
  print("Theoretical distance: ", float(r * epsilon * PHI**m))
  radius = float(r * PHI**m)

  plt.figure(figsize=(6, 6))
  # Scatter plots
  # plt.scatter([xleft, xmax_f, xc_f], [ymin_f] * 3, color="red", alpha=1)
  # plt.scatter([xmin_f, xmax_f], [ymax_f, ymin_f], color="blue", alpha=1, label="ymax")
  # plt.scatter(x_f, yy_val, color="green", label="Sample")
  # plt.plot([xmin_f, topleft_f], [ymax_f, ymax_f], color="blue")
  print("Top line: ", (xmin_f - topleft_f) * mpmath.sin(theta) < r * epsilon**2 * PHI**m / 2)
  for i in range(1, num_samples):
    res, ymin_, ymax_, xc_, yy_i, xx, x_i, xmax_, m_ = RANDOM_SAMPLE_DEBUG(theta, epsilon, r)
    x_i_f = float(x_i)
    yy_i_f = float(yy_i.evaluate()) * mpmath.sqrt(2 - TAU)
    plt.scatter(x_i_f, yy_i_f, color="green", s=10, alpha=1, label="sample")
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


import mpmath


def is_in_parallelogram(px, py, ax, ay, bx, by, cx, cy):
  """
  Check if point (px, py) lies in the parallelogram spanned from point A
  along vectors AB and AC.
  """
  # Vectors
  abx = bx - ax
  aby = by - ay
  acx = cx - ax
  acy = cy - ay
  apx = px - ax
  apy = py - ay

  # Solve for (u, v) in: AP = u * AB + v * AC
  # Using 2x2 linear system:
  # [abx acx] [u] = [apx]
  # [aby acy] [v]   [apy]
  det = abx * acy - aby * acx
  if abs(det) < 1e-20:
    return False  # Degenerate case

  inv_det = 1 / det
  u = (apx * acy - apy * acx) * inv_det
  v = (abx * apy - aby * apx) * inv_det

  return 0 <= u <= 1 and 0 <= v <= 1


def check_samples_in_parallelogram(theta, epsilon, r, num_samples=200):
  from mpmath import mp, tan, sqrt

  TAU = (sqrt(5) - 1) / 2
  PHI = TAU + 1
  theta = mp.mpf(theta)
  epsilon = mp.mpf(epsilon)
  r = mp.mpf(r)

  res, ymin, ymax, xc, yy, xx, x, xmax, m = RANDOM_SAMPLE_DEBUG(theta, epsilon, r)

  ymin = mp.mpf(ymin)
  ymax = mp.mpf(ymax)
  xmax = mp.mpf(xmax)
  xc = mp.mpf(xc)
  x = mp.mpf(x)
  xx = mp.mpf(xx.evaluate())

  xleft = 2 * xc - xmax
  xmin = xmax - (ymax - ymin) * tan(theta)

  ax, ay = xmax, ymin  # A
  bx, by = xleft, ymin  # B
  cx, cy = xmin, ymax  # C

  violations = []

  for _ in range(num_samples):
    _, _, _, _, yy_i, _, x_i, _, _ = RANDOM_SAMPLE_DEBUG(theta, epsilon, r)
    px = mp.mpf(x_i)
    py = mp.mpf(yy_i.evaluate()) * sqrt(2 - TAU)

    if not is_in_parallelogram(px, py, ax, ay, bx, by, cx, cy):
      violations.append((float(px), float(py)))

  if violations:
    print(f"{len(violations)} samples out of {num_samples} were outside the parallelogram.")
    for px, py in violations[:5]: 
      print(f"Outlier at (x={px}, y={py})")
  else:
    print(f"All {num_samples} samples were within the parallelogram.")

  return violations


def test_point_on_circle(theta, epsilon, r):
  TAU = (mpmath.sqrt(5) - 1) / 2
  PHI = TAU + 1
  tau = mpmath.mpf(TAU)
  phi = mpmath.mpf(PHI)
  theta = mpmath.mpf(theta)
  epsilon = mpmath.mpf(epsilon)
  r = mpmath.mpf(r)

  C = mpmath.sqrt(phi / (4 * r))
  m = int(mpmath.ceil(mpmath.log(C * epsilon * r) / mpmath.log(tau))) + 1
  R = r * phi**m

  sin_theta = mpmath.sin(theta)
  cos_theta = mpmath.cos(theta)
  sqrt_expr = mpmath.sqrt(4 - epsilon**2)

  xmax = R * ((1 - epsilon**2 / 2) * cos_theta - epsilon * mpmath.sqrt(1 - epsilon**2 / 4) * sin_theta)
  ymin = R * (sin_theta - epsilon * (sqrt_expr * cos_theta + epsilon * sin_theta) / 2)

  norm_squared = xmax**2 + ymin**2
  return norm_squared / R**2  # should be 1 if it's on the circle


# Try it
mpmath.mp.dps = 400

theta = mpmath.pi / 10
epsilon = 1e-21
r = 1
check_samples_in_parallelogram(theta, epsilon, r)
