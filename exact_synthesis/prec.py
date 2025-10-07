from single_qubit.exact_synthesis.rings import Cyclotomic10, ZTau
import mpmath
import random
from mpmath import exp


def APPROX_REAL(x, n) -> ZTau:
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


def RANDOM_SAMPLE(theta, epsilon, r) -> Cyclotomic10:
  """
  Implements the RANDOM-SAMPLE algorithm with arbitrary precision using mpmath.
  Returns a Cyclotomic10MP element.
  """
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
  yy = APPROX_REAL(y_prime, m)

  y_approx = (yy.a + yy.b * tau) * mpmath.sqrt(2 - tau)
  x = xc - (y_approx - ymin) * mpmath.tan(theta)

  xx = APPROX_REAL(x, m)

  part1 = xx
  part2 = yy.to_cycl() * (Cyclotomic10.Omega() + Cyclotomic10.Omega_(4))
  result = part1.to_cycl() + part2

  return result


# def synthesize_z_rotation(phi: float, eps: float) -> Tuple[List[Gate], ExactUnitary]:
#  """
#  approximates Rz(phi) with O(log(1/eps)) gates and precision at most eps, produces an <F,T> circuit.
#
#  returns:
#    the exact unitary U
#    the Circuit C decomposing U by exact synthesis
#  """
#  phi = mpmath.mpf(phi)
#  eps = mpmath.mpf(eps)
#
#  TAU = (mpmath.sqrt(5) - 1) / 2
#  PHI = TAU + 1
#
#  C = mpmath.sqrt(PHI / 4)
#
#  m = int(mpmath.ceil(mpmath.log(C * eps, TAU)) + 1)
#
#  theta = None
#  k_final = None
#
#  for k in range(-10, 10):  # is solving here faster?
#    theta_candidate = -phi / 2 - math.pi * (k / 5)
#    if 0 <= theta_candidate <= math.pi / 5:
#      theta = theta_candidate
#      k_final = k
#      break
#  assert 0 <= theta <= math.pi / 5, "Theta out of bounds: " + str(theta)
#  if theta is None:
#    raise ValueError("Failed to find suitable k.")
#
#  u = Cyclotomic10.Zero()
#  v = Cyclotomic10.Zero()
#  not_found = True
#  k = k_final
#  while not_found:
#    u0 = RANDOM_SAMPLE(theta, eps, 1)
#
#    xi = ZTau.Phi() * ((ZTau.Phi() ** (2 * m)) - N_i(u0))
#
#    fl = EASY_FACTOR(xi)
#
#    if EASY_SOLVABLE(fl):
#      print("FOUND SOLUTION")
#      not_found = False
#
#      u = Cyclotomic10.Omega_(k) * (Cyclotomic10.Tau() ** (m)) * u0
#
#      v = (Cyclotomic10.Tau() ** (m)) * solve_norm_equation(xi)
#
#  C = exact_synthesize(ExactUnitary(u, v, 0))
#  return C, ExactUnitary(u, v, 0)
#
#
## TODO: all the |u| + tau|v| != 1 but phi? Why?
# def synthesize_zx_rotation(phi: float, eps: float) -> List[Gate]:
#  """
#  approximating Rz (φ)X
#  by an 〈F, T 〉-circuit with O(log(1/ε)) gates and
#  precision at most ε. Runtime is probabilistic polynomial
#  as a function of log(1/ε).
#  """
#  phi = mpmath.mpf(phi)
#  eps = mpmath.mpf(eps)
#  TAU = (mpmath.sqrt(5) - 1) / 2
#  PHI = TAU + 1
#  r = mpmath.sqrt(PHI)
#  C = mpmath.sqrt(PHI / (4 * r))
#  m = int(mpmath.ceil(mpmath.log(C * eps * r, TAU))) + 1
#  theta = None
#  for k in range(-10, 10):
#    theta_candidate = phi / 2 + math.pi / 2 - math.pi * (k / 5)
#    if 0 <= theta_candidate <= math.pi / 5:
#      theta = theta_candidate
#      break
#  assert 0 <= theta <= math.pi / 5, "Theta out of bounds: " + str(theta)
#  print(f"FOund k = {k}")
#  u = Cyclotomic10.Zero()
#  v = Cyclotomic10.Zero()
#  not_found = True
#  while not_found:
#    u0 = RANDOM_SAMPLE(theta, eps, r)
#    xi = (ZTau.Phi() ** (2 * m)) - (ZTau.Tau() * N_i(u0))
#    fl = EASY_FACTOR(xi)
#    if EASY_SOLVABLE(fl):
#      not_found = False
#      tm = Cyclotomic10.Tau() ** m
#      u = Cyclotomic10.Omega_(k) * (tm) * u0
#      v = (tm) * solve_norm_equation(xi)
#  C = exact_synthesize(ExactUnitary(u, v, 0))
#  return C
#
#
def rz_mp(phi):
  return mpmath.matrix([[exp(-1j * phi / 2), 0], [0, exp(1j * phi / 2)]])


# def test_synthesize_zx_rotation():
#  input = mpmath.pi * 0.5
#  epsilon = 1e-2
#  gates = synthesize_zx_rotation(input, epsilon)
#  print(f"ZX-rotation circuit for φ = {input}:")
#  print(f"Number of gates: {len(gates)}")
#  print(f"Gate sequence: {gates}")
#  eval_unitary = evaluate_gate_sequence(gates)
#  eval_matrix = eval_unitary.to_matrix
#  actual_matrix = rz_mp(input) @ X()
#  print("Evaluating gates: ", eval_unitary)
#  print("As numpy matrix:\n", eval_matrix)
#  print("Actual zx matrix:\n", actual_matrix)
#
#
# def run_table_one():
#  prec = 1e-5
#  for k in range(1, 10**3, 100):
#    input = (2 * mpmath.pi * k) / (10**3)
#    print(f"Running for phi = {input}")
#    gates, uni = synthesize_z_rotation(input, prec)
#    print(f"Z-rotation circuit for φ = {input}:")
#    print(f"Number of gates: {len(gates)}")
#    eval_unitary = evaluate_gate_sequence(gates)
#    print(eval_unitary == uni)
#    dist = d(input, eval_unitary.u, eval_unitary.v)
#    print("Dist: ", dist)
#    print("Dist < eps? ", dist < mpmath.mpmathify(prec))
#
#
# def generate_test_values(num_tests=40, start=1.0, factor=0.5):
#  values = []
#  x = start
#  for _ in range(num_tests):
#    values.append(x)
#    x *= factor
#  return values
#
#
# def test_approx_real():
#  n = 400
#  for x in generate_test_values(num_tests=600):
#    print(f"Approx real with  {x}")
#    s = APPROX_REAL(x, n)
#    print(s)
#
#
# def test_synthesize_z_rotation():
#  input = 2 * mpmath.pi / 100
#  epsilon = 1e-70
#  gates, uni = synthesize_z_rotation(input, epsilon)
#  print(f"Z-rotation circuit for φ = {input}:")
#  print(f"Number of gates: {len(gates)}")
#  print(f"Gate sequence: {gates}")
#  eval_unitary = evaluate_gate_sequence(gates)
#  print(eval_unitary == uni)
#  eval_matrix = eval_unitary.to_matrix
#  actual_matrix = rz_mp(input)
#  print("Evaluating gates: ", eval_unitary)
#  print("As numpy matrix:\n", eval_matrix)
#  print("Actual z matrix:\n", actual_matrix)
#  dist = d(input, eval_unitary.u, eval_unitary.v)
#  print("Dist: ", dist)
#  print("Dist < eps? ", dist < mpmath.mpmathify(epsilon))
#  return uni, gates, dist
#
#
# def d(phi, u, v):
#  phi = mpmath.mpf(phi)
#  return mpmath.sqrt(1 - abs((u.evaluate() * mpmath.exp(1j * phi / 2).real)))
#
#
# def test_exact_synthesize():
#  u = Cyclotomic10(1611948661, -4220136381, 4220136381, -1611948661)
#  v = Cyclotomic10(0, 0, 0, 0)
#  k = 0
#  C = exact_synthesize(ExactUnitary(u, v, k))
#  print("Exact synthesis result:", C)
#  eval_unitary = evaluate_gate_sequence(C)
#  print("Evaluated Unitary:", eval_unitary)
#  print("eval unitary is input unitary:", eval_unitary.u == u and eval_unitary.v == v and eval_unitary.k == k)
#
#
# class EpsilonRegion:
#  def __init__(self, theta, epsilon, r=1.0):
#    PHI = (mpmath.sqrt(5) + 1) / 2
#    TAU = (mpmath.sqrt(5) - 1) / 2
#    C = mpmath.sqrt(PHI / (4 * r))
#    m = mpmath.ceil(mpmath.log(C * epsilon * r, TAU)) + 1
#
#
