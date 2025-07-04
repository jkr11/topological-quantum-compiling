from rings import Cyclotomic10, RealCyclotomic10
import mpmath
from mpmath import mp
import random
from numberTheory import RANDOM_SAMPLE, EASY_FACTOR, EASY_SOLVABLE, solve_norm_equation, N_i
from typing import List, Union
from exactUnitary import ExactUnitary
from synthesis import Gate, exact_synthesize


def APPROX_REAL(x, n):
  mp.dps = int(n * 1.6)
  TAU = (mpmath.sqrt(5) - 1) / 2
  PHI = TAU + 1
  x = mpmath.mpf(x)
  tau = mpmath.mpf(TAU)
  phi = mpmath.mpf(PHI)
  PREC = tau ** (n - 1) * (1 - tau**n)

  p = mpmath.fibonacci(n)
  q = mpmath.fibonacci(n + 1)
  u = ((-1) ** (n + 1)) * p
  v = ((-1) ** n) * mpmath.fibonacci(n - 1)

  assert u * p + v * q == 1, f"Failed: u * p + v * q = {u * p + v * q}, expected 1"

  c = mpmath.nint(x * q)
  cuq = c * u / q
  a = c * v + p * mpmath.nint(cuq)
  b = c * u - q * mpmath.nint(cuq)

  approx = a + b * tau
  abs_diff = mpmath.fabs(x - approx)
  abs_b = mpmath.fabs(b)
  phi_pow_n = phi**n
  assert (abs_diff <= PREC) and (abs_b <= phi_pow_n), f"Failed: abs(x - approx)={abs_diff} (PREC={PREC}), abs(b)={abs_b} (PHI^n={phi_pow_n})"
  return RealCyclotomic10(int(a), int(b))


def RANDOM_SAMPLE(theta, epsilon, r):
  """
  Implements the RANDOM-SAMPLE algorithm with arbitrary precision using mpmath.
  Returns a Cyclotomic10MP element.
  """
  TAU = (mpmath.sqrt(5) - 1) / 2
  PHI = TAU + 1
  tau = mpmath.mpf(TAU)
  phi = mpmath.mpf(PHI)
  theta = mpmath.mpf(theta)
  epsilon = mpmath.mpf(epsilon)
  r = mpmath.mpf(r)
  mp.dps = max(50, int(-mpmath.log10(epsilon)) + 10)

  C = mpmath.sqrt(phi / (4 * r))
  m = int(mpmath.ceil(mpmath.log(C * epsilon * r, tau))) + 1
  N = int(mpmath.ceil(phi**m))

  sin_theta = mpmath.sin(theta)
  cos_theta = mpmath.cos(theta)
  sqrt_expr = mpmath.sqrt(4 - epsilon**2)

  ymin = r * phi**m * (sin_theta - epsilon * (sqrt_expr * cos_theta + epsilon * sin_theta) / 2)
  ymax = r * phi**m * (sin_theta + epsilon * (sqrt_expr * cos_theta - epsilon * sin_theta) / 2)

  xmax = r * phi**m * ((1 - epsilon**2 / 2) * cos_theta - epsilon * mpmath.sqrt(1 - epsilon**2 / 4) * sin_theta)
  xc = xmax - (r * epsilon**2 * phi**m) / (4 * cos_theta)

  # Random sampling
  j = random.randint(1, N - 1)
  y = ymin + j * (ymax - ymin) / N

  # Step 11: Approximate y'
  y_prime = y / mpmath.sqrt(2 - tau)
  yy = APPROX_REAL(y_prime, m)

  # Step 12: x calculation
  y_approx = (yy.a + yy.b * tau) * mpmath.sqrt(2 - tau)
  x = xc - (y_approx - ymin) * mpmath.tan(theta)

  # Step 13: Approximate x
  xx = APPROX_REAL(x, m)

  # Step 14: Final return
  part1 = xx
  part2 = yy * RealCyclotomic10(2, -1)
  result = part1 + part2
  # Convert to Cyclotomic10MP if needed (here we just return the RealCyclotomic10MP sum)
  return result.to_cycl()


import math


def synthesize_z_rotation(phi: float, eps: float) -> List[Gate]:
  # Use mpmath for all numerics
  phi = mpmath.mpf(phi)
  eps = mpmath.mpf(eps)
  PHI = (mpmath.sqrt(5) + 1) / 2
  TAU = (mpmath.sqrt(5) - 1) / 2

  C = mpmath.sqrt(PHI / 4)
  print("C: ", C)
  print("C * eps: ", C * eps)
  print("log(C * eps): ", mpmath.log(C * eps, TAU))
  m = int(mpmath.ceil(mpmath.log(C * eps, TAU))) + 1
  print("TAU ** m:", PHI**m)
  print("m: ", m)

  theta = 0
  for k in range(10):
    theta_candidate = -phi / 2 - math.pi * (k / 5)
    if 0 <= theta_candidate <= math.pi / 5:
      theta = theta_candidate
      break
  assert 0 <= theta <= math.pi / 5, "Theta out of bounds: " + str(theta)

  u = Cyclotomic10.Zero()
  v = Cyclotomic10.Zero()
  not_found = True
  while not_found:
    u0 = RANDOM_SAMPLE(theta, eps, 1.0)
    print("No solution found, u0:", u0)
    phi_m = RealCyclotomic10.Phi() ** m
    xi = RealCyclotomic10.Phi() * ((RealCyclotomic10.Phi() ** (2 * m)) - N_i(u0))

    fl = EASY_FACTOR(xi)
    print("Factorization of xi:", fl)
    if EASY_SOLVABLE(fl):
      print("FOUND SOLUTION")
      not_found = False
      u = Cyclotomic10.Omega_(k) * (Cyclotomic10.Tau() ** m) * u0
      ne = solve_norm_equation(xi)
      v = (Cyclotomic10.Tau() ** m) * ne
  print("u:", u)
  print("v:", v)
  C = exact_synthesize(ExactUnitary(u, v, 0))
  return C


def synthesize_zx_rotation(phi: float, eps: float) -> List[Gate]:
  """
  approximating Rz (φ)X
  by an 〈F, T 〉-circuit with O(log(1/ε)) gates and
  precision at most ε. Runtime is probabilistic polynomial
  as a function of log(1/ε).
  """
  phi = mpmath.mpf(phi)
  eps = mpmath.mpf(eps)
  PHI = (mpmath.sqrt(5) + 1) / 2
  TAU = (mpmath.sqrt(5) - 1) / 2
  r = mpmath.sqrt(PHI)
  C = mpmath.sqrt(PHI / (4 * r))
  m = int(mpmath.ceil(mpmath.log(C * eps * r, TAU))) + 1
  theta = 0
  for k in range(10):
    theta_candidate = phi / 2 + math.pi / 2 - math.pi * (k / 5)
    if 0 <= theta_candidate <= math.pi / 5:
      theta = theta_candidate
      break
  assert 0 <= theta <= math.pi / 5, "Theta out of bounds: " + str(theta)
  u = 0
  v = 0
  not_found = True
  while not_found:
    u0 = RANDOM_SAMPLE(theta, eps, 1.0)
    xi = (RealCyclotomic10.Phi() ** (2 * m)) - (RealCyclotomic10.Tau() * N_i(u0))
    fl = EASY_FACTOR(xi)
    print("Factorization of xi:", fl)
    if EASY_SOLVABLE(fl):
      not_found = False
      u = Cyclotomic10.Omega_(k) * (Cyclotomic10.Tau() ** m) * u0
      ne = solve_norm_equation(xi)
      v = (Cyclotomic10.Tau() ** m) * ne
  print("u:", u)
  print("v:", v)
  C = exact_synthesize(ExactUnitary(u, v, 0))
  return C


if __name__ == "__main__":
  n = 150
  mp.dps = int(n)
#
  #real_test = mp.pi()  # Example real number to approximate
#
  #c = APPROX_REAL(real_test, n)
  #print("APPROX_REAL:", c.a + c.b * (mpmath.sqrt(5) - 1) / 2)
#
  ## Example usage of RANDOM_SAMPLE
  #theta = 1 / 10
  #epsilon = 1e-5
  #r = 1
  #sample = RANDOM_SAMPLE(theta, epsilon, r)
  #print("RANDOM_SAMPLE result:", sample)
  #mpmath.mp.dps = 100
  #gates = synthesize_z_rotation(2 * (mpmath.pi() / 10**3) * 2, 1e-7)
  #print(gates)
  #from synthesis import evaluate_gate_sequence, rz
  #from util import trace_norm
#
  #U = evaluate_gate_sequence(gates)
  #print("Evaluated Unitary:", U)
  #print("U.u:", U.u)
  #print("U.v:", U.v)
  #print("U.k:", U.k)
  #print("U.to_numpy:", U.to_numpy)
  #print("U real: ", rz(1 / 10))
  input = 2 * math.pi / (4 * 10**3)
  epsilon = 1e-2
  gates = synthesize_zx_rotation(input, epsilon)
  print(f"ZX-rotation circuit for φ = {input}:")
  print(f"Number of gates: {len(gates)}")
  print(f"Gate sequence: {gates}")
  eval_unitary = evaluate_gate_sequence(gates)
  eval_matrix = eval_unitary.to_numpy
  actual_matrix = rz(input) @ X()
  print("Evaluating gates: ", eval_unitary)
  print("As numpy matrix:\n", eval_matrix)
  print("Actual zx matrix:\n", actual_matrix)
