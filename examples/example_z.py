import mpmath
from exact_synthesis.synthesis import ExactFibonacciSynthesizer, d_z
from exact_synthesis.util import Gates, trace_norm
from exact_synthesis.exactUnitary import ExactUnitary

from dataclasses import dataclass
from typing import List, Union
import math
import mpmath
import numpy as np

from exact_synthesis.exactUnitary import ExactUnitary
from exact_synthesis.rings import ZOmega, ZTau, N_i
from exact_synthesis.numberTheory import easy_factor, easy_solvable, solve_norm_equation
from exact_synthesis.prec import random_sample
from exact_synthesis.util import euler_angles


@dataclass(frozen=True)
class TGate:
  power: int  # modulo 10


@dataclass(frozen=True)
class FGate:
  power: int


@dataclass(frozen=True)
class WIGate:
  power: int  # modulo 10


@dataclass(frozen=True)
class Sigma1:
  power: int


@dataclass(frozen=True)
class Sigma2:
  power: int


Gate = Union[TGate, FGate, WIGate, Sigma1, Sigma2]

T_power_table = [ExactUnitary.T() ** j for j in range(11)]
FT_power_table = [ExactUnitary.F() * T_power_table[j] for j in range(11)]
omega_k_T_j_table = {}
for k in range(11):
  for j in range(11):
    omega_k_T_j_table[ExactUnitary.I().omega_mul(k) * T_power_table[j]] = (k, j)


def G(u):
  return u.gauss_complexity()


def exact_synthesize(U: ExactUnitary) -> List[Gate]:
  # NOTE: indentation in the pasted version of this function was mangled
  # (the `return C` / `raise ValueError` were flattened to the same level
  # as the `if Ur in omega_k_T_j_table:` check, which would make the
  # lookup unconditionally return before ever raising). Restored the
  # evidently-intended nesting: only return once Ur is found in the
  # table; raise otherwise.
  g = G(U)
  Ur = U
  C: List[Gate] = []
  while g > 2:
    min_complexity = math.inf
    J = 0
    for j in range(1, 11):
      candidate = FT_power_table[j] * Ur
      gg = candidate.gauss_complexity()
      if gg < min_complexity:
        min_complexity = gg
        J = j
    Ur = FT_power_table[J] * Ur
    g = Ur.gauss_complexity()
    C.insert(0, TGate((10 - J) % 10))
    C.insert(0, FGate(1))
  if Ur in omega_k_T_j_table:
    k, j = omega_k_T_j_table[Ur]
    if j != 0:
      C.insert(0, TGate(j))
    if k != 0:
      C.insert(0, WIGate(k))
    return C
  raise ValueError("Final reduction failed")


def synthesize_z_rotation(phi, epsilon, max_attempts: int = 100_000) -> ExactUnitary:
  phi = mpmath.mpf(phi)
  epsilon = mpmath.mpf(epsilon)

  tau = (mpmath.sqrt(5) - 1) / 2  # the golden ratio, NOT the input angle
  golden_phi = tau + 1

  Cconst = mpmath.sqrt(golden_phi / 4)
  m = int(mpmath.ceil(mpmath.log(Cconst * epsilon) / mpmath.log(tau))) + 1

  # step = mpmath.pi / 5
  # x = -phi / 2
  # k = int(mpmath.floor(x / step))
  # theta = x - k * step

  k = int(mpmath.floor(-5 * phi / (2 * mpmath.pi)))

  theta = -phi / 2 - (mpmath.pi * k) / 5
  assert 0 <= theta <= mpmath.pi / 5, f"Invalid theta: {theta}"

  tau_m = (ZTau.Tau() ** m).to_cycl()  # tau^m, as a ZOmega
  omega_k = ZOmega.Omega_(k % 10)  # omega^k
  target_norm: ZTau = ZTau.Phi() ** (2 * m)  # golden_phi^(2m), exact in Z[tau]

  u = v = None
  for _ in range(max_attempts):
    u0: ZOmega = random_sample(theta, epsilon, 1)  # Fig. 13, r=1
    xi = ZTau.Phi() * (target_norm - N_i(u0))  # exact in Z[tau]
    fl = easy_factor(xi)
    if easy_solvable(fl):
      u = omega_k * tau_m * u0
      v = tau_m * solve_norm_equation(xi)
      break
  else:
    raise RuntimeError(f"synthesize_z_rotation: no solvable norm equation found in {max_attempts} RANDOM-SAMPLE attempts (phi={phi}, epsilon={epsilon}).")

  target = ExactUnitary(u, v, 0)
  # return exact_synthesize(target)
  return target


def dist(U, phi):
  u_val = U.u.evaluate()
  phi = mpmath.mpf(phi)
  rotated = u_val * mpmath.exp(mpmath.mpc(0, phi / 2))
  return mpmath.sqrt(1 - abs(mpmath.re(rotated)))

def _dist(u, phi):
  phi = mpmath.mpf(phi)
  rot = u.evaluate() * mpmath.exp(mpmath.mpc(0, phi / 2))
  return mpmath.sqrt(1 - abs(mpmath.re(rot)))

def rz_matrix(phi) -> mpmath.matrix:
  """Returns R_z(phi) as a 2x2 mpmath matrix."""
  phi_val = mpmath.mpf(phi)
  return mpmath.matrix([
    [mpmath.exp(mpmath.mpc(0, -phi_val / 2)), 0],
    [0, mpmath.exp(mpmath.mpc(0, phi_val / 2))],
  ])


def dist_trace_norm(U: "ExactUnitary", phi) -> mpmath.mpf:
  """Computes the trace-norm operator distance:

  d(R_z(phi), U) = sqrt(1 - (1/2) * |Tr(R_z(phi)^dagger * U)|)
  """
  # 1. Get exact matrix representations
  Rz = rz_matrix(phi)
  U_mat = U.to_matrix()

  # 2. R_z^\dagger is conjugate transpose of R_z
  Rz_dagger = Rz.H

  # 3. Compute matrix product diagonal sum (Trace)
  prod = Rz_dagger * U_mat
  trace_val = prod[0, 0] + prod[1, 1]

  # 4. Operator distance formula
  fidelity = abs(trace_val) / 2
  return mpmath.sqrt(mpmath.mpf(1) - fidelity)


def other_test(phi):
  u = random_sample(phi, 1e-10)
  norm = _dist(u, phi)
  print(norm)

if __name__ == "__main__":
  mpmath.mp.dps = 400
  phi = mpmath.pi / 10
  epsilon = mpmath.mpf(1e-80)

  other_test(phi)
  exit(0)
  while True:
    try:
      g = synthesize_z_rotation(phi, epsilon) * ExactUnitary.T() ** 5
      print("d(R_z, U[u,v,0] =", dist(g, phi))
      print("Trace Norm =", dist_trace_norm(g, phi))
      print(epsilon >= dist(g, phi))
      print(f"Gate sequence: {len(ExactFibonacciSynthesizer._exact_synthesize(g))}")
      actual = rz_matrix(phi)
      approx = (g).to_matrix()
      break
    except ValueError as e:
      print(f"Caught error: {e} -- retrying ...")
