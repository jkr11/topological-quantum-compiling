from dataclasses import dataclass

import mpmath

from exact_synthesis.exactUnitary import ExactUnitary
from exact_synthesis.rings import ZOmega, ZTau, N_i
from exact_synthesis.numberTheory import easy_factor, easy_solvable, solve_norm_equation
from exact_synthesis.prec import random_sample


@dataclass(frozen=True)
class TGate:
  power: int  # modulo 10


@dataclass(frozen=True)
class FGate:
  power: int


@dataclass(frozen=True)
class WIGate:
  power: int  # modulo 10


Gate = TGate | FGate | WIGate

_T_power_table = [ExactUnitary.T() ** j for j in range(11)]
_FT_power_table = [ExactUnitary.F() * _T_power_table[j] for j in range(11)]
_omega_k_T_j_table = {ExactUnitary.I().omega_mul(k) * _T_power_table[j]: (k, j) for k in range(11) for j in range(11)}


def exact_synthesize(U: ExactUnitary) -> list[Gate]:
  g = U.gauss_complexity()
  Ur = U
  C: list[Gate] = []

  while g > 2:
    min_complexity = None
    J = None
    for j in range(1, 11):  # j in {1, ..., 10}, per Fig. 4 line 4
      gg = (_FT_power_table[j] * Ur).gauss_complexity()
      if min_complexity is None or gg < min_complexity:
        min_complexity = gg
        J = j
    assert J is not None, "J was None"
    Ur = _FT_power_table[J] * Ur
    g = Ur.gauss_complexity()
    C.insert(0, TGate((10 - J) % 10))
    C.insert(0, FGate(1))

  if Ur not in _omega_k_T_j_table:
    raise ValueError("Final reduction failed: Ur is not of the form omega^k T^j")

  k, j = _omega_k_T_j_table[Ur]
  if j != 0:
    C.insert(0, TGate(j))
  if k != 0:
    C.insert(0, WIGate(k))
  return C


def _find_theta_k(base_angle, k_range=range(-11, 11)):
  pi_over_5 = mpmath.pi / 5
  for k in k_range:
    theta_candidate = base_angle - mpmath.pi * (mpmath.mpf(k) / 5)
    if 0 <= theta_candidate < pi_over_5:
      return theta_candidate, k
  raise ValueError(f"Failed to find suitable k for base angle {base_angle}")


def synthesize_z_rotation(phi, epsilon) -> ExactUnitary:
  phi = mpmath.mpf(phi)
  eps = mpmath.mpf(epsilon)
  TAU = (mpmath.sqrt(5) - 1) / 2
  PHI = TAU + 1

  C = mpmath.sqrt(PHI / 4)
  m = int(mpmath.ceil(mpmath.log(C * eps, TAU))) + 1

  theta, k = _find_theta_k(-phi / 2)

  while True:
    u0 = random_sample(theta, eps, 1)
    xi = ZTau.Phi() * ((ZTau.Phi() ** (2 * m)) - N_i(u0))
    fl = easy_factor(xi)
    if easy_solvable(fl):
      u = (ZOmega.Omega() ** (k % 10)) * (ZOmega.Tau() ** m) * u0
      v = (ZOmega.Tau() ** m) * solve_norm_equation(xi)
      return ExactUnitary(u, v, 0)


def synthesize_zx_rotation(phi, epsilon) -> ExactUnitary:
  phi = mpmath.mpf(phi)
  eps = mpmath.mpf(epsilon)
  TAU = (mpmath.sqrt(5) - 1) / 2
  PHI = TAU + 1
  r = mpmath.sqrt(PHI)

  C = mpmath.sqrt(PHI / (4 * r))
  m = int(mpmath.ceil(mpmath.log(C * eps * r, TAU))) + 1

  theta, k = _find_theta_k(phi / 2 + mpmath.pi / 2)

  while True:
    u0 = random_sample(theta, eps, r)
    xi = (ZTau.Phi() ** (2 * m)) - ZTau.Tau() * N_i(u0)
    fl = easy_factor(xi)
    if easy_solvable(fl):
      v = (ZOmega.Omega() ** (k % 10)) * (ZOmega.Tau() ** m) * u0
      u = (ZOmega.Tau() ** m) * solve_norm_equation(xi)
      return ExactUnitary(u, v, 0)


def evaluate_gate_sequence(gates: list[Gate]) -> ExactUnitary:
  U = ExactUnitary.I()
  for gate in gates:
    if isinstance(gate, TGate):
      U = U * ExactUnitary.T() ** gate.power
    elif isinstance(gate, FGate):
      U = U * ExactUnitary.F()
    elif isinstance(gate, WIGate):
      U = U * ExactUnitary.I().omega_mul(gate.power)
    else:
      raise ValueError(f"Unknown gate type: {gate}")
  return U


def d_z(phi, U: ExactUnitary):
  phi = mpmath.mpf(phi)
  z = mpmath.mpc(U.u.evaluate()) * mpmath.exp(1j * phi / 2)
  x = mpmath.fabs(mpmath.re(z))
  if x > 0.999999:
    return mpmath.sqrt(2 * (1 - x))  # Do we have to?
  return mpmath.sqrt(1 - x)


def d_zx(phi, U: ExactUnitary):
  phi = mpmath.mpf(phi)
  TAU = (mpmath.sqrt(5) - 1) / 2
  z = U.v.evaluate() * mpmath.exp(-1j * (phi / 2 + mpmath.pi / 2))
  return mpmath.sqrt(1 - mpmath.sqrt(TAU) * abs(z.real))
