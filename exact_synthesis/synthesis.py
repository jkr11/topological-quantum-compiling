from dataclasses import dataclass
from typing import List, Union
import math
from exact_synthesis.exactUnitary import ExactUnitary
from exact_synthesis.rings import ZOmega, ZTau, N_i
from exact_synthesis.numberTheory import easy_factor, easy_solvable, solve_norm_equation
from exact_synthesis.prec import random_sample
import numpy as np
import mpmath
from exact_synthesis.util import euler_angles


@dataclass(frozen=True)
class TGate:
  power: int  # modulo 10


@dataclass(frozen=True)
class FGate:
  power: int
  pass


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


class ExactFibonacciSynthesizer:
  @classmethod
  def __G(cls, u: ExactUnitary):
    return u.gauss_complexity()

  @classmethod
  def _synthesize_z_rotation(cls, phi, epsilon) -> ExactUnitary:
    """
    approximates Rz(phi) with O(log(1/eps)) gates and precision at most eps, produces an <F,T> circuit.

    returns:
      the exact unitary U
      the Circuit C decomposing U by exact synthesis
    """
    phi = mpmath.mpf(phi)
    eps = mpmath.mpf(epsilon)
    TAU = (mpmath.sqrt(5) - 1) / 2
    PHI = TAU + 1

    C = mpmath.sqrt(PHI / 4)

    m = int(mpmath.ceil(mpmath.log(C * eps, TAU)) + 1)

    theta = None
    k_final = None

    pi_over_5 = mpmath.pi / 5
    k_candidate = round(-5 * phi / (2 * mpmath.pi))
    for k in range(-11, 11):
      theta_candidate = -phi / 2 - mpmath.pi * (mpmath.mpf(k) / 5)
      if 0 <= theta_candidate < pi_over_5:
        theta = theta_candidate
        k_final = k
        break

    if theta is None:
      raise ValueError("Failed to find suitable k.")
    assert k_final is not None, "k was None"
    assert 0 <= theta <= pi_over_5, "Theta out of bounds: " + str(theta)

    u = ZOmega.Zero()
    v = ZOmega.Zero()
    not_found = True
    k = k_final
    while not_found:
      u0 = random_sample(theta, eps, 1)

      xi = ZTau.Phi() * ((ZTau.Phi() ** (2 * m)) - N_i(u0))

      fl = easy_factor(xi)

      if easy_solvable(fl):
        not_found = False

        u = (ZOmega.Omega() ** (k_final % 10)) * (ZOmega.Tau() ** m) * u0

        v = (ZOmega.Tau() ** m) * solve_norm_equation(xi)

    return ExactUnitary(u, v, 0)

  @classmethod
  def synthesize_zx_rotation(cls, phi, eps) -> ExactUnitary:
    """
    approximating Rz (φ)X
    by an 〈F, T 〉-circuit with O(log(1/ε)) gates and
    precision at most ε. Runtime is probabilistic polynomial
    as a function of log(1/ε).
    """
    phi = mpmath.mpf(phi)
    eps = mpmath.mpf(eps)
    TAU = (mpmath.sqrt(5) - 1) / 2
    PHI = TAU + 1
    r = mpmath.sqrt(PHI)
    C = mpmath.sqrt(PHI / (4 * r))
    m = int(mpmath.ceil(mpmath.log(C * eps * r, TAU))) + 1
    theta = None
    k = None
    for kk in range(-11, 11):
      theta_candidate = phi / 2 + math.pi / 2 - math.pi * (kk / 5)
      if 0 <= theta_candidate <= math.pi / 5:
        theta = theta_candidate
        k = kk
        break

    assert theta is not None, "Theta not found."
    assert 0 <= theta < math.pi / 5, "Theta out of bounds: " + str(theta)
    u = ZOmega.Zero()
    v = ZOmega.Zero()
    not_found = True
    while not_found:
      u0 = random_sample(theta, eps, r)
      xi = (ZTau.Phi() ** (2 * m)) - ZTau.Tau() * N_i(u0)
      fl = easy_factor(xi)

      if easy_solvable(fl):
        not_found = False
        v = ZOmega.Omega_(k) * (ZOmega.Tau() ** m) * u0
        ne = solve_norm_equation(xi)
        u = (ZOmega.Tau() ** m) * ne

    return ExactUnitary(u, v, 0)

  @classmethod
  def synthesize_z_rotation(cls, phi, eps) -> ExactUnitary:
    EU = cls._synthesize_z_rotation(phi, eps)
    return EU

  @classmethod
  def _exact_synthesize(cls, U: ExactUnitary) -> List[Gate]:
    g = cls.__G(U)
    Ur = U
    C: List[Gate] = []
    while g > 2:
      min_complexity = math.inf
      J = 0
      for j in range(11):
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

  @classmethod
  def synthesize_unitary(cls, U: np.ndarray, epsilon: float):
    alpha, beta, gamma, delta = euler_angles(U)
    decomp1 = cls.synthesize_z_rotation(alpha, epsilon)
    # print("D1: ", decomp1.to_numpy())
    # print("Actual: ", Gates.Rz(alpha))
    decomp2 = cls.synthesize_zx_rotation(beta, epsilon)
    # print("D2: ", decomp2.to_numpy())
    # print("Actual2: ", Gates.Rz(beta))
    decomp3 = cls.synthesize_z_rotation(gamma, epsilon)
    # print("D3: ", decomp3.to_numpy)
    # print("Actual 3: ", Gates.Rz(gamma))
    return delta, decomp1 * ExactUnitary.T(5) * decomp2 * decomp3  # TODO: check if this is right


def evaluate_gate_sequence(gates: List[Gate]) -> ExactUnitary:
  U = ExactUnitary.I()
  for gate in gates:
    if isinstance(gate, TGate):
      U = U * ExactUnitary.T() ** gate.power
    elif isinstance(gate, FGate):
      U = U * ExactUnitary.F()
    elif isinstance(gate, WIGate):
      U = U * ExactUnitary.I().omega_mul(gate.power)
    elif isinstance(gate, Sigma1):
      U = U * ExactUnitary.sigma1()
    elif isinstance(gate, Sigma2):
      U = U * ExactUnitary.sigma2()
    else:
      raise ValueError(f"Unknown gate type: {gate}")
  return U


def d_z_(phi, U: ExactUnitary):
  phi = mpmath.mpf(phi)
  return mpmath.sqrt(1 - mpmath.fabs(mpmath.re(U.u.evaluate() * mpmath.exp(1j * phi / 2))))


def _d_z(phi, u: ZOmega):
  phi = mpmath.mpf(phi)
  return mpmath.sqrt(1 - mpmath.fabs(mpmath.re(u.evaluate() * mpmath.exp(1j * phi / 2))))


def d_z(phi, U: ExactUnitary):
  phi = mpmath.mpf(phi)
  z = mpmath.mpc(U.u.evaluate()) * mpmath.exp(1j * phi / 2)
  x = mpmath.fabs(mpmath.re(z))
  if x > 0.999999:
    return mpmath.sqrt(2 * (1 - x))
  return mpmath.sqrt(1 - x)


def d_zx(phi, U: ExactUnitary):
  phi = mpmath.mpf(phi)
  TAU = (mpmath.sqrt(5) - 1) / 2
  return mpmath.sqrt(1 - mpmath.sqrt(TAU) * abs((U.v.evaluate() * mpmath.exp(-1j * (phi / 2 + mpmath.pi / 2))).real))
