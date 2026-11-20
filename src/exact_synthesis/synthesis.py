from dataclasses import dataclass

import mpmath

from exact_synthesis.exactUnitary import ExactUnitary
from exact_synthesis.rings import ZOmega, ZTau, N_i
from exact_synthesis.numberTheory import easy_factor, easy_solvable, solve_norm_equation
from exact_synthesis.prec import random_sample
from exact_synthesis.util import d_z, d_trace

import numpy as np  # or get rid of numpy instead
import cmath  # get rid of this for phase? but its not an extra dep so w/e.


@dataclass(frozen=True)
class TGate:
  power: int  # modulo 10


@dataclass(frozen=True)
class FGate:
  power: int = 0


@dataclass(frozen=True)
class WIGate:
  power: int  # modulo 10


Gate = TGate | FGate | WIGate

_T_power_table = [ExactUnitary.T() ** j for j in range(11)]
_FT_power_table = [ExactUnitary.F() * _T_power_table[j] for j in range(11)]
_omega_k_T_j_table = {ExactUnitary.I().omega_mul(k) * _T_power_table[j]: (k, j) for k in range(11) for j in range(11)}


def exact_synthesize(U: ExactUnitary) -> list[Gate]:
  Ur = U
  gates: list[Gate] = []

  while (g := Ur.gauss_complexity()) > 2:
    J, Ur = min(
      ((j, _FT_power_table[j] * Ur) for j in range(1, 11)),
      key=lambda c: c[1].gauss_complexity(),
    )
    if Ur.gauss_complexity() >= g:
      raise ValueError(f"Complexity is not decreasing: {g}")
    gates += [TGate((10 - J) % 10), FGate(1)]

  k, j = _omega_k_T_j_table[Ur]
  if j:
    gates.append(TGate(j))
  if k:
    gates.append(WIGate(k))
  return gates


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
  m = int(mpmath.ceil(mpmath.log(C * eps) / mpmath.log(TAU))) + 1

  theta, k = _find_theta_k(-phi / 2)

  while True:
    u0 = random_sample(theta, eps, 1)
    xi = ZTau.Phi() * ((ZTau.Phi() ** (2 * m)) - N_i(u0))
    fl = easy_factor(xi)
    if easy_solvable(fl):
      u = (ZOmega.Omega() ** (k % 10)) * (ZOmega.Tau() ** m) * u0
      v = (ZOmega.Tau() ** m) * solve_norm_equation(xi)
      res = ExactUnitary(u, v, 5)
      assert d_z(res, phi) <= epsilon
      return res  # return ExactUnitary(u, v, 0)


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


def decompose(U: np.ndarray):
  # global phase
  det = np.linalg.det(U)
  global_phase = cmath.phase(det) / 2
  U_prime = U * np.exp(-1j * global_phase)

  u00 = U_prime[0, 0]
  u01 = U_prime[0, 1]

  if np.isclose(np.abs(u00), 0.0):
    # U = e^{i \delta_{alt}} R_z(\phi) X
    p1 = cmath.phase(U[0, 1])
    p2 = cmath.phase(U[1, 0])
    return {"type": "X", "delta": (p1 + p2) / 2, "phi": p2 - p1}

  tau = (np.sqrt(5) - 1) / 2

  if np.isclose(np.abs(u01), 0.0):
    return {"type": "Z", "delta": global_phase, "alpha": -2 * cmath.phase(u00), "beta": 0.0, "gamma": 0.0}

  cos_beta = np.clip(1.0 - (np.abs(u01) ** 2) / (2 * tau**3), -1.0, 1.0)
  beta = np.arccos(cos_beta)

  M00 = tau**2 * np.exp(-1j * beta / 2) + tau * np.exp(1j * beta / 2)
  M01 = -2j * tau * np.sqrt(tau) * np.sin(beta / 2)

  phase_plus = cmath.phase(M00 / u00)
  phase_minus = cmath.phase(M01 / u01)

  return {"type": "Z", "delta": global_phase, "alpha": phase_plus + phase_minus, "beta": beta, "gamma": phase_plus - phase_minus}


def reconstruct_decomposition(decomp):
  alpha = decomp["alpha"]
  beta = decomp["beta"]
  gamma = decomp["gamma"]

  def Rz(theta):
    return np.diag([np.exp(-1j * theta / 2), np.exp(1j * theta / 2)])

  F = ExactUnitary.F().to_numpy()

  return np.exp(1j * decomp["delta"]) * Rz(alpha) @ F @ Rz(beta) @ F @ Rz(gamma)


def synthesize_unitary(target_matrix: np.ndarray, epsilon) -> ExactUnitary:
  decomp = decompose(target_matrix)
  assert np.allclose(target_matrix, reconstruct_decomposition(decomp))
  if decomp["type"] == "X":
    return synthesize_zx_rotation(decomp["phi"], epsilon)
  elif decomp["type"] == "Z":
    Ralpha = synthesize_z_rotation(decomp["alpha"], epsilon / 3)
    print("alpha: ", decomp["alpha"])
    Rbeta = synthesize_z_rotation(decomp["beta"], epsilon / 3)
    Rgamma = synthesize_z_rotation(decomp["gamma"], epsilon / 3)
    return Ralpha * ExactUnitary.F() * Rbeta * ExactUnitary.F() * Rgamma
  else:
    raise NotImplementedError("Unknown decomposition")
