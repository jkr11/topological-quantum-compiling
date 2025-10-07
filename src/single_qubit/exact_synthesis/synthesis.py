from dataclasses import dataclass
from typing import List, Union
import math
from single_qubit.exact_synthesis.exactUnitary import ExactUnitary
from single_qubit.exact_synthesis.rings import Cyclotomic10, ZTau, N_i
from single_qubit.exact_synthesis.numberTheory import RANDOM_SAMPLE, EASY_FACTOR, EASY_SOLVABLE, solve_norm_equation
import numpy as np
import mpmath
from scipy.optimize import minimize
from single_qubit.gates import Gates


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


@dataclass(frozen=True)
class RewriteRule:
  pattern: List[Gate]
  replacement: List[Gate]
  phase_delta: int  # power of WIGate to add to global phase mod 10


rules = [
  RewriteRule(pattern=[TGate(7)], replacement=[Sigma1(1)], phase_delta=-6),
  RewriteRule(pattern=[Sigma1(1)], replacement=[TGate(7)], phase_delta=6),
  RewriteRule(pattern=[FGate(1), TGate(7), FGate(1)], replacement=[Sigma2(1)], phase_delta=-6),
  RewriteRule(pattern=[Sigma2(1)], replacement=[FGate(1), TGate(7), FGate(1)], phase_delta=6),
  RewriteRule(pattern=[Sigma1(1), Sigma1(1), Sigma1(1)], replacement=[TGate(1)], phase_delta=-2),
  RewriteRule(pattern=[TGate(1)], replacement=[Sigma1(1), Sigma1(1), Sigma1(1)], phase_delta=2),
  RewriteRule(pattern=[Sigma1(1), Sigma2(1), Sigma1(1)], replacement=[FGate(1)], phase_delta=-4),
  RewriteRule(pattern=[FGate(1)], replacement=[Sigma1(1), Sigma2(1), Sigma1(1)], phase_delta=4),
]


def gates_equal(g1: Gate, g2: Gate) -> bool:
  # Compare types and powers modulo 10
  return isinstance(type(g1), type(g2)) and (g1.power % 10) == (g2.power % 10)


class Circuit:
  def __init__(self, gates: List[Gate]):
    self.gates = gates
    self.global_phase = 0  # mod 10

  def __repr__(self):
    gates_str = " ".join(f"{g.__class__.__name__}({g.power})" for g in self.gates)
    return f"Circuit: {gates_str}  (Global phase ω^{self.global_phase})"

  def peephole_optimize_forward(self, rules: List[RewriteRule]):
    changed = True
    while changed:
      changed = False
      i = 0
      while i < len(self.gates):
        for rule in rules:
          plen = len(rule.pattern)
          if i + plen <= len(self.gates):
            window = self.gates[i : i + plen]
            if all(gates_equal(w, p) for w, p in zip(window, rule.pattern)):
              # Replace with Sigma gates and add phase delta
              self.gates[i : i + plen] = rule.replacement
              self.global_phase = (self.global_phase + rule.phase_delta) % 10
              changed = True
              break
        if changed:
          break
        i += 1


T_power_table = [ExactUnitary.T() ** j for j in range(11)]
FT_power_table = [ExactUnitary.F() * T_power_table[j] for j in range(11)]
omega_k_T_j_table = {}
for k in range(11):
  for j in range(11):
    omega_k_T_j_table[ExactUnitary.I().omega_mul(k) * T_power_table[j]] = (k, j)


def G(u):
  return u.gauss_complexity()


def global_phase(U):
  # determinant = e^{2i delta} => delta = 0.5 * arg(det)
  det = np.linalg.det(U)
  delta = np.angle(det) / 2
  return delta


def solve_beta(U00, tau=0.6180339887):
  c = np.abs(U00) / tau
  rhs = (c**2 - 1 - tau**2) / (2 * tau)

  if np.abs(rhs) > 1:
    raise ValueError("No real solution for beta — numerical instability or invalid tau/U00.")

  beta = np.arccos(rhs)
  return beta


F = ExactUnitary.F().to_numpy()

def _decompose_U(U, tol=1e-12):
  delta = global_phase(U)
  U_prime = U * np.exp(-1j * delta)  # remove global phase
  print(np.linalg.det(U_prime))
  if np.abs(U_prime[0, 0]) > tol:
    U00 = U_prime[0, 0]
    beta = solve_beta(U00)

    def error_func(params):
      alpha, gamma = params
      M = Gates.Rz(alpha) @ F @ Gates.Rz(beta) @ F @ Gates.Rz(gamma)
      return np.linalg.norm(M - U_prime) / np.linalg.norm(U_prime)

    res2 = minimize(
      error_func,
      [0, 0],
      bounds=[(-2 * np.pi, 2 * np.pi), (-2 * np.pi, 2 * np.pi)],
    )
    alpha, gamma = res2.x

    return {
      "delta": delta,
      "alpha": alpha,
      "beta": beta,
      "gamma": gamma,
      "form": "Rz-F-Rz-F-Rz",
    }
  else:
    # zero upper-left entry case
    X = Gates.X()
    Rz_phi = U_prime @ X
    diag = np.diag(Rz_phi)
    phi = np.angle(0.5 * diag[0])

    return {"delta": delta, "phi": phi, "form": "Rz-X"}


class ExactFibonacciSynthesizer:
  decimals: int
  TAU = (mpmath.sqrt(5) - 1) / 2
  PHI = TAU + 1

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

    TAU = cls.TAU
    PHI = cls.PHI

    C = mpmath.sqrt(PHI / 4)

    m = int(mpmath.ceil(mpmath.log(C * eps, TAU)) + 1)

    theta = None
    k_final = None

    pi_over_5 = math.pi / 5

    for k in range(-10, 10):  # is solving here faster?
      theta_candidate = -phi / 2 - math.pi * (k / 5)
      if 0 <= theta_candidate <= pi_over_5:
        theta = theta_candidate
        # print("MATH PI/5", math.pi / 5)
        # print("THETA: ", theta)
        k_final = k
        break
    # assert 0 <= theta <= pi_over_5, "Theta out of bounds: " + str(theta)
    if theta is None:
      raise ValueError("Failed to find suitable k.")

    u = Cyclotomic10.Zero()
    v = Cyclotomic10.Zero()
    not_found = True
    k = k_final
    while not_found:
      u0 = RANDOM_SAMPLE(theta, eps, 1)

      xi = ZTau.Phi() * ((ZTau.Phi() ** (2 * m)) - N_i(u0))

      fl = EASY_FACTOR(xi)

      if EASY_SOLVABLE(fl):
        # print("FOUND SOLUTION")
        not_found = False

        u = Cyclotomic10.Omega_(k) * (Cyclotomic10.Tau() ** (m)) * u0

        v = (Cyclotomic10.Tau() ** (m)) * solve_norm_equation(xi)

    # C = exact_synthesize(ExactUnitary(u, v, 0))
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
    assert 0 <= theta <= math.pi / 5, "Theta out of bounds: " + str(theta)
    u = Cyclotomic10.Zero()
    v = Cyclotomic10.Zero()
    not_found = True
    while not_found:
      u0 = RANDOM_SAMPLE(theta, eps, r)
      xi = (ZTau.Phi() ** (2 * m)) - ZTau.Tau() * N_i(u0)
      fl = EASY_FACTOR(xi)

      if EASY_SOLVABLE(fl):
        not_found = False
        v = Cyclotomic10.Omega_(k) * (Cyclotomic10.Tau() ** m) * u0
        ne = solve_norm_equation(xi)
        u = (Cyclotomic10.Tau() ** m) * ne

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
    state = _decompose_U(U)
    if state["form"] == "Rz-F-Rz-F-Rz":
      delta = state["delta"]
      alpha = state["alpha"]
      beta = state["beta"]
      gamma = state["gamma"]
      decomp1 = cls.synthesize_z_rotation(alpha, epsilon)
      # print("D1: ", decomp1.to_numpy())
      # print("Actual: ", Gates.Rz(alpha))
      decomp2 = cls.synthesize_z_rotation(beta, epsilon)
      # print("D2: ", decomp2.to_numpy())
      # print("Actual2: ", Gates.Rz(beta))
      decomp3 = cls.synthesize_z_rotation(gamma, epsilon)
      # print("D3: ", decomp3.to_numpy)
      # print("Actual 3: ", Gates.Rz(gamma))
      return delta, decomp1 * ExactUnitary.F() * decomp2 * ExactUnitary.F() * decomp3, [decomp1.to_numpy(), decomp2.to_numpy(), decomp3.to_numpy()]


def evaluate_gate_sequence(gates: List[Gate]) -> ExactUnitary:
  """
  Applies a sequence of gates to the identity ExactUnitary and returns the result.
  """
  U = ExactUnitary.I()
  for gate in gates:
    if isinstance(gate, TGate):
      U = U * ExactUnitary.T() ** gate.power
    elif isinstance(gate, FGate):
      U = U * ExactUnitary.F()
    elif isinstance(gate, WIGate):
      U = U * ExactUnitary.I().omega_mul(gate.power)
    elif isinstance(gate, Sigma1):
      U = U * ExactUnitary.Sigma1()
    elif isinstance(gate, Sigma2):
      U = U * ExactUnitary.Sigma2()
    else:
      raise ValueError(f"Unknown gate type: {gate}")
  return U


def rz(phi: float) -> np.ndarray:
  """
  Returns the Rz(φ) rotation as a 2x2 numpy array.
  Rz(φ) = exp(-i φ/2 Z) = [[e^{-iφ/2}, 0], [0, e^{iφ/2}]]
  """
  return np.array([[np.exp(-1j * phi / 2), 0], [0, np.exp(1j * phi / 2)]], dtype=complex)


def X() -> np.ndarray:
  """
  Returns the X gate as a 2x2 numpy array.
  X = [[0, 1], [1, 0]]
  """
  return np.array([[0, 1], [1, 0]], dtype=complex)


def simplify_wi_prefix(gates: List[Gate]) -> List[Gate]:
  i = 0
  total_power = 0

  # Collect consecutive WIGates at the front
  while i < len(gates) and isinstance(gates[i], WIGate):
    total_power += gates[i].power
    i += 1

  # Reduce power modulo 10
  reduced_power = total_power % 10

  # Construct the simplified gate list
  simplified = []
  if reduced_power != 0:
    simplified.append(WIGate(power=reduced_power))

  # Append the rest of the gates untouched
  simplified.extend(gates[i:])
  return simplified


def expand_T_power(k: int) -> List[Gate]:
  k_mod = k % 10
  if k_mod == 0:
    return []  # Identity
  return [WIGate(2), Sigma1(), Sigma1(), Sigma1()] * k_mod


def expand_F() -> List[Gate]:
  return [WIGate(4), Sigma1(), Sigma2(), Sigma1()]


def transpile_ft_sigma(gates: List[Gate]) -> List[Gate]:
  result = []

  for gate in gates:
    if isinstance(gate, WIGate):
      if gate.power % 10 == 0:
        pass
    elif isinstance(gate, TGate):
      result.extend(expand_T_power(gate.power))
    elif isinstance(gate, FGate):
      result.extend(expand_F())
    else:
      result.append(gate)

  return result


def reconstruct_from_params(params):
  if params["form"] == "Rz-X":
    return np.exp(1j * params["delta"]) * Gates.Rz(params["phi"]) @ Gates.X()
  elif params["form"] == "Rz-F-Rz-F-Rz":
    delta = params["delta"]
    alpha = params["alpha"]
    # print(alpha)
    beta = params["beta"]
    # print(beta)
    gamma = params["gamma"]
    # print(gamma)
    # rza = ExactFibonacciSynthesizer.synthesize_z_rotation(alpha, 1e-10)
    # rzb = ExactFibonacciSynthesizer.synthesize_z_rotation(beta, 1e-10)
    # rzg = ExactFibonacciSynthesizer.synthesize_z_rotation(gamma, 1e-10)
    return np.exp(1j * delta) * Gates.Rz(alpha) @ F @ Gates.Rz(beta) @ F @ Gates.Rz(gamma)


def norm(U, V):
  Vt = np.conj(np.transpose(V))
  UVt = U @ Vt
  trace = np.trace(UVt)
  return np.sqrt(1 - (np.abs(trace) / 2))


def d_z(phi, U: ExactUnitary):
  phi = mpmath.mpf(phi)
  return mpmath.sqrt(1 - abs((U.u.evaluate() * mpmath.exp(1j * phi / 2)).real))


def d_zx(phi, U: ExactUnitary):
  phi = mpmath.mpf(phi)
  TAU = (mpmath.sqrt(5) - 1) / 2
  return mpmath.sqrt(1 - mpmath.sqrt(TAU) * abs((U.v.evaluate() * mpmath.exp(-1j * (phi / 2 + mpmath.pi / 2))).real))


forward_rules = [
  # σ₁ = (ωI)^6 T^7
  RewriteRule(pattern=[TGate(7)], replacement=[Sigma1(1)], phase_delta=6),
  # σ₂ = (ωI)^6 F T^7 F
  RewriteRule(pattern=[FGate(1), TGate(7), FGate(1)], replacement=[Sigma2(1)], phase_delta=6),
  # Can add more rules here if needed...
]


if __name__ == "__main__":
  mpmath.mp.dps = 400
  phi = 4 * math.pi / 1000
  epsilon = 1e-70

  # g = ExactFibonacciSynthesizer.synthesize_z_rotation(phi, epsilon)
  # print(d_z(phi, g))

  gX = ExactFibonacciSynthesizer.synthesize_zx_rotation(phi, epsilon)
  print(d_zx(phi, gX))

  gXnp = (gX * ExactUnitary.T() ** 5).to_numpy(1e-10)
  actual = Gates.Rz(phi) @ Gates.X
  print("Approximation: ", 1j * gXnp)
  print("Actual matrix: ", actual)
  print(norm(1j * gXnp, actual))

  circ = ExactFibonacciSynthesizer._exact_synthesize(gX)
  print(len(circ))

  c = Circuit(circ)
  c.peephole_optimize_forward(forward_rules)
  print(c)
