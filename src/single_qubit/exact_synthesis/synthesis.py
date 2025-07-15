from dataclasses import dataclass
from typing import List, Union
import math
from single_qubit.exact_synthesis.exactUnitary import ExactUnitary
from single_qubit.exact_synthesis.rings import Cyclotomic10, ZTau, N_i
from single_qubit.exact_synthesis.numberTheory import RANDOM_SAMPLE, EASY_FACTOR, EASY_SOLVABLE, solve_norm_equation
from single_qubit.exact_synthesis.util import trace_norm
import numpy as np
import mpmath


@dataclass(frozen=True)
class TGate:
  power: int  # modulo 10


@dataclass(frozen=True)
class FGate:
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


def canonicalize_and_reduce_identities(gates: List[Gate]) -> List[Gate]:
  t_accum = 0
  w_accum = 0
  result = []

  i = 0
  while i < len(gates):
    gate = gates[i]
    if isinstance(gate, TGate):
      t_accum += gate.power
    elif isinstance(gate, WIGate):
      w_accum += gate.power
    elif isinstance(gate, FGate):
      if t_accum % 10 != 0:
        result.append(TGate(t_accum % 10))
        t_accum = 0
      if w_accum % 10 != 0:
        result.insert(0, WIGate(w_accum % 10))
        w_accum = 0
      if result and isinstance(result[-1], FGate):
        result.pop()  # F * F = I
      else:
        result.append(FGate())
    else:
      if t_accum % 10 != 0:
        result.append(TGate(t_accum % 10))
        t_accum = 0
      if w_accum % 10 != 0:
        result.insert(0, WIGate(w_accum % 10))
        w_accum = 0
      result.append(gate)
    i += 1

  if t_accum % 10 != 0:
    result.append(TGate(t_accum % 10))
  if w_accum % 10 != 0:
    result.insert(0, WIGate(w_accum % 10))

  cleaned = []
  for g in result:
    if isinstance(g, TGate) and g.power % 10 == 0:
      continue
    if isinstance(g, WIGate) and g.power % 10 == 0:
      continue
    cleaned.append(g)
  return cleaned


REWRITE_RULES = [
  ([WIGate(6), TGate(7)], [Sigma1()]),
  ([WIGate(6), FGate(), TGate(7), FGate()], [Sigma2()]),
  ([FGate()], [WIGate(4), Sigma1(), Sigma2(), Sigma1()]),
]


def peephole_rewrite_to_sigma(gates: List[Gate]) -> List[Gate]:
  i = 0
  result = []
  while i < len(gates):
    matched = False
    for pattern, replacement in REWRITE_RULES:
      if gates[i : i + len(pattern)] == pattern:
        print(pattern)
        result.extend(replacement)
        i += len(pattern)
        matched = True
        break
    if not matched:
      result.append(gates[i])
      i += 1
  return result


def fully_reduce_to_sigma(gates: List[Gate]) -> List[Gate]:
  gates = canonicalize_and_reduce_identities(gates)
  while True:
    new_gates = peephole_rewrite_to_sigma(gates)
    if new_gates == gates:
      break
    gates = new_gates
  return gates


T_power_table = [ExactUnitary.T() ** j for j in range(11)]
FT_power_table = [ExactUnitary.F() * T_power_table[j] for j in range(11)]
omega_k_T_j_table = {}
for k in range(11):
  for j in range(11):
    omega_k_T_j_table[ExactUnitary.I().omega_mul(k) * T_power_table[j]] = (k, j)


def G(u):
  return u.gauss_complexity()


def __exact_synthesize(U: ExactUnitary) -> List[Gate]:
  g = G(U)
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
    C.insert(0, FGate())
  if Ur in omega_k_T_j_table:
    k, j = omega_k_T_j_table[Ur]
    if j != 0:
      C.insert(0, TGate(j))
    if k != 0:
      C.insert(0, WIGate(k))
    return C
  raise ValueError("Final reduction failed")


def __synthesize_z_rotation(phi: float, eps: float) -> List[Gate]:
  phi = mpmath.mpf(phi)
  eps = mpmath.mpf(eps)
  PHI = (mpmath.sqrt(5) + 1) / 2
  TAU = (mpmath.sqrt(5) - 1) / 2
  C = mpmath.sqrt(PHI / 4)

  m = int(mpmath.ceil(mpmath.log(C * eps, TAU))) + 1

  theta = 0
  for k in range(-10, 10):
    theta_candidate = -phi / 2 - math.pi * (k / 5)
    if 0 <= theta_candidate <= math.pi / 5:
      theta = theta_candidate
      break
  assert 0 <= theta <= math.pi / 5, "Theta out of bounds: " + str(theta)
  u = 0
  v = 0
  not_found = True
  while not_found:
    u0 = RANDOM_SAMPLE(theta, eps, 1.0)
    xi = ZTau.Phi() * ((ZTau.Phi() ** (2 * m)) - N_i(u0))
    fl = EASY_FACTOR(xi)
    print("Factorization of xi:", fl)
    if EASY_SOLVABLE(fl):
      not_found = False
      u = Cyclotomic10.Omega_(k) * (Cyclotomic10.Tau() ** m) * u0
      v = (Cyclotomic10.Tau() ** m) * solve_norm_equation(xi)

  C = __exact_synthesize(ExactUnitary(u, v, 0))
  return C


def __synthesize_zx_rotation(phi: float, eps: float) -> List[Gate]:
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
    xi = (ZTau.Phi() ** (2 * m)) - ZTau.Tau() * N_i(u0)
    fl = EASY_FACTOR(xi)
    print("Factorization of xi:", fl)
    if EASY_SOLVABLE(fl):
      not_found = False
      u = Cyclotomic10.Omega_(k) * (Cyclotomic10.Tau() ** m) * u0
      ne = solve_norm_equation(xi)
      v = (Cyclotomic10.Tau() ** m) * ne
  print("u:", u)
  print("v:", v)
  C = __exact_synthesize(ExactUnitary(u, v, 0))
  return C


class ExactFibonacciSynthesizer:
  decimals: int
  TAU = (mpmath.sqrt(5) - 1) / 2
  PHI = TAU + 1

  @classmethod
  def __G(u: ExactUnitary):
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

    for k in range(-10, 10):  # is solving here faster?
      theta_candidate = -phi / 2 - math.pi * (k / 5)
      if 0 <= theta_candidate <= math.pi / 5:
        theta = theta_candidate
        k_final = k
        break
    assert 0 <= theta <= math.pi / 5, "Theta out of bounds: " + str(theta)
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
        print("FOUND SOLUTION")
        not_found = False

        u = Cyclotomic10.Omega_(k) * (Cyclotomic10.Tau() ** (m)) * u0

        v = (Cyclotomic10.Tau() ** (m)) * solve_norm_equation(xi)

    # C = exact_synthesize(ExactUnitary(u, v, 0))
    return ExactUnitary(u, v, 0)

  @classmethod
  def synthesize_z_rotation(cls, phi, eps) -> List[Gate]:
    return cls._synthesize_z_rotation(phi, eps)

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
      C.insert(0, FGate())
    if Ur in omega_k_T_j_table:
      k, j = omega_k_T_j_table[Ur]
      if j != 0:
        C.insert(0, TGate(j))
      if k != 0:
        C.insert(0, WIGate(k))
      return C
    raise ValueError("Final reduction failed")


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


if __name__ == "__main__":
  phi = 4 * math.pi / 1000
  epsilon = 1e-7
  z_gates = ExactFibonacciSynthesizer.synthesize_z_rotation(phi, epsilon)
  print("Z-rotation circuit for φ = π/10:")
  print(f"Number of gates: {len(z_gates)}")
  print(f"Gate sequence: {z_gates}")
  print
  gates_seq = z_gates
  print("Evaluating gates: ", evaluate_gate_sequence(gates_seq))
  print("As numpy matrix:\n", evaluate_gate_sequence(gates_seq).to_matrix)
  print("Actual z matrix:\n", rz(4 * math.pi / 1000))
  eval_unitary = evaluate_gate_sequence(gates_seq)
  eval_matrix = eval_unitary.to_matrix
  actual_matrix = rz(4 * math.pi / 1000)
  print("Evaluating gates: ", eval_unitary)
  print("As numpy matrix:\n", eval_matrix)
  print("Actual z matrix:\n", actual_matrix)
  # Check if the matrices are close within a given tolerance
  tol = 1e-5  # or whatever accuracy you want
  norm = trace_norm(np.array(eval_matrix).reshape(2, 2), actual_matrix)
  print(f"Trace norm between evaluated and actual: {norm}")
  if norm < tol:
    print("PASS: The evaluated matrix is within the desired accuracy.")
  else:
    print("FAIL: The evaluated matrix is NOT within the desired accuracy.")
