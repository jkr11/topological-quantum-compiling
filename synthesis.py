from dataclasses import dataclass
from typing import List, Union
import math
from exactUnitary import *
from numberTheory import RANDOM_SAMPLE, EASY_FACTOR, EASY_SOLVABLE, solve_norm_equation
from rings import N
from util import CONSTANTS


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
  pass


@dataclass(frozen=True)
class Sigma2:
  pass


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


def exact_synthesize(U) -> List[Gate]:
  F = U.__class__.F()
  T = U.__class__.T()
  I = U.__class__.I()
  g = U.gauss_complexity()
  Ur = U
  C: List[Gate] = []

  while g > 2:
    min_complexity = math.inf
    J = 0
    for j in range(11):
      gg = (F * T**j * Ur).gauss_complexity()
      if gg < min_complexity:
        min_complexity = gg
        J = j
    Ur = F * T**J * Ur
    g = Ur.gauss_complexity()
    C.insert(0, TGate((10 - J) % 10))
    C.insert(0, FGate())

  # Final matching to I * ω^k * T^j form
  for k in range(11):
    for j in range(11):
      t = I.omega_mul(k) * T**j
      if t == Ur:
        if j != 0:
          C.insert(0, TGate(j))
        if k != 0:
          C.insert(0, WIGate(k))
        return canonicalize_and_reduce_identities(C)
  raise ValueError("Final reduction failed")


def synthesize_z_rotation(phi: float, eps: float) -> List[Gate]:
  C = np.sqrt(CONSTANTS.PHI / 4)
  print("C: ", C)
  print("C * eps: ", C * eps)
  print("log(C * eps): ", math.log(C * eps, CONSTANTS.TAU))
  m = math.ceil(math.log(C * eps, CONSTANTS.TAU)) + 1
  print("TAU ** m:", CONSTANTS.PHI ** m)
  print("m: ", m)
  theta = 0
  for k in range(10):
    theta_candidate = -phi / 2 - math.pi * (k / 5)
    if 0 <= theta_candidate <= math.pi / 5:
      theta = theta_candidate
      break
  u = 0
  v = 0
  found = False
  while not found:
    u0 = RANDOM_SAMPLE(theta, eps, 1.0)
    print("2 * m", 2 * m)
    xi = RealCyclotomic10.Phi() * ((RealCyclotomic10.Phi() ** (2 * m)) - N(u0) ** 2)
    fl = EASY_FACTOR(xi)
    if EASY_SOLVABLE(fl):
      found = True
      u = Cyclotomic10.Omega_(k) * Cyclotomic10.Tau() ** m * u0
      v = Cyclotomic10.Tau() ** m * solve_norm_equation(xi)
  C = exact_synthesize(ExactUnitary(u, v, 0))
  return C


if __name__ == "__main__":
  #U = ExactUnitary.F()
  #gates = exact_synthesize(U)
  #print("FT circuit:", gates)
  #
  #reduced = fully_reduce_to_sigma(gates)
  #print("σ₁σ₂ circuit:", reduced)
  #
  #U2 = ExactUnitary.I()
  #gates = exact_synthesize(U2)
  #print(gates)
  #red = fully_reduce_to_sigma(gates)
  #print(red)
  #
  print("Cycl PHI**m", (RealCyclotomic10.Phi() ** 52).evaluate())
  print("Const PHI**m", CONSTANTS.PHI ** 52)
  phi = math.pi / 5  # π/10 rotation
  epsilon = 1e-1  # precision
  z_gates = synthesize_z_rotation(phi, epsilon)
  print(f"Z-rotation circuit for φ = π/10:")
  print(f"Number of gates: {len(z_gates)}")
  print(f"Gate sequence: {z_gates}")
  #print("Test")
  #print("ApproxReal", RANDOM_SAMPLE(math.pi/6, 1e-5, 1.0))
  