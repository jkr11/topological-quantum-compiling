from dataclasses import dataclass
import numpy as np
from typing import List, Tuple, Dict
from enum import Enum
import itertools

type Tensor = np.ndarray


@dataclass
class Gate:
  tensor: Tensor
  name: List[str]

  def __matmul__(self, other):
    return Gate(self.tensor @ other.tensor, self.name + other.name)

  def adjoint(self) -> "Gate":
    return Gate(self.tensor.conj().T, list(reversed(self.name)))


@dataclass
class twoQubitCircuit:
  gates: List[Gate]

  def _eval_circuit(self):
    init = np.eye(2, dtype=complex)
    for gate in self.gates:
      init @ gate[0]
    return init


def hgate() -> np.ndarray:
  return (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]])


def pzgate(phase: complex) -> Tensor:
  return np.array([[1, 0], [0, phase]])


def Tgate() -> Gate:
  return Gate(pzgate(np.exp(1j * np.pi / 4)), "T")


def Hgate() -> Gate:
  return Gate(hgate(), "H")


def rx(theta: float) -> np.ndarray:
  """Return the X-axis rotation gate (R_x) by angle theta."""
  return np.array([[np.cos(theta / 2), -1j * np.sin(theta / 2)], [-1j * np.sin(theta / 2), np.cos(theta / 2)]])


def ry(theta: float) -> np.ndarray:
  """Return the Y-axis rotation gate (R_y) by angle theta."""
  return np.array([[np.cos(theta / 2), -np.sin(theta / 2)], [np.sin(theta / 2), np.cos(theta / 2)]])


class GateType(Enum):
  FIB = 1
  HT = 2
  MAJ = 3


ADJOINT_MAP: Dict[GateType, Dict[str, str]] = {
  GateType.HT: {"H": "H", "T": "T†", "T†": "T"},
  GateType.FIB: {
    "σ1": "σ1⁻¹",
    "σ1⁻¹": "σ1",
    "σ2": "σ2⁻¹",
    "σ2⁻¹": "σ2",
  },
}


def adjoin_gates(gates: List[Gate]) -> List[Gate]:
  return [gate.adjoint() for gate in reversed(gates)]


def eval_circuit(gates: List[Gate]) -> np.ndarray:
  init = np.eye(2)
  print(len(gates))
  for gate in gates:
    init = init @ gate.tensor
  return init


def _generate_binary_list(n: int = 13) -> List[List[int]]:
  """Generate all binary strings from length 1 to n."""
  binary_list = []
  for length in range(1, n + 1):
    binary_list.extend([list(map(int, bin(i)[2:].zfill(length))) for i in range(2**length)])
  return binary_list


def generate_nary_sequences(base: int, max_len: int) -> List[List[int]]:
  """
  Generate all sequences of digits [0, ..., base-1] up to length `max_len`.
  """
  sequences = []
  for length in range(1, max_len + 1):
    sequences.extend(itertools.product(range(base), repeat=length))
  return [list(seq) for seq in sequences]


def _create_gate_list(gate_dict: Dict[int, Gate], max_len: int = 5) -> List[Gate]:
  """
  Generate all gate sequences from the provided gate_dict.
  `gate_dict` should map integers to gates. Supports any number of gates.
  """
  base = len(gate_dict)
  nary_sequences = generate_nary_sequences(base, max_len)
  gate_list = []
  for sequence in nary_sequences:
    composed_gate = Gate(np.eye(2, dtype=complex), [])
    for idx in sequence:
      gate = gate_dict.get(idx)
      if gate is None:
        continue
      composed_gate = composed_gate @ gate
    gate_list.append(composed_gate)
  return gate_list


def _create_gate_list() -> List[Gate]:
  """Create the basic gate list by concatenating Hs and Ts."""
  gate_list = []
  binary_list = _generate_binary_list()
  for bits in binary_list:
    u = Gate(np.eye(2), "")

    for bit in bits:
      if bit:
        gate = Tgate()
      else:
        gate = Hgate()

      u = u @ gate

    gate_list.append(u)

  return gate_list


def trace_dist(u: Tensor, v: Tensor) -> float:
  """Compute trace distance between two 2x2 matrices."""
  return np.real(0.5 * np.trace(np.sqrt((u - v).conj().transpose() @ (u - v))))


@dataclass
class SolovayKitaev:
  gate_list: List[Gate]

  def _to_bloch(self, u: Tensor):
    """Compute angle and axis for a unitary."""

    angle = np.real(np.arccos((u[0, 0] + u[1, 1]) / 2))
    sin = np.sin(angle)
    if sin < 1e-10:
      axis = [0, 0, 1]
    else:
      nx = (u[0, 1] + u[1, 0]) / (2j * sin)
      ny = (u[0, 1] - u[1, 0]) / (2 * sin)
      nz = (u[0, 0] - u[1, 1]) / (2j * sin)
      axis = [nx, ny, nz]
    return axis, 2 * angle

  def gc_decomp(self, u: Tensor) -> Tuple[Tensor, Tensor]:
    """Group commutator decomposition."""

    def diagonalize(u: Tensor) -> Tensor:
      _, v = np.linalg.eig(u)
      return v

    axis, theta = self._to_bloch(u)

    phi = 2.0 * np.arcsin(np.sqrt(np.sqrt((0.5 - 0.5 * np.cos(theta / 2)))))

    v = rx(phi)
    if axis[2] > 0:
      w = ry(2 * np.pi - phi)
    else:
      w = ry(phi)

    ud = diagonalize(u)
    vwvdwd = diagonalize(v @ w @ v.conj().T @ w.conj().T)
    s = ud @ vwvdwd.conj().T

    v_hat = s @ v @ s.conj().T
    w_hat = s @ w @ s.conj().T
    return v_hat, w_hat

  def solovay_kitaev(self, u: Tensor, n: int) -> List[Gate]:
    assert u.shape == (2, 2)
    # self.gate_list = _create_gate_list()
    output_gates: List[Gate] = []
    approx = self._sk_iter(u, output_gates, n)
    print(f"Trace distance: {trace_dist(u, approx)}")
    prgate = ""
    for gate in output_gates:
      prgate += "".join(gate.name)
    print(prgate)
    return output_gates

  def _sk_iter(self, u: Tensor, output_gate: List[Gate], n: int) -> Tensor:
    """Solovay-Kitaev Algorithm."""

    if n == 0:
      gate = self._find_closest_u(u)
      output_gate.append(gate)
      return gate.tensor
    else:
      ugate, vgate, wgate = [], [], []
      u_next = self._sk_iter(u, ugate, n - 1)
      v, w = self.gc_decomp(u @ u_next.conj().T)
      v_next = self._sk_iter(v, vgate, n - 1)
      w_next = self._sk_iter(w, wgate, n - 1)

      vadj, wadj = adjoin_gates(vgate), adjoin_gates(wgate)
      output_gate.clear()
      output_gate.extend(vgate)
      output_gate.extend(wgate)
      output_gate.extend(vadj)
      output_gate.extend(wadj)
      output_gate.extend(ugate)
    return v_next @ w_next @ v_next.conj().T @ w_next.conj().T @ u_next

  def _find_closest_u(self, unitary: Tensor) -> Gate:
    distances = np.array([trace_dist(g.tensor, unitary) for g in self.gate_list])
    return self.gate_list[np.argmin(distances)]


if __name__ == "__main__":
  target_unit = np.array([[0, 1], [1, 0]])
  sk = SolovayKitaev(_create_gate_list())
  gates = sk.solovay_kitaev(target_unit, 4)
  assert np.allclose(eval_circuit(gates), target_unit)
