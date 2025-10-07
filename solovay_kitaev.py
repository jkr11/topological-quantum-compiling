from dataclasses import dataclass
import numpy as np
from typing import List, Tuple, Dict
from enum import Enum
import itertools
from src.exact_synthesis.exactUnitary import ExactUnitary
from src.exact_synthesis.util import Gates

type Tensor = np.ndarray


@dataclass
class Gate:
  tensor: Tensor
  name: List[str]

  def __matmul__(self, other):
    return Gate(self.tensor @ other.tensor, self.name + other.name)

  def adjoint(self) -> "Gate":
    return Gate(self.tensor.conj().T, list(reversed(self.name)))


def hgate() -> np.ndarray:
  return (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]])


def pzgate(phase: complex) -> Tensor:
  return np.array([[1, 0], [0, phase]])


def Tgate() -> Gate:
  return Gate(pzgate(np.exp(1j * np.pi / 4)), "T")


def Hgate() -> Gate:
  return Gate(hgate(), "H")


Nomega = np.exp(1j * np.pi / 5)

Ntau = (np.sqrt(5) - 1) / 2
sqrt_tau = np.sqrt(Ntau)

Tnp = np.array([[1, 0], [0, Nomega]], dtype=complex)

Fnp = np.array([[Ntau, sqrt_tau], [sqrt_tau, -Ntau]], dtype=complex)

sigma1 = np.array([[Nomega**6, 0], [0, Nomega**13]], dtype=complex)

sigma2 = Fnp @ sigma1 @ Fnp.T


def S1gate() -> Gate:
  return Gate(sigma1, "S1")


def S2gate() -> Gate:
  return Gate(sigma2, "S2")


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


def _generate_binary_list(n: int = 20) -> List[List[int]]:
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
        gate = S1gate()
      else:
        gate = S2gate()

      u = u @ gate

    gate_list.append(u)

  return gate_list


def trace_dist(u: Tensor, v: Tensor) -> float:
  """Compute trace distance between two 2x2 matrices."""
  return np.real(0.5 * np.trace(np.sqrt((u - v).conj().transpose() @ (u - v))))


def _to_quaternion(U: Tensor) -> Tensor:
  r"""
  Converts :math:U :math:\in :math:\text{SU}_2{\mathbb{C}} as a unique quaternion
  """
  return np.array([
    U[0, 0].real,
    U[0, 0].imag,
    U[0, 1].imag,
    -U[0, 1].real,
  ])


def _from_quaternion(q: np.ndarray) -> Tensor:
  a, b, c, d = q
  return np.array(
    [
      [a + 1j * d, -c + 1j * b],
      [c + 1j * b, a - 1j * d],
    ],
    dtype=np.complex128,
  )


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


import scipy as sp
from typing import Any

_CLIFFORD_T_BASIS = {
  "H": Gates.H,
  "T": Gates.T,
  "T*": Gates.T.conj().T,
}

_FIBONACCI_BASIS = {
  "σ1": sigma1,
  "σ1*": sigma1.conj().T,
  "σ2": sigma2,
  "σ2*": sigma2.conj().T,
}


def _SU2_transform(U):
  """Strip global phase and return SU(2)-normalized matrix and global phase."""
  phase = np.angle(np.linalg.det(U)) / 2
  su2 = U * np.exp(-1j * phase)
  return su2, phase % np.pi


def _is_approximated(target_op: np.ndarray, ops: List[np.ndarray] = None, kd_tree: sp.spatial.KDTree = None, tol=1e-8) -> Tuple[bool, Tensor, Any]:
  gate_points = [_to_quaternion(target_op)]
  tree = kd_tree or sp.spatial.KDTree(ops)
  dist, indx = tree.query(gate_points, workers=-1)
  return (dist[0] < tol, gate_points[0], indx[0])


def _build_eps_net(gate_set: dict[str, np.ndarray], max_depth=10):
  gates = list(gate_set.keys())
  gate_mats = {k: _SU2_transform(v)[0] for k, v in gate_set.items()}
  gate_phases = {k: _SU2_transform(v)[1] for k, v in gate_set.items()}

  approx_ids = [[g] for g in gates]
  approx_mats = [gate_mats[g] for g in gates]
  approx_phs = [gate_phases[g] for g in gates]
  approx_quats = [_to_quaternion(m) for m in approx_mats]

  for depth in range(max_depth - 1):
    kdtree = sp.spatial.KDTree(np.array(approx_quats))
    new_ids, new_mats, new_phs, new_quats = [], [], [], []
    for seq, mat, phase in zip(approx_ids, approx_mats, approx_phs):
      for g in gates:
        if len(seq) > 0 and seq[-1] == g + "*":
          continue
        new_seq = seq + [g]
        total_phase = gate_phases[g] + phase
        new_mat = (-1) ** (total_phase >= np.pi) * gate_mats[g] @ mat
        norm_phase = total_phase % np.pi
        exists, quat, idx = _is_approximated(new_mat, kd_tree=kdtree)
        if not exists or norm_phase != approx_phs[idx]:
          new_ids.append(new_seq)
          new_mats.append(new_mat)
          new_phs.append(norm_phase)
          new_quats.append(quat)
    approx_ids += new_ids
    approx_mats += new_mats
    approx_phs += new_phs
    approx_quats += new_quats
  return approx_ids, approx_mats, approx_phs, approx_quats


def __run_example():
  target_unit = np.array([[0, 1], [1, 0]])
  sk = SolovayKitaev(_create_gate_list())
  gates = sk.solovay_kitaev(target_unit, 4)
  assert np.allclose(eval_circuit(gates), target_unit)


if __name__ == "__main__":
  i, m, p, q = _build_eps_net(_FIBONACCI_BASIS, max_depth=6)
  print(q)
