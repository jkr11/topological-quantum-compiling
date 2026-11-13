from __future__ import annotations

import itertools
from dataclasses import dataclass
from functools import reduce
from operator import matmul

import numpy as np

Tensor = np.ndarray


def _dagger_name(name: str) -> str:
  if name == "H":  # self-adjoint
    return name
  return name[:-1] if name.endswith("*") else name + "*"


@dataclass
class Gate:
  tensor: Tensor
  name: tuple[str, ...]

  def __matmul__(self, other: Gate) -> Gate:
    return Gate(self.tensor @ other.tensor, self.name + other.name)

  def adjoint(self) -> Gate:
    return Gate(self.tensor.conj().T, tuple(_dagger_name(n) for n in reversed(self.name)))


def rx(theta: float) -> Tensor:
  c, s = np.cos(theta / 2), np.sin(theta / 2)
  return np.array([[c, -1j * s], [-1j * s, c]])


def ry(theta: float) -> Tensor:
  c, s = np.cos(theta / 2), np.sin(theta / 2)
  return np.array([[c, -s], [s, c]])


_H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
_T = np.diag([1, np.exp(1j * np.pi / 4)])
CLIFFORD_T = {"H": _H, "T": _T, "T*": _T.conj().T}

_w = np.exp(1j * np.pi / 5)
_tau = (np.sqrt(5) - 1) / 2
_F = np.array([[_tau, np.sqrt(_tau)], [np.sqrt(_tau), -_tau]])
_s1 = np.diag([_w**6, _w**13])
_s2 = _F @ _s1 @ _F.T
FIBONACCI = {"σ1": _s1, "σ1*": _s1.conj().T, "σ2": _s2, "σ2*": _s2.conj().T}


def create_gate_list(basis: dict[str, Tensor], max_len: int) -> list[Gate]:
  gates = [Gate(t, (n,)) for n, t in basis.items()]
  return [
    reduce(matmul, seq)
    for length in range(1, max_len + 1)
    for seq in itertools.product(gates, repeat=length)
  ]


def trace_dist(u: Tensor, v: Tensor) -> Tensor:
  return 0.5 * np.linalg.norm(u - v, axis=-2).sum(axis=-1)


def _bloch(u: Tensor):
  """Rotation axis and angle of a unitary."""
  angle = np.real(np.arccos((u[0, 0] + u[1, 1]) / 2))
  sin = np.sin(angle)
  if sin < 1e-10:
    axis = [0, 0, 1]
  else:
    axis = [
      (u[0, 1] + u[1, 0]) / (2j * sin),
      (u[0, 1] - u[1, 0]) / (2 * sin),
      (u[0, 0] - u[1, 1]) / (2j * sin),
    ]
  return axis, 2 * angle


def _gc_decomp(u: Tensor) -> tuple[Tensor, Tensor]:
  """Group commutator decomposition: u ~ v w v† w†."""
  axis, theta = _bloch(u)
  phi = 2.0 * np.arcsin(np.sqrt(np.sqrt(0.5 - 0.5 * np.cos(theta / 2))))
  v = rx(phi)
  w = ry(2 * np.pi - phi) if np.real(axis[2]) > 0 else ry(phi)

  ud = np.linalg.eig(u)[1]
  cd = np.linalg.eig(v @ w @ v.conj().T @ w.conj().T)[1]
  s = ud @ cd.conj().T
  return s @ v @ s.conj().T, s @ w @ s.conj().T


class SolovayKitaev:
  def __init__(self, gate_list: list[Gate]):
    self.gate_list = gate_list
    self._tensors = np.stack([g.tensor for g in gate_list])

  def _closest(self, u: Tensor) -> Gate:
    return self.gate_list[np.argmin(trace_dist(self._tensors, u))]

  def _approx(self, u: Tensor, n: int) -> Gate:
    if n == 0:
      return self._closest(u)
    u_prev = self._approx(u, n - 1)
    v, w = _gc_decomp(u @ u_prev.tensor.conj().T)
    v, w = self._approx(v, n - 1), self._approx(w, n - 1)
    return v @ w @ v.adjoint() @ w.adjoint() @ u_prev

  def solovay_kitaev(self, u: Tensor, n: int) -> Gate:
    assert u.shape == (2, 2)
    gate = self._approx(u, n)
    print(f"Trace distance: {trace_dist(gate.tensor, u)}")
    print("".join(gate.name))
    return gate


if __name__ == "__main__":
  target = np.array([[0, 1], [1, 0]])
  sk = SolovayKitaev(create_gate_list(FIBONACCI, 10))
  result = sk.solovay_kitaev(target, 2)
  assert np.allclose(result.tensor, target)
  # TODO: expose peephole module for this as well