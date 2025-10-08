import numpy as np
from typing import Tuple, Optional
import mpmath


SQRT_5 = mpmath.mp.sqrt(5)
TAU = (SQRT_5 - 1) / 2
PHI = (SQRT_5 - 1) / 2 + 1


class CONSTANTS:
  PHI = PHI
  TAU = TAU


def phase(z):
  return np.angle(z)


def mag(z):
  return np.abs(z)


def cis(theta):
  return np.cos(theta) + 1j * np.sin(theta)


def adj(z):
  return np.conj(z)


def euler_angles(U: np.ndarray) -> Tuple[float, float, float, float]:
  a, b = U[0, 0], U[0, 1]
  c, d = U[1, 0], U[1, 1]

  det = np.linalg.det(U)
  alpha = np.angle(det) / 2

  gamma = 2 * np.arctan2(mag(b), mag(a))

  i = 1j

  delta = phase(b * d * i * adj(det))

  beta = 2 * phase(d * cis(-alpha - delta / 2) + c * cis(-alpha + delta / 2) * i)

  return (alpha, beta, gamma, delta)


def matrix_of_euler_angles(angles: Tuple) -> np.ndarray:
  alpha, beta, gamma, delta = angles

  def cplx_cis(theta):
    return np.cos(theta) + 1j * np.sin(theta)

  hadamard = Gates.H

  def zrot(gamma):
    return np.array([[cplx_cis(-gamma / 2), 0], [0, cplx_cis(gamma / 2)]], dtype=complex)

  opa = cplx_cis(alpha) * np.identity(2, dtype=complex)
  opb = zrot(beta)
  opc = hadamard @ zrot(gamma) @ hadamard
  opd = zrot(delta)

  op = opa @ opb @ opc @ opd
  return op


def is_Rz(U: np.ndarray, tol: float = 1e-10) -> Tuple[bool, Optional[float]]:
  if U.shape != (2, 2):
    return False, None
  if not np.allclose(U, np.diag(np.diag(U)), atol=tol):
    return False, None
  d0, d1 = np.diag(U)
  if not np.allclose(U.conj().T @ U, np.eye(2), atol=tol):
    return False, None
  if not np.allclose(d1, np.conj(d0), atol=tol):
    return False, None
  theta = -2 * np.angle(d0)
  return True, theta


def trace_norm(U: np.ndarray, V: np.ndarray) -> float:
  trace = np.abs(np.trace(U @ V.conj().T))
  print("Trace: ", trace)
  return np.sqrt(1 - trace / 2)


class Gates(object):
  """
  Collection of common quantum gates.
  """

  # Pauli matrices
  X = np.array([[0.0, 1.0], [1.0, 0.0]])
  Y = np.array([[0.0, -1.0j], [1.0j, 0.0]])
  Z = np.array([[1.0, 0.0], [0.0, -1.0]])
  # Hadamard gate
  H = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2)
  # S gate
  S = np.array([[1.0, 0.0], [0.0, 1.0j]])
  # T gate
  T = np.array([[1.0, 0.0], [0.0, np.sqrt(1j)]])
  # swap gate
  swap = np.identity(4)[[0, 2, 1, 3]]

  CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)


if __name__ == "__main__":
  op = np.array([[1 / np.sqrt(2), 1j / np.sqrt(2)], [1j / np.sqrt(2), 1 / np.sqrt(2)]], dtype=complex)

  alpha, beta, gamma, delta = euler_angles(op)
  print(f"Euler angles: alpha={alpha}, beta={beta}, gamma={gamma}, delta={delta}")
  print(f"Original op: {op}")
  op_reconstructed = matrix_of_euler_angles((alpha, beta, gamma, delta))
  print("Reconstructed operator:\n", op_reconstructed)
  assert np.allclose(op, op_reconstructed)
