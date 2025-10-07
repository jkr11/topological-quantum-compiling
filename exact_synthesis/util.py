import numpy as np
from typing import Tuple, Optional
import mpmath


SQRT_5 = mpmath.mp.sqrt(5)
TAU = (SQRT_5 - 1) / 2
PHI = (SQRT_5 - 1) / 2 + 1


class CONSTANTS:
  PHI = PHI
  TAU = TAU


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
