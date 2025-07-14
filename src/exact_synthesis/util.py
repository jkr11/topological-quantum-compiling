from exact_synthesis.exactUnitary import ExactUnitary
import numpy as np
from typing import Union, Tuple, Optional
import math
import mpmath
from mpmath import mp


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


def trace_norm(U: np.ndarray, V: Union[np.ndarray, ExactUnitary]) -> float:
  if isinstance(V, np.ndarray):
    return np.sqrt(1 - np.abs(np.trace(U @ V.conj().T)) / 2)
  elif isinstance(V, ExactUnitary):
    r, theta = is_Rz(U)
    if r and V.k == 0:
      return np.sqrt(1 - np.abs(np.real(V.u.evaluate() * np.exp(1j * theta / 2))))
    else:
      return trace_norm(U, V.to_matrix_np)


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
  # Example usage
  mp.dps = 200
  PHI = CONSTANTS.PHI
  TAU = CONSTANTS.TAU

  print("TAU**2", TAU**2)
  print("TAU - 1 ", 1 - TAU)
