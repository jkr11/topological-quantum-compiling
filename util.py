from exactUnitary import ExactUnitary
import numpy as np
from typing import Union, Tuple, Optional
import math

SQRT_5 = math.sqrt(5)
TAU = (SQRT_5 - 1) / 2  # Here we use this directly, no need to factor down from the rings
PHI = TAU + 1


class CONSTANTS:
  PHI = TAU + 1  # Golden ratio
  TAU = TAU  # Tau constant



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

if __name__ == "__main__":
  # Example usage
  PHI = CONSTANTS.PHI
  TAU = CONSTANTS.TAU

  print("TAU**2 = TAU - 1", TAU**2 == TAU - 1)
  print("TAU**2", TAU**2)
  print("TAU - 1 ", 1 - TAU)