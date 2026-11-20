import numpy as np
from exact_synthesis.exactUnitary import ExactUnitary
import mpmath
from mpmath import mpf
from typing import Any


def haar_random_su2():  # TODO: check
  u = np.random.normal(size=4)
  u /= np.linalg.norm(u)  # generates a unit quaternion on S3

  a, b, c, d = u

  return np.array([[a + 1j * b, c + 1j * d], [-c + 1j * d, a - 1j * b]])


def d_z(U: ExactUnitary, phi: Any) -> Any:
  phi = mpmath.mpf(phi)
  u = U.u.evaluate()

  rotated = u * mpmath.exp(1j * phi / 2)

  return mpmath.sqrt(max(mpmath.mpf(0), 1 - abs(mpmath.re(rotated))))


def d_zx(U: ExactUnitary, phi: Any) -> Any:
  tau = (mpmath.sqrt(5) - 1) / 2

  phi = mpmath.mpf(phi)
  v = U.v.evaluate()
  inner = abs(mpmath.re(v * mpmath.exp(-1j * (phi / 2 + mpmath.pi / 2))))
  return mpmath.sqrt(1 - mpmath.sqrt(tau) * inner)


def d_trace(U, V):
  if isinstance(U, ExactUnitary):
    U = U.to_matrix()
  if isinstance(V, ExactUnitary):
    V = V.to_matrix()
  tr = sum(U[i, j] * mpmath.conj(V[i, j]) for i in range(2) for j in range(2))
  return mpmath.sqrt(max(mpf(0), 1 - abs(tr) / 2))


def rz_mp(theta):
  theta = mpmath.mpf(float(theta))
  return mpmath.diag([mpmath.exp(-1j * theta / 2), mpmath.exp(1j * theta / 2)])


def d_operator(U: ExactUnitary, V: ExactUnitary) -> Any:
  U_mat = U.to_matrix()
  V_mat = V.to_matrix()

  product = U_mat.H * V_mat
  trace = product[0, 0] + product[1, 1]

  fidelity = min(mpmath.mpf(1), abs(trace) / 2)

  return mpmath.sqrt(2 - 2 * fidelity)
