import mpmath
from exact_synthesis.synthesis import synthesize_z_rotation, exact_synthesize
from exact_synthesis.exactUnitary import ExactUnitary


def dist(U, phi):
  u_val = U.u.evaluate()
  phi = mpmath.mpf(phi)
  rotated = u_val * mpmath.exp(mpmath.mpc(0, phi / 2))
  return mpmath.sqrt(1 - abs(mpmath.re(rotated)))


def _dist(u, phi):
  phi = mpmath.mpf(phi)
  rot = u.evaluate() * mpmath.exp(mpmath.mpc(0, phi / 2))
  return mpmath.sqrt(1 - abs(mpmath.re(rot)))


def rz_matrix(phi):
  phi_val = mpmath.mpf(phi)
  return mpmath.matrix([
    [mpmath.exp(mpmath.mpc(0, -phi_val / 2)), 0],
    [0, mpmath.exp(mpmath.mpc(0, phi_val / 2))],
  ])


def dist_trace_norm(U: ExactUnitary, phi):
  """Computes the trace-norm:
  d(R_z(phi), U) = sqrt(1 - (1/2) * |Tr(R_z(phi)^dagger * U)|)
  """
  Rz = rz_matrix(phi)
  U_mat = U.to_matrix()

  Rz_dagger = Rz.H

  prod = Rz_dagger * U_mat
  trace_val = prod[0, 0] + prod[1, 1]

  fidelity = abs(trace_val) / 2
  return mpmath.sqrt(mpmath.mpf(1) - fidelity)


if __name__ == "__main__":
  mpmath.mp.dps = 400
  phi = mpmath.pi / 2
  epsilon = mpmath.mpf(1e-20)
  seps = mpmath.mp.sqrt(epsilon)
  print("Sqrt epsilon: ", seps)
  while True:
    try:
      g = synthesize_z_rotation(phi, epsilon) * ExactUnitary.T() ** 5
      print("d(R_z, U[u,v,0] =", dist(g, phi))
      print("Trace Norm =", dist_trace_norm(g, phi))
      print("Is less that epsilon: ", epsilon >= dist(g, phi))
      print(f"Gate sequence: {len(exact_synthesize(g))}")
      actual = rz_matrix(phi)
      approx = (g).to_matrix()
      print("Is less that sqrt(epsilon):", dist(g, phi) <= seps)
      break
    except ValueError as e:
      print(f"Caught error: {e} -- retrying ...")
