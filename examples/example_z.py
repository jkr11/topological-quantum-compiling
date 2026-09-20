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
  """
  d(R_z(phi), U) = sqrt(1 - (1/2) * |Tr(R_z(phi)^T * U)|)
  """
  Rz = rz_matrix(phi)
  U_mat = U.to_matrix()

  Rz_dagger = Rz.H

  prod = Rz_dagger * U_mat
  trace_val = prod[0, 0] + prod[1, 1]

  fidelity = abs(trace_val) / 2
  return mpmath.sqrt(mpmath.mpf(1) - fidelity)


if __name__ == "__main__":
  mpmath.mp.dps = 200
  # phi = mpmath.pi / 2
  phi = mpmath.mpf(2.028610148604752)

  epsilon = mpmath.mpf(1e-40)/3
  seps = mpmath.mp.sqrt(epsilon)
  print("Sqrt epsilon: ", seps)
  while True: # Also not sure why this should be the case.
    try:
      g = synthesize_z_rotation(phi, epsilon) * ExactUnitary.T()**5 # This seems to be necessary to get trace norm to work
      print("d(R_z, U[u,v,0] =", dist(g, phi))
      print("Trace Norm =", dist_trace_norm(g, phi)) # THis is not actually equal without the T**5 adddition.
      # However it is when we do 1-trace_norm then we get <= epsilon. Strange.  
      print("Is less that epsilon: ", epsilon >= dist(g, phi)) 
      print(f"Gate sequence: {len(exact_synthesize(g))}")
      actual = rz_matrix(phi)
      approx = (g).to_matrix()
      print("Is less that sqrt(epsilon):", dist(g, phi) <= seps)
      print("d(g,phi) - sqrt(eps)", dist(g,phi) - seps)
      print("log(1/eps): ", mpmath.log(1/epsilon))
      break
    except ValueError as e:
      print(f"Caught error: {e} -- retrying ...")
