import mpmath

from exact_synthesis.synthesis import (
  synthesize_z_rotation,
  exact_synthesize,
)
from exact_synthesis.util import d_z, d_trace, rz_mp


def main():
  mpmath.mp.dps = 4 * 8 + 60  # TODO make a func for this, make TAU and w instance time functions, these should be cachable as well. Maybe for sqrts as well?

  phi = mpmath.rand()
  epsilon = mpmath.mpf("1e-8")
  sqrt_epsilon = mpmath.sqrt(epsilon)

  print("Phi:", phi)
  print("Epsilon:", epsilon)
  print("Sqrt epsilon:", sqrt_epsilon)

  raw = synthesize_z_rotation(phi, epsilon)

  raw_dz = d_z(raw, phi)

  raw_trace = d_trace(raw, rz_mp(phi))

  print(raw.to_matrix())

  print("Algebraic Z distance:", raw_dz)
  print("Trace distance:", raw_trace)
  print("Pass epsilon:", raw_dz <= epsilon)
  gates = exact_synthesize(raw)
  print(gates)
  print(len(gates))


if __name__ == "__main__":
  main()
