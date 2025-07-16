import argparse
import mpmath
from single_qubit.exact_synthesis.prec import synthesize_z_rotation, synthesize_zx_rotation


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("theta", type=str)
  parser.add_argument("epsilon", type=str)
  parser.add_argument("--zx", action="store_true", help="Use ZX synthesis instead of Z synthesis")
  parser.add_argument("--precision", type=int, default=100, help="Precision for mpmath (default: 100)")
  parser.add_argument("--draw", action="store_true", help="Draw the circuit")
  parser.add_argument("--braids", action="store_true", help="draw braids")

  args = parser.parse_args()
  mpmath.mp.dps = args.precision
  mpmath.mp.pretty = True
  theta = mpmath.mpmathify(args.theta)
  epsilon = mpmath.mpmathify(args.epsilon)

  gates = []
  if args.zx:
    gates = synthesize_zx_rotation(theta, epsilon)
    print(f"ZX-rotation circuit for φ = {theta}:")
  else:
    gates = synthesize_z_rotation(theta, epsilon)
    print(f"Z-rotation circuit for φ = {theta}:")
  print(f"Number of gates: {len(gates)}")
  print(f"Gate sequence: {gates}")


if __name__ == "__main__":
  main()
