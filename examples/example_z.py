import mpmath
from exact_synthesis.synthesis import ExactFibonacciSynthesizer, d_z
from exact_synthesis.util import Gates, trace_norm
from exact_synthesis.exactUnitary import ExactUnitary

if __name__ == "__main__":
  mpmath.mp.dps = 400
  phi = 4 * mpmath.pi / 10
  epsilon = mpmath.mpf(1e-70)

  while True:
    try:
      g = ExactFibonacciSynthesizer.synthesize_z_rotation(phi, epsilon) * ExactUnitary.T() ** 5
      print(d_z(phi, g))
      print(epsilon >= d_z(phi, g))
      # print(epsilon)
      print(f"Gate sequence: {len(ExactFibonacciSynthesizer._exact_synthesize(g))}")
      actual = Gates.Rz(float(phi))
      approx = (g).to_numpy()
      print("Trace norm: ", trace_norm(actual, approx))
      print(f"Approximation: {approx}")
      print("Actual matrix: ", actual)
      print(f"Difference total: {approx - actual}")
      break
    except ValueError as e:
      print(f"Caught error: {e} -- retrying ...")
