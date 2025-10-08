import mpmath
from exact_synthesis.synthesis import ExactFibonacciSynthesizer
from exact_synthesis.synthesis import d_z
from exact_synthesis.exactUnitary import ExactUnitary
from exact_synthesis.util import Gates

if __name__ == "__main__":
  mpmath.mp.dps = 400
  phi = float(4 * mpmath.pi / 1000)
  epsilon = 1e-70

  while True:
    try:
      g = ExactFibonacciSynthesizer.synthesize_z_rotation(phi, epsilon)
      print(d_z(phi, g))

      actual = Gates.Rz(phi)
      approx = (g * ExactUnitary.T() ** 5).to_numpy(1e-10)
      
      print(f"Approximation: {approx}",)
      print("Actual matrix: ", actual)
      print(f"Difference total: {approx - actual}")
      break
    except ValueError as e:
      print(f"Caught error: {e} -- retrying ...")
