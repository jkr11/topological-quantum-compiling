import unittest
import numpy as np
from numpy.testing import assert_almost_equal

from single_qubit.exact_synthesis.rings import Cyclotomic10, ZTau
from single_qubit.exact_synthesis.rings import *
from single_qubit.exact_synthesis.numberTheory import solve_norm_equation, N_i, EASY_FACTOR
from single_qubit.exact_synthesis.exactUnitary import ExactUnitary
from single_qubit.exact_synthesis.exactUnitary import *

# TODO: remove these
Nomega = np.exp(1j * np.pi / 5)

# Define the conjugate golden ratio τ = (sqrt(5) - 1)/2
Ntau = (np.sqrt(5) - 1) / 2
sqrt_tau = np.sqrt(Ntau)

# Define T gate
Tnp = np.array([[1, 0], [0, Nomega]], dtype=complex)

# Define F matrix
Fnp = np.array([[Ntau, sqrt_tau], [sqrt_tau, -Ntau]], dtype=complex)

NOmegaI = np.array([[Nomega, 0], [0, Nomega]], dtype=complex)

# Define σ1 and σ2
sigma1 = np.array([[Nomega**6, 0], [0, Nomega**13]], dtype=complex)

sigma2 = Fnp @ sigma1 @ Fnp.T


class TestGaussComplexity(unittest.TestCase):
  def test_proposition_4_a(self):
    for i in range(10):
      omega = Cyclotomic10.Omega_(i)
      self.assertEqual(omega.gauss_complexity().evaluate(), 2)

  def test_zero(self):
    zero = Cyclotomic10(0, 0, 0, 0)
    self.assertEqual(zero.gauss_complexity().evaluate(), 0)

  def test_c(self):
    c = Cyclotomic10(1, 2, 3, 4)
    self.assertTrue(c.gauss_complexity().evaluate() >= 3)

  def test_upper_bound(self):
    num = Cyclotomic10(1, 2, 3, 4)
    bound = 5 / 2 * (1 + 4 + 9 + 16)
    self.assertTrue(num.gauss_complexity().evaluate() <= bound, f"Expected {num.gauss_complexity().evaluate()} <= {bound}")


class TestEasyFactor(unittest.TestCase):
  def test_paper_example(self):
    solution = [(2, 2), (5, 1), (ZTau(2, -1), 1), (ZTau(15, -8), 1)]
    xi = ZTau(760, -780)
    EF = EASY_FACTOR(xi)
    self.assertEqual(EF, solution)


class TestNormEquation(unittest.TestCase):
  def test_paper_example(self):
    xi = ZTau(760, -780)
    x = solve_norm_equation(xi)
    self.assertEqual(N_i(x), xi)


class TestCyclotomic10(unittest.TestCase):
  def test_automorphism(self):
    tau = Cyclotomic10.Tau()
    phi = Cyclotomic10.Tau() + Cyclotomic10.One()
    self.assertEqual(tau.automorphism(), -phi)

  def test_omega_4th_power(self):
    omega = Cyclotomic10(0, 1, 0, 0)
    self.assertEqual(omega**4, Cyclotomic10(-1, 1, -1, 1))

  def test_division_example(self):
    y = Cyclotomic10(3, 2, -7, 7)
    yy = N_i(y)
    print("yy: ", yy)
    x = Cyclotomic10(15, 0, -8, 8) // yy.to_cycl()
    self.assertEqual(x.to_subring(), ZTau(5, 3))


class TestExactUnitary(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.omega = Cyclotomic10(0, 1, 0, 0)
    cls.tau = cls.omega**2 - cls.omega**3

    # Define T = U[1, 0, 6] (ω^6 = -ω)
    cls.T = ExactUnitary(Cyclotomic10(1, 0, 0, 0), Cyclotomic10(0, 0, 0, 0), 6)

    # Define F = U[τ, 1, 0]
    cls.F = ExactUnitary(cls.tau, Cyclotomic10(1, 0, 0, 0), 0)

    # Identity unitary (u=1, v=0, k=0)
    cls.I = ExactUnitary(Cyclotomic10(1, 0, 0, 0), Cyclotomic10(0, 0, 0, 0), 0)

  def test_rings(self):
    test = Cyclotomic10(0, -1, 0, 0)
    result = self.omega**6
    self.assertEqual(result, test)
    Tau = Cyclotomic10.Tau()
    self.assertEqual(Tau.evaluate(), Ntau)
    self.assertEqual(Cyclotomic10.One().evaluate(), 1)
    self.assertEqual(Cyclotomic10.One().conjugate().evaluate(), 1)
    self.assertEqual(Cyclotomic10.Tau().automorphism().evaluate(), -Ntau - 1)

  def test_F_squared_eq_one(self):
    tt = ExactUnitary.F()
    test = tt**2
    self.assertEqual(test, ExactUnitary.I())

  def test_T_tenth_power_eq_one(self):
    tt = ExactUnitary.T()
    test = tt**10
    self.assertEqual(test, ExactUnitary.I())

  def test_omega_10th_power(self):
    o = Cyclotomic10(0, 1, 0, 0)
    test = o**10
    self.assertEqual(test, Cyclotomic10.One())

  def test_identity_multiplication(self):
    # I * I = I
    T = ExactUnitary.T()
    I = T**10
    result = I * I
    self.assertEqual(result.u, Cyclotomic10(1, 0, 0, 0))
    self.assertEqual(result.v, Cyclotomic10(0, 0, 0, 0))
    self.assertEqual(result.k, 5)

    # I * T = T
    result = I * T
    self.assertEqual(result.u, T.u)
    self.assertEqual(result.v, T.v)
    self.assertEqual(result.k, T.k)

    result = T * I
    self.assertEqual(result.u, T.u)
    self.assertEqual(result.v, T.v)
    self.assertEqual(result.k, T.k)

  def test_T_multiplication(self):
    result = self.T * self.T

    # Verify matrix representation ω^6 * ω^6 = ω^12 = ω^2
    omega_6 = self.omega**6
    omega_12 = (omega_6 * omega_6).evaluate()
    omega_2 = ((self.omega) ** 2).evaluate()
    assert_almost_equal(omega_12, omega_2, decimal=6)

  def test_gauss_complexity(self):
    res = [
      2,
      3,
      13,
    ]  # I think the paper is wrong here, the first one needs to be 2. There is also a different definition in the shorter version of the article
    comp = [Cyclotomic10(1, 0, 0, 0), Cyclotomic10(0, 0, 1, -1), Cyclotomic10(2, -1, 0, 1)]
    for i in range(0, 3):
      self.assertEqual(comp[i].gauss_complexity().evaluate().real, res[i])

  def test_gauss_complexity_from_FT(self):
    res = [2, 3, 13]
    for i in range(0, 3):
      U = (self.F * self.T) ** i
      print(f"(FT)^{i} : ", U)

      self.assertEqual(U.gauss_complexity(), res[i], f"failed at (FT)^{i}")

  def test_powers(self):
    T = ExactUnitary.T()
    self.assertEqual(T**10, ExactUnitary.I(), "T^10 should be identity")

  def test_braidwords(self):
    I = ExactUnitary.I()
    T = ExactUnitary.T()
    F = ExactUnitary.F()
    I6 = I.omega_mul(6)
    T7 = T**7
    sigma1_lhs = I6 * T7

    sigma2_lhs = I6 * F * T7 * F

    I2 = I.omega_mul(2)
    S3 = sigma1_lhs**3
    self.assertEqual(T, I2 * S3)

    I4 = I.omega_mul(4)
    S121 = sigma1_lhs * sigma2_lhs * sigma1_lhs

    self.assertEqual(F, I4 * S121)

  def test_conjugate(self):
    Fc = ExactUnitary.F().conjugate().transpose()
    self.assertEqual(Fc, ExactUnitary.F())

    sigma1ct = ExactUnitary.sigma1().transpose()
    self.assertEqual(sigma1ct, ExactUnitary.sigma1())


if __name__ == "__main__":
  unittest.main()
