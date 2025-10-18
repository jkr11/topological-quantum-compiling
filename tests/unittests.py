import unittest
import numpy as np
from numpy.testing import assert_almost_equal

from exact_synthesis.rings import ZOmega, ZTau
from exact_synthesis.numberTheory import solve_norm_equation, N_i, easy_factor
from exact_synthesis.exactUnitary import ExactUnitary
from exact_synthesis.util import euler_angles, matrix_of_euler_angles, haar_random_su2
from exact_synthesis.prec import random_sample

class TestRandomSample(unittest.TestCase):
  def test_random_sample(self):
    theta = 1/10
    epsilon = 1e-10
    r = 1.0
    appr = random_sample(theta, epsilon, r)
    

class TestGaussComplexity(unittest.TestCase):
  def test_proposition_4_a(self):
    for i in range(10):
      omega = ZOmega.Omega_(i)
      self.assertEqual(omega.gauss_complexity().evaluate(), 2)

  def test_zero(self):
    zero = ZOmega(0, 0, 0, 0)
    self.assertEqual(zero.gauss_complexity().evaluate(), 0)

  def test_c(self):
    c = ZOmega(1, 2, 3, 4)
    self.assertTrue(c.gauss_complexity().evaluate() >= 3)

  def test_upper_bound(self):
    num = ZOmega(1, 2, 3, 4)
    bound = 5 / 2 * (1 + 4 + 9 + 16)
    self.assertTrue(num.gauss_complexity().evaluate() <= bound, f"Expected {num.gauss_complexity().evaluate()} <= {bound}")


class TestEasyFactor(unittest.TestCase):
  def test_paper_example(self):
    solution = [(2, 2), (5, 1), (ZTau(2, -1), 1), (ZTau(15, -8), 1)]
    xi = ZTau(760, -780)
    EF = easy_factor(xi)
    self.assertEqual(EF, solution)


class TestNormEquation(unittest.TestCase):
  def test_paper_example(self):
    xi = ZTau(760, -780)
    x = solve_norm_equation(xi)
    self.assertEqual(N_i(x), xi)


class TestCyclotomic10(unittest.TestCase):
  def test_automorphism(self):
    tau = ZOmega.Tau()
    phi = ZOmega.Tau() + ZOmega.One()
    self.assertEqual(tau.automorphism(), -phi)

  def test_omega_4th_power(self):
    omega = ZOmega(0, 1, 0, 0)
    self.assertEqual(omega**4, ZOmega(-1, 1, -1, 1))

  def test_division_example(self):
    y = ZOmega(3, 2, -7, 7)
    yy = N_i(y)
    x = ZOmega(15, 0, -8, 8) // yy.to_cycl()
    self.assertEqual(x.to_subring(), ZTau(5, 3))

  def test_divmod(self):
    a = ZOmega(3, 2, -1, 0)
    b = ZOmega(1, 0, 1, 1)

    q, r = divmod(a, b)
    lhs = b * q + r
    self.assertTrue(lhs == a)


class TestExactUnitary(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.omega = ZOmega(0, 1, 0, 0)
    cls.tau = cls.omega**2 - cls.omega**3

    cls.T = ExactUnitary(ZOmega(1, 0, 0, 0), ZOmega(0, 0, 0, 0), 6)
    cls.F = ExactUnitary(cls.tau, ZOmega(1, 0, 0, 0), 0)
    cls.I = ExactUnitary(ZOmega(1, 0, 0, 0), ZOmega(0, 0, 0, 0), 0)

  def test_F_squared_eq_one(self):
    tt = ExactUnitary.F()
    test = tt**2
    self.assertEqual(test, ExactUnitary.I())

  def test_T_tenth_power_eq_one(self):
    tt = ExactUnitary.T()
    test = tt**10
    self.assertEqual(test, ExactUnitary.I())

  def test_omega_10th_power(self):
    o = ZOmega(0, 1, 0, 0)
    test = o**10
    self.assertEqual(test, ZOmega.One())

  def test_identity_multiplication(self):
    # I * I = I
    T = ExactUnitary.T()
    I = T**10
    result = I * I
    self.assertEqual(result.u, ZOmega(1, 0, 0, 0))
    self.assertEqual(result.v, ZOmega(0, 0, 0, 0))
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
    ]  # I think the table in the paper is wrong here, the first n needs to be 2. Check the released version for this
    comp = [ZOmega(1, 0, 0, 0), ZOmega(0, 0, 1, -1), ZOmega(2, -1, 0, 1)]
    for i in range(0, 3):
      self.assertEqual(comp[i].gauss_complexity().evaluate().real, res[i])

  def test_gauss_complexity_from_FT(self):
    res = [2, 3, 13]
    for i in range(0, 3):
      U = (self.F * self.T) ** i

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


class TestEulerAngles(unittest.TestCase):
  def test_euler_angles(self):
    U = haar_random_su2()
    a, b, c, d = euler_angles(U)
    rU = matrix_of_euler_angles((a, b, c, d))
    self.assertTrue(np.allclose(U, rU))


if __name__ == "__main__":
  unittest.main()
