import unittest
import numpy as np
from numpy.testing import assert_almost_equal
from rings import *
from exactUnitary import *

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

  def test_defs(self):
    self.assertTrue(np.allclose(ExactUnitary.F().to_matrix(), Fnp))
    self.assertTrue(np.allclose(ExactUnitary.T().to_matrix(), Tnp))

  def test_rings(self):
    test = Cyclotomic10(0, -1, 0, 0)
    result = self.omega**6
    self.assertEqual(result, test)
    Tau = Cyclotomic10.Tau()
    self.assertEqual(Tau.evaluate(), Ntau)
    self.assertEqual(Cyclotomic10.One().evaluate(), 1)
    self.assertEqual(Cyclotomic10.One().conjugate().evaluate(), 1)
    self.assertEqual(Cyclotomic10.Tau().automorphism().evaluate(), -Ntau - 1)

  def test_global_phase(self):
    I = ExactUnitary.I()
    self.assertTrue(
        np.allclose(I.omega_mul(6).to_matrix(), Nomega**6 * np.eye(2)))
    T = ExactUnitary.T()
    self.assertTrue(
        np.allclose((I.omega_mul(6) * (T**7)).to_matrix(),
                    Nomega**6 * np.eye(2) @ Tnp**7))

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
    omega_2 = ((self.omega)**2).evaluate()
    assert_almost_equal(omega_12, omega_2, decimal=6)

  def test_gauss_complexity(self):
    res = [
        2, 3, 13
    ]  # I think the paper is wrong here, the first one needs to be 2. There is also a different definition in the shorter version of the article
    comp = [
        Cyclotomic10(1, 0, 0, 0),
        Cyclotomic10(0, 0, 1, -1),
        Cyclotomic10(2, -1, 0, 1)
    ]
    for i in range(0, 3):
      self.assertEqual(comp[i].gauss_complexity().evaluate().real, res[i])

  def test_gauss_complexity_from_FT(self):
    res = [2, 3, 13]
    for i in range(0, 3):
      U = (self.F * self.T)**i
      print(f"(FT)^{i} : ", U)

      self.assertEqual(U.gauss_complexity(), res[i], f"failed at (FT)^{i}")

  def test_powers(self):
    T = ExactUnitary.T()
    T2 = T**2
    T2np = Tnp**2
    self.assertTrue(np.allclose(T2.to_matrix(), T2np))
    self.assertTrue(np.allclose((T**7).to_matrix(), Tnp**7))

  def test_braidwords(self):
    I = ExactUnitary.I()
    T = ExactUnitary.T()
    F = ExactUnitary.F()
    I6 = I.omega_mul(6)
    T7 = T**7
    sigma1_lhs = I6 * T7
    self.assertTrue(np.allclose(sigma1_lhs.to_matrix(), sigma1))

    sigma2_lhs = I6 * F * T7 * F
    self.assertTrue(np.allclose(sigma2_lhs.to_matrix(), sigma2))

    I2 = I.omega_mul(2)
    S3 = sigma1_lhs**3
    self.assertEqual(T, I2 * S3)

    I4 = I.omega_mul(4)
    S121 = sigma1_lhs * sigma2_lhs * sigma1_lhs
    #print("F: ", F)
    #print("(w^4)s1s2s1: ", I4 * S121)
    self.assertTrue(np.allclose(
        F.to_matrix(),
        (I4 * S121
         ).to_matrix()))  # TODO there is an error here with the multiplication


if __name__ == "__main__":
  unittest.main()
