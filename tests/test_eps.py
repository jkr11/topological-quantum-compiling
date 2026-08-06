import math

import pytest
from mpmath import mp, mpf

from exact_synthesis.prec import approx_real, random_sample

_LOG10_INV_TAU = math.log10(2 / (math.sqrt(5) - 1))
_GUARD_DIGITS = 40


def dps_for(n_max: int) -> int:
  return max(50, int(math.ceil(_LOG10_INV_TAU * max(n_max, 1))) + _GUARD_DIGITS)


def tau_mpf():
  return (mp.sqrt(5) - 1) / 2


def phi_mpf():
  return (mp.sqrt(5) + 1) / 2


def _m_and_N(eps, r):
  """Fig. 13:
      C = sqrt(phi/(4r))
      m = ceil(log_tau(C*eps*r)) + 1
      N = ceil(phi^m)
  """
  phi = phi_mpf()
  tau = tau_mpf()
  C = mp.sqrt(phi / (4 * r))
  m = int(mp.ceil(mp.log(C * eps * r) / mp.log(tau))) + 1
  N = int(mp.ceil(phi**m))
  return m, N


def _reference_bounds(theta, eps, r, phi_m):
  """
  y_min = r*phi^m*(sin(theta) - eps*(sqrt(4-eps^2)*cos(theta)
                                       + eps*sin(theta))/2)
  y_max = r*phi^m*(sin(theta) + eps*(sqrt(4-eps^2)*cos(theta)
                                       - eps*sin(theta))/2)
  x_max = r*phi^m*((1-eps^2/2)*cos(theta)
                    - eps*sqrt(1-eps^2/4)*sin(theta))
  x_c   = x_max - r*eps^2*phi^m/(4*cos(theta))
  """
  sqrt_4_minus_eps2 = mp.sqrt(4 - eps**2)
  y_min = r * phi_m * (mp.sin(theta) - eps * (sqrt_4_minus_eps2 * mp.cos(theta) + eps * mp.sin(theta)) / 2)
  y_max = r * phi_m * (mp.sin(theta) + eps * (sqrt_4_minus_eps2 * mp.cos(theta) - eps * mp.sin(theta)) / 2)
  x_max = r * phi_m * ((1 - eps**2 / 2) * mp.cos(theta) - eps * mp.sqrt(1 - eps**2 / 4) * mp.sin(theta))
  x_c = x_max - r * eps**2 * phi_m / (4 * mp.cos(theta))
  return y_min, y_max, x_c


@pytest.mark.parametrize("n", [2, 5, 10, 20, 40, 80])
@pytest.mark.parametrize("x_val", [0.0, 1.0 / 3, 1.0, -2.71828, 5.5, -0.000123, 3.14159265])
def test_approx_real_precision_bound(x_val, n):
  """Lemma 18: |x - (a + b*tau)| <= tau^(n-1) * (1 - tau^n)."""
  with mp.workdps(dps_for(n)):
    x = mpf(x_val)
    tau = tau_mpf()

    u = approx_real(x, n)
    approx = u.evaluate()
    bound = tau ** (n - 1) * (1 - tau**n)
    err = abs(x - approx)

    assert err <= bound, f"approx_real(x={x_val}, n={n}) violated the Lemma 18 precision bound: |x-(a+b*tau)| = {err} > {bound}"


@pytest.mark.parametrize("n", [2, 5, 10, 20, 40, 80])
@pytest.mark.parametrize("x_val", [0.0, 1.0 / 3, 1.0, -2.71828, 5.5, -0.000123])
def test_approx_real_b_coefficient_bound(x_val, n):
  """Lemma 18: |b| <= phi^n."""
  with mp.workdps(dps_for(n)):
    x = mpf(x_val)
    phi = phi_mpf()

    u = approx_real(x, n)
    _, b = u.a, u.b

    assert abs(b) <= phi**n, f"approx_real(x={x_val}, n={n}) produced |b|={b} exceeding phi^n = {phi**n}"


# TODO
_RANDOM_SAMPLE_CASES = [
  (1 / 20, 0.10, 1.0),
  (1 / 10, 0.05, 2.0),
  (3 / 20, 0.01, 5.0),
]


@pytest.mark.parametrize("theta_frac,eps_val,r_val", _RANDOM_SAMPLE_CASES)
def test_random_sample_y_coordinate_in_band(theta_frac, eps_val, r_val):
  eps, r = mpf(eps_val), mpf(r_val)
  m, N = _m_and_N(eps, r)

  with mp.workdps(dps_for(m)):
    theta = mpf(theta_frac) * mp.pi
    eps, r = mpf(eps_val), mpf(r_val)
    tau = tau_mpf()
    phi_m = phi_mpf() ** m

    y_min, y_max, _ = _reference_bounds(theta, eps, r, phi_m)
    delta_m = tau ** (m - 1) * (1 - tau**m)
    slack = delta_m * mp.sqrt(2 - tau)

    n_trials = 30
    for _ in range(n_trials):
      u0 = random_sample(theta, eps, r)
      y_coord = u0.evaluate().imag

      assert y_min - slack <= y_coord <= y_max + slack, (
        f"random_sample(theta={theta_frac}*pi, eps={eps_val}, r={r_val}) produced y-coordinate {y_coord}, outside [{y_min}, {y_max}] (+/- slack {slack})"
      )


@pytest.mark.parametrize("theta_frac,eps_val,r_val", _RANDOM_SAMPLE_CASES)
def test_random_sample_x_matches_tangent_line_construction(theta_frac, eps_val, r_val):
  eps, r = mpf(eps_val), mpf(r_val)
  m, N = _m_and_N(eps, r)

  with mp.workdps(dps_for(m)):
    theta = mpf(theta_frac) * mp.pi
    eps, r = mpf(eps_val), mpf(r_val)
    tau = tau_mpf()
    phi_m = phi_mpf() ** m

    y_min, _, x_c = _reference_bounds(theta, eps, r, phi_m)
    delta_m = tau ** (m - 1) * (1 - tau**m)
    tan_theta = mp.tan(theta)

    n_trials = 30
    for _ in range(n_trials):
      u0 = random_sample(theta, eps, r)
      z = u0.evaluate()
      y_hat = z.imag  # == sqrt(2-tau)*(a_y + b_y*tau), exactly, by construction

      x_target = x_c - (y_hat - y_min) * tan_theta

      assert abs(z.real - x_target) <= delta_m, (
        f"random_sample(theta={theta_frac}*pi, eps={eps_val}, r={r_val}) produced x-coordinate {z.real}, expected APPROX-REAL(x, m) of {x_target} (+/- {delta_m})"
      )


import mpmath as mp
from exact_synthesis.prec import _random_sample


def calculate_parallelogram_bounds(theta, epsilon, r, m):
  theta_mp = mp.mpf(theta)
  eps_mp = mp.mpf(epsilon)
  r_mp = mp.mpf(r)

  phi = (1 + mp.sqrt(5)) / 2
  tau = phi - 1

  sin_t, cos_t = mp.sin(theta_mp), mp.cos(theta_mp)
  sqrt4_eps2 = mp.sqrt(4 - eps_mp**2)

  y_min = r_mp * phi**m * (sin_t - eps_mp * (sqrt4_eps2 * cos_t + eps_mp * sin_t) / 2)
  y_max = r_mp * phi**m * (sin_t + eps_mp * (sqrt4_eps2 * cos_t - eps_mp * sin_t) / 2)

  x_max = r_mp * phi**m * ((1 - eps_mp**2 / 2) * cos_t - eps_mp * mp.sqrt(1 - eps_mp**2 / 4) * sin_t)
  x_min = x_max - r_mp * phi**m * eps_mp * sqrt4_eps2 * sin_t

  return y_min, y_max, x_min, x_max


@pytest.mark.parametrize("theta", [0.1, 0.3, mp.pi / 6])
@pytest.mark.parametrize("epsilon", [0.05, 0.01])
@pytest.mark.parametrize("r", [1.0, 2.0])
def test_random_sample_inside_parallelogram(theta, epsilon, r):
  mp.mp.dps = 50

  sample_zomega, ayby, y  = _random_sample(theta, epsilon, r)

  phi = (1 + mp.sqrt(5)) / 2
  tau = phi - 1
  C = mp.sqrt(phi / (4 * r))
  m = int(mp.ceil(mp.log(C * epsilon * r) / mp.log(tau))) + 1

  y_min, y_max, x_min, x_max = calculate_parallelogram_bounds(theta, epsilon, r, m)

  
  eval_val = sample_zomega.evaluate()
  x_val = mp.re(eval_val)
  y_val = mp.im(eval_val)  # / mp.sqrt(2 - tau)  


  assert x_min <= x_val <= x_max, f"x value {x_val} outside [{x_min}, {x_max}]"

  assert y_min <= y_val <= y_max, f"y value {y_val} outside [{y_min}, {y_max}]"


@pytest.mark.parametrize("theta", [0.1, 0.3, mp.pi / 6])
@pytest.mark.parametrize("epsilon", [0.05, 0.01])
@pytest.mark.parametrize("r", [1.0, 2.0])
def test_bound_19(theta, epsilon, r):
  mp.mp.dps = 100
  sample_zomega, ab, y = _random_sample(theta, epsilon, r)

  tau = (mp.sqrt(5) - 1) / 2

  sqrt_2_minus_tau = mp.sqrt(2 - tau)

  assert abs(y - ab.evaluate()) * sqrt_2_minus_tau 
