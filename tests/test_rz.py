from exact_synthesis.exactUnitary import ExactUnitary

import math

import pytest
from mpmath import mp, mpc, mpf

from exact_synthesis.synthesis import ExactFibonacciSynthesizer

_LOG10_INV_TAU = math.log10(2 / (math.sqrt(5) - 1))
_GUARD_DIGITS = 40


def dps_for(n_max: int) -> int:
  return max(50, int(math.ceil(_LOG10_INV_TAU * max(n_max, 1))) + _GUARD_DIGITS)


def tau_mpf():
  return (mp.sqrt(5) - 1) / 2


def phi_mpf():
  return (mp.sqrt(5) + 1) / 2


def _m_for_rz(eps):
  """Fig. 12, lines 1-2: C = sqrt(phi/4), m = ceil(log_tau(C*eps)) + 1."""
  C = mp.sqrt(phi_mpf() / 4)
  return int(mp.ceil(mp.log(C * eps) / mp.log(tau_mpf()))) + 1


def _get_u_v_k(phi_angle, eps):
  U = ExactFibonacciSynthesizer.synthesize_z_rotation(phi_angle, eps) * ExactUnitary.T()
  return U.u, U.v, U.k


# TODO
_RZ_SAMPLE_CASES = [
  (0.3, 0.05),
  (1.7, 0.01),
  (-2.9, 0.01),
  (mpf(mp.pi), 0.001),
]


@pytest.mark.parametrize("phi_val,eps_val", _RZ_SAMPLE_CASES)
def test_rz_sample_quality_bound(phi_val, eps_val):
  """d(R_z(phi), U[u,v,0]) = sqrt(1 - |Re(u*exp(i*phi/2))|) <= eps."""
  eps = mpf(eps_val)
  m = _m_for_rz(eps)

  with mp.workdps(dps_for(m)):
    phi_angle = mpf(phi_val)
    eps = mpf(eps_val)

    u, v, k = _get_u_v_k(phi_angle, eps)
    # assert k == 0, "k nonzero"
    u_val = u.evaluate()

    phase = mp.exp(mpc(0, phi_angle / 2))
    re_part = (u_val * phase).real
    quality_arg = 1 - abs(re_part)

    floating_slack = mpf("1e-30")
    assert quality_arg >= -floating_slack, f"1 - |Re(u*exp(i*phi/2))| = {quality_arg} < 0 for phi={phi_val}, eps={eps_val}"
    quality = mp.sqrt(max(quality_arg, mpf(0)))

    assert quality <= eps, f"rz_sample(phi={phi_val}, eps={eps_val}) failed the quality bound: d(R_z(phi), U[u,v,0]) = {quality} > eps = {eps}"


@pytest.mark.parametrize("phi_val,eps_val", _RZ_SAMPLE_CASES)
def test_rz_sample_norm_equation_holds(phi_val, eps_val):
  eps = mpf(eps_val)
  m = _m_for_rz(eps)

  with mp.workdps(dps_for(m)):
    phi_angle = mpf(phi_val)
    eps = mpf(eps_val)
    phi = phi_mpf()

    u, v, k = _get_u_v_k(phi_angle, eps)
    u_val, v_val = u.evaluate(), v.evaluate()

    lhs = abs(v_val) ** 2
    rhs = phi * (1 - abs(u_val) ** 2)

    tight_tol = mpf(10) ** (-(mp.dps - 15))
    assert abs(lhs - rhs) <= tight_tol, f"rz_sample(phi={phi_val}, eps={eps_val}) violated eq. (22): |v|^2={lhs} != phi*(1-|u|^2)={rhs} (diff {abs(lhs - rhs)}, tol {tight_tol})"


@pytest.mark.parametrize("phi_val,eps_val", _RZ_SAMPLE_CASES)
def test_rz_sample_u_is_a_valid_amplitude(phi_val, eps_val):
  eps = mpf(eps_val)
  m = _m_for_rz(eps)

  with mp.workdps(dps_for(m)):
    phi_angle = mpf(phi_val)
    eps = mpf(eps_val)

    u, v, k = _get_u_v_k(phi_angle, eps)
    u_norm = abs(u.evaluate())

    assert u_norm <= 1 + mpf("1e-30"), f"rz_sample(phi={phi_val}, eps={eps_val}) produced |u|={u_norm} > 1"
