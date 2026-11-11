import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jimgw.core.single_event.waveform import RippleIMRPhenomD, RippleIMRPhenomXAS

from conftest import LISA_PARAMS
from gwresponse import SpaceBased, check_waveform, sky_averaged_sensitivity, tdi_noise_psd, time_to_merger
from gwresponse.constants import C_SI, LISA_ARM_LENGTH
from gwresponse.noise import acceleration_noise, oms_noise
from jimgw.core.single_event.data import Data


def test_time_to_merger_is_the_stationary_phase_time_of_ripple():
    wf = RippleIMRPhenomD(f_ref=1e-4)
    f = jnp.linspace(1e-4, 0.03, 20001)
    _, dphase = jax.jvp(lambda x: wf.phase(x, LISA_PARAMS), (f,), (jnp.ones_like(f),))
    np.testing.assert_allclose(time_to_merger(f, LISA_PARAMS), -dphase / (2 * jnp.pi), rtol=1e-9, atol=1e-6)
    # Monotonic through inspiral and merger (up to 2 f_RD = 8.5 mHz for these parameters;
    # IMRPhenomD's ringdown phase makes t(f) turn over again near 20 mHz, where the
    # amplitude is exponentially small), and the same on a non-uniform grid
    t = time_to_merger(f, LISA_PARAMS)
    assert bool(jnp.all(jnp.diff(t)[f[1:] < 8.5e-3] > 0))
    g = jnp.geomspace(1e-4, 0.03, 300)
    _, dphase_g = jax.jvp(lambda x: wf.phase(x, LISA_PARAMS), (g,), (jnp.ones_like(g),))
    np.testing.assert_allclose(time_to_merger(g, LISA_PARAMS), -dphase_g / (2 * jnp.pi), rtol=1e-9, atol=1e-6)


def test_only_imrphenomd_is_supported():
    check_waveform(RippleIMRPhenomD())
    with pytest.raises(TypeError):
        check_waveform(RippleIMRPhenomXAS())


def test_tdi_noise_matches_michelson_cross_spectra():
    f = jnp.geomspace(1e-5, 0.05, 200)
    x = 2 * jnp.pi * f * LISA_ARM_LENGTH / C_SI
    s_acc, s_oms = acceleration_noise(f), oms_noise(f)
    s_xx = tdi_noise_psd(f, "X")
    # Cross spectrum of two Michelson variables for equal arms
    s_xy = -8 * jnp.sin(x) ** 2 * jnp.cos(x) * (s_oms + 4 * s_acc)
    np.testing.assert_allclose(tdi_noise_psd(f, "A"), s_xx - s_xy, rtol=1e-10)
    np.testing.assert_allclose(tdi_noise_psd(f, "E"), s_xx - s_xy, rtol=1e-10)
    # S_XX + 2 S_XY cancels to O(x^4) at low frequency, so compare where it is well conditioned
    well_conditioned = f > 1e-3
    np.testing.assert_allclose(
        tdi_noise_psd(f, "T")[well_conditioned], (s_xx + 2 * s_xy)[well_conditioned], rtol=1e-8
    )


def test_sky_averaged_sensitivity_matches_robson_et_al():
    """One channel, averaged over sky and polarization, is twice the two-channel curve.

    Robson et al.'s fit of the averaged response, 3/10 / (1 + 0.6 (f/f*)^2), is
    accurate to a few percent only well below f* = 19 mHz; at 10 mHz it differs from
    the exact response by ~7%.
    """
    f = jnp.geomspace(1e-4, 5e-3, 6)
    det = SpaceBased("LISA_A", channel="A")
    det.data = Data.from_td(jnp.zeros(16), 1.0, start_time=0.0, trigger_time=8.0)
    n = 4000
    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    ra = jax.random.uniform(k1, (n,)) * 2 * jnp.pi
    dec = jnp.arcsin(jax.random.uniform(k2, (n,)) * 2 - 1)
    psi = jax.random.uniform(k3, (n,)) * jnp.pi
    base = {"M_c": 1e6, "eta": 0.25, "s1_z": 0.0, "s2_z": 0.0, "t_c": 0.0}

    def response(ra, dec, psi):
        T = det.tdi_transfer(f, {**base, "ra": ra, "dec": dec, "psi": psi})
        return (jnp.abs(T["p"]) ** 2 + jnp.abs(T["c"]) ** 2) / 2

    R = jnp.mean(jax.vmap(response)(ra, dec, psi), axis=0)
    ratio = tdi_noise_psd(f, "A") / R / sky_averaged_sensitivity(f)
    np.testing.assert_allclose(ratio, 2.0, rtol=0.04)
