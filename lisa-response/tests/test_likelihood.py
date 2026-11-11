import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jimgw.core.single_event.detector import get_H1, get_L1
from jimgw.core.single_event.likelihood import HeterodynedTransientLikelihoodFD, TransientLikelihoodFD
from jimgw.core.single_event.waveform import RippleIMRPhenomXAS

from conftest import GPS, LISA_DATA, LISA_PARAMS, make_lisa
from gwresponse import MultibandLikelihood

GROUND_PARAMS = {**LISA_PARAMS, "M_c": 30.0, "d_L": 800.0, "t_c": 0.01}


def make_ground(waveform, params=GROUND_PARAMS, trigger_time=GPS):
    ifos = [get_H1(), get_L1()]
    for ifo in ifos:
        ifo.load_and_set_psd()
        ifo.set_zero_noise(duration=4.0, sampling_frequency=2048.0, trigger_time=trigger_time, f_min=20.0, f_max=1024.0)
        ifo.inject_signal(waveform, params)
    return ifos


def test_lisa_likelihood_at_injection_is_half_the_snr_squared(waveform):
    ifos = make_lisa(waveform)
    likelihood = TransientLikelihoodFD(ifos, waveform, f_min=LISA_DATA["f_min"], f_max=LISA_DATA["f_max"])
    snr2 = 0.0
    for ifo in ifos:
        frequencies, data, psd = ifo.slice_to_frequency_range(LISA_DATA["f_min"], LISA_DATA["f_max"])
        snr2 += 4 * jnp.sum(jnp.abs(data) ** 2 / psd) / ifo.data.duration
    np.testing.assert_allclose(likelihood.evaluate(LISA_PARAMS), snr2 / 2, rtol=1e-6)
    assert float(snr2) > 100**2
    # The injection is the maximum: zero gradient in every parameter
    grad = jax.grad(likelihood.evaluate)(LISA_PARAMS)
    assert likelihood.evaluate({**LISA_PARAMS, "t_c": 50.5}) < likelihood.evaluate(LISA_PARAMS)
    assert abs(float(grad["t_c"])) < 1e-3 and abs(float(grad["ra"])) < 1e-2


def test_heterodyned_lisa_likelihood_matches_full(waveform):
    ifos = make_lisa(waveform, noise_key=jax.random.key(2))
    kwargs = {"f_min": LISA_DATA["f_min"], "f_max": LISA_DATA["f_max"]}
    full = TransientLikelihoodFD(ifos, waveform, **kwargs)
    het = HeterodynedTransientLikelihoodFD(ifos, waveform, reference_parameters=LISA_PARAMS, n_bins=200, **kwargs)
    for shift in ({}, {"t_c": 50.2}, {"M_c": 4.0004e5}, {"ra": 1.002, "psi": 0.71}, {"d_L": 3.1e4}):
        params = {**LISA_PARAMS, **shift}
        np.testing.assert_allclose(het.evaluate(params), full.evaluate(params), rtol=1e-6, atol=1e-2)


def test_multiband_likelihood_adds_lisa_and_ground(waveform):
    lisa = TransientLikelihoodFD(make_lisa(waveform, GROUND_PARAMS | {"M_c": 4e5, "d_L": 3e4}), waveform)
    ground = TransientLikelihoodFD(make_ground(waveform), waveform)
    joint = MultibandLikelihood([lisa, ground])
    params = {**GROUND_PARAMS, "M_c": 30.0}
    np.testing.assert_allclose(joint.evaluate(params), lisa.evaluate(params) + ground.evaluate(params))
    assert joint.detector_names == ["LISA_A", "LISA_E", "H1", "L1"]
    np.testing.assert_allclose(jax.jit(joint.evaluate)(params), joint.evaluate(params))


def test_multiband_likelihood_validation(waveform):
    lisa_ifos = make_lisa(waveform)
    with pytest.raises(ValueError, match="marginalized"):
        MultibandLikelihood([
            TransientLikelihoodFD(lisa_ifos, waveform, phase_marginalization=True),
            TransientLikelihoodFD(make_ground(waveform), waveform),
        ])
    with pytest.raises(ValueError, match="trigger time"):
        MultibandLikelihood([
            TransientLikelihoodFD(lisa_ifos, waveform),
            TransientLikelihoodFD(make_ground(waveform, trigger_time=GPS + 1), waveform),
        ])
    with pytest.raises(TypeError, match="IMRPhenomD"):
        MultibandLikelihood([TransientLikelihoodFD(lisa_ifos, RippleIMRPhenomXAS(f_ref=3e-3))])
    with pytest.raises(TypeError, match="IMRPhenomD"):
        make_lisa(RippleIMRPhenomXAS(f_ref=3e-3))


def test_jim_samples_the_joint_likelihood(waveform):
    """End-to-end smoke test: Jim's priors, transforms and flowMC on LISA + ground data."""
    from jimgw.core.jim import Jim
    from jimgw.core.prior import CombinePrior, UniformPrior
    from jimgw.core.single_event.helpers import get_default_likelihood_transforms
    from jimgw.samplers.config import FlowMCConfig

    params = {**GROUND_PARAMS, "M_c": 4e5, "d_L": 3e4}
    fixed = {k: v for k, v in params.items() if k not in ("M_c", "t_c")}
    lisa = TransientLikelihoodFD(make_lisa(waveform, params), waveform, fixed_parameters=fixed)
    # A million solar mass binary is silent in ground-based detectors: noise-free empty data
    ground_ifos = [get_H1(), get_L1()]
    for ifo in ground_ifos:
        ifo.load_and_set_psd()
        ifo.set_zero_noise(duration=4.0, sampling_frequency=2048.0, trigger_time=GPS, f_min=20.0, f_max=1024.0)
    ground = TransientLikelihoodFD(ground_ifos, waveform, fixed_parameters=fixed)
    prior = CombinePrior([
        UniformPrior(3.999e5, 4.001e5, parameter_names=["M_c"]),
        UniformPrior(-0.5, 0.5, parameter_names=["t_c"]),
    ])
    jim = Jim(
        MultibandLikelihood([lisa, ground]),
        prior,
        likelihood_transforms=get_default_likelihood_transforms(prior, [], waveform),
        sampler_config=FlowMCConfig(
            n_chains=20, n_local_steps=5, n_global_steps=5, n_training_loops=2,
            n_production_loops=2, n_epochs=2, global_thinning=1, batch_size=100,
        ),
    )
    jim.sample()
    samples = jim.get_samples()
    assert np.all(np.isfinite(samples["M_c"])) and samples["M_c"].size > 0
