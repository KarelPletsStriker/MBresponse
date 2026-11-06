"""Joint LISA + LIGO/Virgo parameter estimation of a multiband binary black hole.

LISA observes the inspiral weeks before merger at ~0.03 Hz, the ground-based
network observes the merger. Each band keeps its own data (with very different
durations and sampling rates) in its own Jim likelihood, and
``MultibandLikelihood`` adds them. All bands share the same source parameters,
including the geocentric ``t_c``, ``ra``, ``dec`` and ``psi``.

    python examples/joint_lisa_ground.py           # production settings
    python examples/joint_lisa_ground.py --quick   # smoke test
"""

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from jimgw.core.jim import Jim
from jimgw.core.prior import CombinePrior, CosinePrior, PowerLawPrior, SinePrior, UniformPrior
from jimgw.core.single_event.detector import get_H1, get_L1, get_V1
from jimgw.core.single_event.helpers import (
    get_default_likelihood_transforms,
    get_default_periodic,
    get_default_sample_transforms,
)
from jimgw.core.single_event.likelihood import HeterodynedTransientLikelihoodFD
from jimgw.core.single_event.waveform import RippleIMRPhenomD
from jimgw.samplers.config import FlowMCConfig

from gwresponse import MultibandLikelihood, get_LISA

parser = argparse.ArgumentParser()
parser.add_argument("--quick", action="store_true", help="tiny sampler settings")
args = parser.parse_args()

waveform = RippleIMRPhenomD(f_ref=20.0)
gps = 1.4e9
injection_parameters = {
    "M_c": 150.0,
    "eta": 0.24,
    "s1_z": 0.3,
    "s2_z": -0.1,
    "d_L": 300.0,
    "iota": 0.5,
    "phase_c": 0.3,
    "ra": 1.0,
    "dec": 0.3,
    "psi": 0.7,
    "t_c": 0.01,
}
noise_key = jax.random.key(1)

# ----------------------------------------------------------------------
# LISA band: ~48 days of data at 0.125 Hz, ending shortly after the merger
# ----------------------------------------------------------------------

lisa_f_min, lisa_f_max = 0.025, 0.058
lisa = get_LISA("AE", trigger_time=gps)
for ifo in lisa:
    ifo.load_and_set_psd()
    ifo.simulate_and_set_noise(
        duration=2**22, sampling_frequency=0.125, trigger_time=gps,
        f_min=lisa_f_min, f_max=lisa_f_max, post_trigger_duration=1e3, rng_key=noise_key,
    )
    ifo.inject_signal(waveform, injection_parameters)

# ----------------------------------------------------------------------
# Ground band: 8 s of data at 2048 Hz around the merger
# ----------------------------------------------------------------------

ground_f_min, ground_f_max = 10.0, 1024.0
ground = [get_H1(), get_L1(), get_V1()]
for ifo in ground:
    ifo.load_and_set_psd()
    ifo.simulate_and_set_noise(
        duration=8.0, sampling_frequency=2048.0, trigger_time=gps,
        f_min=ground_f_min, f_max=ground_f_max, rng_key=noise_key,
    )
    ifo.inject_signal(waveform, injection_parameters)

# ----------------------------------------------------------------------
# Prior and transforms. The ground-based network fixes the detector-frame
# time and sky parametrizations, which LISA reads back as t_c, ra and dec.
# ----------------------------------------------------------------------

prior = CombinePrior(
    [
        UniformPrior(140.0, 160.0, parameter_names=["M_c"]),
        UniformPrior(0.3, 1.0, parameter_names=["q"]),
        UniformPrior(-0.99, 0.99, parameter_names=["s1_z"]),
        UniformPrior(-0.99, 0.99, parameter_names=["s2_z"]),
        SinePrior(parameter_names=["iota"]),
        PowerLawPrior(50.0, 1000.0, 2.0, parameter_names=["d_L"]),
        UniformPrior(-0.1, 0.1, parameter_names=["t_c"]),
        UniformPrior(0.0, 2 * jnp.pi, parameter_names=["phase_c"]),
        UniformPrior(0.0, jnp.pi, parameter_names=["psi"]),
        UniformPrior(0.0, 2 * jnp.pi, parameter_names=["ra"]),
        CosinePrior(parameter_names=["dec"]),
    ]
)
sample_transforms = get_default_sample_transforms(prior, ground)
likelihood_transforms = get_default_likelihood_transforms(prior, ground, waveform)

# ----------------------------------------------------------------------
# Joint likelihood. No marginalization: phase_c, t_c and d_L are shared by both bands.
# ----------------------------------------------------------------------

likelihood = MultibandLikelihood(
    [
        HeterodynedTransientLikelihoodFD(
            lisa, waveform, f_min=lisa_f_min, f_max=lisa_f_max, n_bins=500,
            reference_parameters=injection_parameters,
        ),
        HeterodynedTransientLikelihoodFD(
            ground, waveform, f_min=ground_f_min, f_max=ground_f_max, n_bins=500,
            reference_parameters=injection_parameters,
        ),
    ]
)
lisa_logl, ground_logl = likelihood.evaluate_per_band(injection_parameters)
print(f"log-likelihood at the injection: LISA {lisa_logl:.1f}, ground {ground_logl:.1f}")

if args.quick:
    sampler_config = FlowMCConfig(
        n_chains=50, n_local_steps=10, n_global_steps=10, n_training_loops=2,
        n_production_loops=2, n_epochs=2, global_thinning=1, n_NFproposal_batch_size=100,
    )
else:
    sampler_config = FlowMCConfig(
        n_chains=1000, n_local_steps=100, n_global_steps=1000, n_training_loops=50,
        n_production_loops=10, global_thinning=100, n_NFproposal_batch_size=100,
    )

jim = Jim(
    likelihood,
    prior,
    sample_transforms=sample_transforms,
    likelihood_transforms=likelihood_transforms,
    periodic=get_default_periodic(prior, sample_transforms),
    sampler_config=sampler_config,
)
start = time.time()
jim.sample()
print(f"Sampling took {(time.time() - start) / 60:.2f} min")

samples = jim.get_samples()
truths = dict(injection_parameters)
for transform in reversed(likelihood_transforms):
    truths = transform.backward(truths)
for name in jim.prior.parameter_names:
    values = np.asarray(samples[name])
    print(f"{name:>8}: {values.mean():.6g} +/- {values.std():.3g}  (truth {float(truths[name]):.6g})")
