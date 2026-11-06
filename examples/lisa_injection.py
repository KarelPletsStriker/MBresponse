"""Injection-recovery of a massive black hole binary in LISA, with Jim and flowMC.

The same workflow as Jim's ground-based ``examples/injection.py``: detectors,
noise simulation, injection, prior, transforms, likelihood and sampler are all
Jim's; only the detectors (the LISA TDI channels A and E) come from gwresponse.

    python examples/lisa_injection.py           # production settings
    python examples/lisa_injection.py --quick   # smoke test, a few minutes on a CPU
"""

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from jimgw.core.jim import Jim
from jimgw.core.prior import CombinePrior, CosinePrior, PowerLawPrior, SinePrior, UniformPrior
from jimgw.core.single_event.helpers import (
    get_default_likelihood_transforms,
    get_default_periodic,
    get_default_sample_transforms,
)
from jimgw.core.single_event.likelihood import HeterodynedTransientLikelihoodFD
from jimgw.core.single_event.waveform import RippleIMRPhenomD
from jimgw.samplers.config import FlowMCConfig

from gwresponse import get_LISA

parser = argparse.ArgumentParser()
parser.add_argument("--quick", action="store_true", help="tiny sampler settings")
args = parser.parse_args()

# ----------------------------------------------------------------------
# Waveform model: only IMRPhenomD is supported by the LISA response
# ----------------------------------------------------------------------

f_min, f_max = 3e-4, 0.05
waveform = RippleIMRPhenomD(f_ref=f_min)

# ----------------------------------------------------------------------
# Detectors, noise and injection
# ----------------------------------------------------------------------

gps = 1.4e9
injection_parameters = {
    "M_c": 4.0e5,  # detector-frame chirp mass, solar masses
    "eta": 0.24,
    "s1_z": 0.3,
    "s2_z": -0.1,
    "d_L": 3.0e4,  # Mpc
    "iota": 0.5,
    "phase_c": 0.3,
    "ra": 1.0,
    "dec": 0.3,
    "psi": 0.7,
    "t_c": 100.0,  # geocentric arrival time of the peak, relative to gps
}

# A and E channels; the constellation trails the Earth by 20 degrees at the trigger
ifos = get_LISA("AE", trigger_time=gps)
for ifo in ifos:
    ifo.load_and_set_psd()  # analytic SciRD noise of the TDI channel
    ifo.simulate_and_set_noise(
        duration=2**21,  # ~24 days
        sampling_frequency=0.1,
        trigger_time=gps,
        f_min=f_min,
        f_max=f_max,
        post_trigger_duration=2e4,
        rng_key=jax.random.key(1),
    )
    ifo.inject_signal(waveform, injection_parameters)

# ----------------------------------------------------------------------
# Prior, in the same parametrization as for ground-based detectors
# ----------------------------------------------------------------------

prior = CombinePrior(
    [
        UniformPrior(3.9e5, 4.1e5, parameter_names=["M_c"]),
        UniformPrior(0.3, 1.0, parameter_names=["q"]),
        UniformPrior(-0.99, 0.99, parameter_names=["s1_z"]),
        UniformPrior(-0.99, 0.99, parameter_names=["s2_z"]),
        SinePrior(parameter_names=["iota"]),
        PowerLawPrior(1e4, 6e4, 2.0, parameter_names=["d_L"]),
        UniformPrior(0.0, 200.0, parameter_names=["t_c"]),
        UniformPrior(0.0, jnp.pi, parameter_names=["psi"]),
        UniformPrior(0.0, 2 * jnp.pi, parameter_names=["ra"]),
        CosinePrior(parameter_names=["dec"]),
    ]
)

# The detector-frame time and sky transforms are specific to ground-based
# detectors, so LISA samples t_c and ra/dec directly.
sample_transforms = get_default_sample_transforms(
    prior, ifos, detector_frame_time=False, detector_frame_sky=False
)
likelihood_transforms = get_default_likelihood_transforms(prior, ifos, waveform)

# ----------------------------------------------------------------------
# Likelihood: relative binning makes LISA likelihoods ~100x faster
# ----------------------------------------------------------------------

likelihood = HeterodynedTransientLikelihoodFD(
    ifos,
    waveform=waveform,
    f_min=f_min,
    f_max=f_max,
    n_bins=500,
    reference_parameters=injection_parameters,
    phase_marginalization=True,
)

# ----------------------------------------------------------------------
# Sample
# ----------------------------------------------------------------------

if args.quick:
    sampler_config = FlowMCConfig(
        n_chains=50, n_local_steps=10, n_global_steps=10, n_training_loops=2,
        n_production_loops=2, n_epochs=2, global_thinning=1, n_NFproposal_batch_size=20, chain_batch_size=50,
    )
else:
    sampler_config = FlowMCConfig(
        n_chains=1000, n_local_steps=100, n_global_steps=1000, n_training_loops=50,
        n_production_loops=10, global_thinning=100, n_NFproposal_batch_size=20, chain_batch_size=50,
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

# ----------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------

samples = jim.get_samples()
truths = dict(injection_parameters)
for transform in reversed(likelihood_transforms):
    truths = transform.backward(truths)
for name in jim.prior.parameter_names:
    values = np.asarray(samples[name])
    print(f"{name:>8}: {values.mean():.6g} +/- {values.std():.3g}  (truth {float(truths[name]):.6g})")

if not args.quick:
    import corner

    names = jim.prior.parameter_names
    fig = corner.corner(
        np.stack([samples[k] for k in names]).T,
        labels=list(names),
        truths=[float(truths[k]) for k in names],
    )
    fig.savefig(Path(__file__).with_suffix(".png"))
