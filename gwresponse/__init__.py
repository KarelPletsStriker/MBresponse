"""gwresponse: LISA response for parameter estimation with Jim.

Implements LISA (TDI channels A, E, T) as a Jim ``Detector``, so that Jim's data
handling, injections, likelihoods, priors, transforms and samplers work for
space-based detectors as they do for ground-based ones, and a likelihood
combining LISA and ground-based detectors.
"""

import jax

# Jim's detectors require 64-bit precision
jax.config.update("jax_enable_x64", True)

from .detector import SpaceBased, get_LISA  # noqa: E402
from .likelihood import MultibandLikelihood  # noqa: E402
from .noise import sky_averaged_sensitivity, tdi_noise_psd  # noqa: E402
from .orbits import lisa_orbit_trailing_earth, spacecraft_positions  # noqa: E402
from .sky import ecliptic_to_equatorial, equatorial_to_ecliptic  # noqa: E402
from .waveform import check_waveform, time_to_merger  # noqa: E402

__all__ = [
    "MultibandLikelihood",
    "SpaceBased",
    "check_waveform",
    "ecliptic_to_equatorial",
    "equatorial_to_ecliptic",
    "get_LISA",
    "lisa_orbit_trailing_earth",
    "sky_averaged_sensitivity",
    "spacecraft_positions",
    "tdi_noise_psd",
    "time_to_merger",
]
