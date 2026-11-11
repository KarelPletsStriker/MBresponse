"""Joint likelihood of space-based and ground-based detectors.

Jim's single-event likelihoods require every detector to share one frequency
spacing, i.e. data of the same duration. LISA data last days to years and are
sampled at ~0.1-1 Hz, while ground-based data last seconds to minutes at kHz
rates, so the two cannot be put in a single ``TransientLikelihoodFD``.

:class:`MultibandLikelihood` instead adds the log-likelihoods of independent Jim
likelihoods (one per frequency band, e.g. LISA and a ground-based network), which
is exact for independent noise. All of them see the same source parameters: the
sky position ``ra``, ``dec``, the polarization ``psi`` and the geocentric arrival
time ``t_c`` have the same meaning for :class:`~gwresponse.detector.SpaceBased`
and for Jim's ground-based detectors.
"""

from collections.abc import Mapping, Sequence

import jax.numpy as jnp

from jimgw.core.base import LikelihoodBase
from jimgw.core.single_event.likelihood import SingleEventLikelihood
from jimgw.typing import FloatLike, FloatScalar

from .detector import SpaceBased
from .waveform import check_waveform


class MultibandLikelihood(LikelihoodBase):
    """Sum of the log-likelihoods of Jim single-event likelihoods analysing the same signal.

    Args:
        likelihoods: Jim likelihoods (``TransientLikelihoodFD``,
            ``HeterodynedTransientLikelihoodFD``, ...), each with its own detectors and
            data. Their data must share the same trigger time, so that ``t_c`` has the
            same meaning in all of them.

    Raises:
        ValueError: If no likelihood is given, if their trigger times differ or detector
            names repeat, if a barycentre-referenced LISA channel is combined with
            ground-based detectors, or if any of several likelihoods marginalizes over
            a parameter: a marginalization done separately in each band would treat
            the shared parameter as independent.
        TypeError: If a likelihood containing a space-based detector does not use
            the IMRPhenomD waveform.

    Example:
        >>> lisa = TransientLikelihoodFD(get_LISA("AE", gps), waveform, f_min=1e-4, f_max=0.05)
        >>> ground = TransientLikelihoodFD([get_H1(), get_L1()], waveform, f_min=20, f_max=1024)
        >>> likelihood = MultibandLikelihood([lisa, ground])
        >>> likelihood.evaluate(params)
    """

    def __init__(self, likelihoods: Sequence[SingleEventLikelihood]):
        likelihoods = tuple(likelihoods)
        if not likelihoods:
            raise ValueError("MultibandLikelihood needs at least one likelihood.")
        for likelihood in likelihoods:
            if not isinstance(likelihood, SingleEventLikelihood):
                raise TypeError(
                    f"Expected Jim single-event likelihoods, got {type(likelihood).__name__}."
                )
            marginalized = [
                name
                for flag, name in (
                    ("time_marginalization", "t_c"),
                    ("phase_marginalization", "phase_c"),
                    ("distance_marginalization", "d_L"),
                )
                if getattr(likelihood, flag, False)
            ]
            if marginalized and len(likelihoods) > 1:
                raise ValueError(
                    f"Parameters {marginalized} are marginalized in one band. They are "
                    "shared by all bands, so sample them instead."
                )
            if any(isinstance(d, SpaceBased) for d in likelihood.detectors):
                check_waveform(likelihood.waveform)

        trigger_times = {likelihood.trigger_time for likelihood in likelihoods}
        if len(trigger_times) != 1:
            raise ValueError(
                f"All likelihoods must use data with the same trigger time, got {sorted(trigger_times)}."
            )
        detectors = [d for likelihood in likelihoods for d in likelihood.detectors]
        ssb_referenced = [
            d.name for d in detectors if isinstance(d, SpaceBased) and d.time_reference != "geocenter"
        ]
        if ssb_referenced and not all(isinstance(d, SpaceBased) for d in detectors):
            raise ValueError(
                f"Detectors {ssb_referenced} reference t_c to the barycentre, while Jim's "
                "ground-based detectors reference it to the geocentre. Use "
                'time_reference="geocenter" so that t_c means the same in every band.'
            )
        names = [d.name for d in detectors]
        if len(set(names)) != len(names):
            raise ValueError(f"Detector names must be unique across likelihoods, got {names}.")
        self.likelihoods = likelihoods

    @property
    def detectors(self) -> tuple:
        """All detectors of all likelihoods."""
        return tuple(d for likelihood in self.likelihoods for d in likelihood.detectors)

    @property
    def detector_names(self) -> list[str]:
        return [d.name for d in self.detectors]

    @property
    def trigger_time(self) -> float:
        return self.likelihoods[0].trigger_time

    @property
    def waveform(self):
        """Waveform of the first likelihood."""
        return self.likelihoods[0].waveform

    def evaluate(self, params: Mapping[str, FloatLike]) -> FloatScalar:
        """Sum of the log-likelihoods of every band."""
        log_likelihood: FloatScalar = jnp.zeros(())
        for likelihood in self.likelihoods:
            log_likelihood += likelihood.evaluate(params)
        return log_likelihood

    def evaluate_per_band(self, params: Mapping[str, FloatLike]) -> list[FloatScalar]:
        """Log-likelihood of each band, in the order of ``likelihoods``."""
        return [likelihood.evaluate(params) for likelihood in self.likelihoods]
