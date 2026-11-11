"""IMRPhenomD support for the space-based response.

The LISA response is evaluated along the time-frequency map t(f) of the signal
(stationary phase approximation), so it depends on the waveform model. This
package currently supports only ripple's ``IMRPhenomD`` (the waveform behind
Jim's ``RippleIMRPhenomD``), whose t(f) is computed here directly from ripple's
phenomenological phase.
"""

from collections.abc import Mapping

import jax
import jax.numpy as jnp
import ripplegw
from jaxtyping import Array, Float
from ripplegw.constants import MTSUN
from ripplegw.conversions import Mc_eta_to_ms
from ripplegw.waveforms.cbc.IMRPhenomD.IMRPhenomD import Phase, get_IIb_raw_phase
from ripplegw.waveforms.cbc.IMRPhenomD.IMRPhenomD_QNMdata import fM_CUT
from ripplegw.waveforms.cbc.IMRPhenomD.IMRPhenomD_utils import (
    get_coeffs,
    get_transition_frequencies,
)

from jimgw.typing import FloatLike

#: The ripple IMRPhenomD class (ripple only exposes it through its registry).
IMRPhenomD = type(ripplegw.waveform("IMRPhenomD"))

#: Waveform models supported by the space-based response.
SUPPORTED_WAVEFORMS = (IMRPhenomD,)


def check_waveform(waveform) -> None:
    """Raise if ``waveform`` is not a waveform model supported by the LISA response.

    Raises:
        TypeError: If ``waveform`` is not ripple's IMRPhenomD (Jim's ``RippleIMRPhenomD``).
    """
    if not isinstance(waveform, SUPPORTED_WAVEFORMS):
        raise TypeError(
            "The space-based response currently supports only IMRPhenomD "
            "(jimgw.core.single_event.waveform.RippleIMRPhenomD), got "
            f"{waveform!r}."
        )


def time_to_merger(
    frequency: Float[Array, " n"], params: Mapping[str, FloatLike]
) -> Float[Array, " n"]:
    """Time-frequency map of the IMRPhenomD (2, 2) mode, relative to the peak of the signal.

    Given by the stationary phase approximation, t(f) = -(1/2 pi) d phase / df with
    ``phase`` the phase of ripple's IMRPhenomD (``h ~ exp(i phase)``). Its origin is
    ripple's (and so Jim's) definition of ``t_c``, a few M before the amplitude peak.
    It increases through inspiral and merger, turns over in the ringdown well above
    the ringdown frequency, and is constant above the cut frequency M f = 0.2 of the
    model, where the amplitude vanishes.

    Unlike differentiating ``IMRPhenomD.phase``, it is exact on any frequency grid,
    including the non-uniform grids of Jim's heterodyned likelihood.

    Args:
        frequency: Frequencies, in Hz.
        params: Parameters containing ``M_c`` (solar masses), ``eta``, ``s1_z`` and ``s2_z``.

    Returns:
        Time to the peak of the signal, in s.
    """
    m1, m2 = Mc_eta_to_ms(jnp.array([params["M_c"], params["eta"]]))
    theta = jnp.array([m1, m2, params["s1_z"], params["s2_z"]])
    coeffs = get_coeffs(theta)
    transition_freqs = get_transition_frequencies(theta, coeffs[5], coeffs[6])
    _, _, _, f4, f_RD, f_damp = transition_freqs
    M_s = (m1 + m2) * MTSUN

    # Time shift ripple applies so that the amplitude peak is at t = 0
    t0 = jax.grad(get_IIb_raw_phase)(f4 * M_s, theta, coeffs, f_RD, f_damp) * M_s

    f = jnp.minimum(frequency, fM_CUT / M_s)
    # ripple's Phase is elementwise in f, so a unit tangent gives its derivative
    _, dpsi_df = jax.jvp(
        lambda fr: Phase(fr, theta, coeffs, transition_freqs), (f,), (jnp.ones_like(f),)
    )
    return (dpsi_df - t0) / (2 * jnp.pi)
