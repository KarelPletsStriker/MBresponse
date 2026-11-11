"""Instrumental noise of the first-generation TDI channels of an equal-arm LISA.

The spectra are in fractional frequency units, as the output of
:meth:`gwresponse.detector.SpaceBased.fd_response`, and use the SciRD noise
levels (test-mass acceleration ``A = 3 fm s^-2 Hz^-1/2`` and optical metrology
``P = 15 pm Hz^-1/2``), as in the LISA Data Challenges and arXiv:2003.00357.

With ``x = 2 pi f L / c`` and the per-link noises ``S_acc``, ``S_oms``:

- ``S_X = 16 sin^2 x [S_oms + (3 + cos 2x) S_acc]``
- ``S_A = S_E = 8 sin^2 x [(2 + cos x) S_oms + 2 (3 + 2 cos x + cos 2x) S_acc]``
- ``S_T = 32 sin^2 x sin^2(x/2) [S_oms + 4 sin^2(x/2) S_acc]``

The A, E and T channels are uncorrelated for equal arms, so they can be analysed
as independent Jim detectors.
"""

import jax.numpy as jnp
from jaxtyping import Array, Float

from jimgw.typing import FloatLike

from .constants import C_SI, LISA_ARM_LENGTH

CHANNELS = ("X", "A", "E", "T")


def acceleration_noise(f: Float[Array, " n"], A: FloatLike = 3.0) -> Float[Array, " n"]:
    """Test-mass acceleration noise PSD of a single link, in fractional frequency (1/Hz).

    Args:
        f: Frequencies, in Hz.
        A: Acceleration noise amplitude, in units of 1e-15 m s^-2 Hz^-1/2.
    """
    return (
        (A * 1e-15) ** 2
        * (1 + (4e-4 / f) ** 2)
        * (1 + (f / 8e-3) ** 4)
        / (2 * jnp.pi * f * C_SI) ** 2
    )


def oms_noise(f: Float[Array, " n"], P: FloatLike = 15.0) -> Float[Array, " n"]:
    """Optical metrology system noise PSD of a single link, in fractional frequency (1/Hz).

    Args:
        f: Frequencies, in Hz.
        P: Displacement noise amplitude, in units of 1e-12 m Hz^-1/2.
    """
    return (P * 1e-12) ** 2 * (1 + (2e-3 / f) ** 4) * (2 * jnp.pi * f / C_SI) ** 2


def tdi_noise_psd(
    f: Float[Array, " n"],
    channel: str,
    A: FloatLike = 3.0,
    P: FloatLike = 15.0,
    arm_length: FloatLike = LISA_ARM_LENGTH,
) -> Float[Array, " n"]:
    """One-sided noise PSD of a first-generation TDI channel, in fractional frequency (1/Hz).

    Args:
        f: Positive frequencies, in Hz.
        channel: ``"X"``, ``"A"``, ``"E"`` or ``"T"``.
        A: Test-mass acceleration noise amplitude, in units of 1e-15 m s^-2 Hz^-1/2.
        P: Optical metrology noise amplitude, in units of 1e-12 m Hz^-1/2.
        arm_length: Arm length, in m.
    """
    if channel not in CHANNELS:
        raise ValueError(f"Invalid TDI channel {channel!r}, valid options are {CHANNELS}.")
    x = 2 * jnp.pi * f * arm_length / C_SI
    s_acc, s_oms = acceleration_noise(f, A), oms_noise(f, P)
    if channel == "X":
        return 16 * jnp.sin(x) ** 2 * (s_oms + (3 + jnp.cos(2 * x)) * s_acc)
    if channel in ("A", "E"):
        return (
            8
            * jnp.sin(x) ** 2
            * (
                (2 + jnp.cos(x)) * s_oms
                + 2 * (3 + 2 * jnp.cos(x) + jnp.cos(2 * x)) * s_acc
            )
        )
    half = jnp.sin(x / 2) ** 2
    return 32 * jnp.sin(x) ** 2 * half * (s_oms + 4 * half * s_acc)


def sky_averaged_sensitivity(
    f: Float[Array, " n"],
    A: FloatLike = 3.0,
    P: FloatLike = 15.0,
    arm_length: FloatLike = LISA_ARM_LENGTH,
) -> Float[Array, " n"]:
    """Sky-averaged strain sensitivity of LISA, eq. (13) of Robson et al. (arXiv:1803.01944).

    Without the galactic confusion noise. Provided as a reference curve; it is not
    used in the likelihood.
    """
    f_star = C_SI / (2 * jnp.pi * arm_length)
    p_oms = (P * 1e-12) ** 2 * (1 + (2e-3 / f) ** 4)
    p_acc = (A * 1e-15) ** 2 * (1 + (4e-4 / f) ** 2) * (1 + (f / 8e-3) ** 4)
    return (
        10
        / (3 * arm_length**2)
        * (p_oms + 2 * (1 + jnp.cos(f / f_star) ** 2) * p_acc / (2 * jnp.pi * f) ** 4)
        * (1 + 0.6 * (f / f_star) ** 2)
    )
