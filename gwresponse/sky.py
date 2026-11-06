"""Sky geometry shared by the space-based and ground-based detectors.

All LISA computations are done in the ecliptic frame centred at the Solar System
Barycentre (SSB). Source parameters are, as everywhere in Jim, the equatorial
right ascension ``ra``, declination ``dec`` and polarization angle ``psi``, so a
joint LISA + ground-based analysis shares one set of sky parameters.

The polarization tensors are built with Jim's own
:class:`~jimgw.core.single_event.polarization.Polarization` convention (evaluated
with zero sidereal time, where Jim's Earth-fixed frame coincides with the
equatorial frame) and rotated to the ecliptic frame, so that ``psi`` has the
same meaning for LISA and for the ground-based detectors.
"""

import jax.numpy as jnp
from jaxtyping import Array, Float

from jimgw.core.single_event.polarization import Polarization
from jimgw.typing import FloatLike

from .constants import OBLIQUITY

_COS_EPS = jnp.cos(OBLIQUITY)
_SIN_EPS = jnp.sin(OBLIQUITY)

#: Rotation matrix from equatorial (ICRS) to ecliptic Cartesian coordinates.
EQUATORIAL_TO_ECLIPTIC = jnp.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, _COS_EPS, _SIN_EPS],
        [0.0, -_SIN_EPS, _COS_EPS],
    ]
)


def equatorial_to_ecliptic(
    ra: FloatLike, dec: FloatLike
) -> tuple[FloatLike, FloatLike]:
    """Convert equatorial ``(ra, dec)`` to ecliptic longitude and latitude, in rad."""
    beta = jnp.arcsin(jnp.sin(dec) * _COS_EPS - jnp.cos(dec) * _SIN_EPS * jnp.sin(ra))
    lambd = jnp.arctan2(
        jnp.sin(ra) * _COS_EPS + jnp.tan(dec) * _SIN_EPS, jnp.cos(ra)
    )
    return jnp.mod(lambd, 2 * jnp.pi), beta


def ecliptic_to_equatorial(
    lambd: FloatLike, beta: FloatLike
) -> tuple[FloatLike, FloatLike]:
    """Convert ecliptic longitude and latitude to equatorial ``(ra, dec)``, in rad."""
    dec = jnp.arcsin(
        jnp.sin(beta) * _COS_EPS + jnp.cos(beta) * _SIN_EPS * jnp.sin(lambd)
    )
    ra = jnp.arctan2(
        jnp.sin(lambd) * _COS_EPS - jnp.tan(beta) * _SIN_EPS, jnp.cos(lambd)
    )
    return jnp.mod(ra, 2 * jnp.pi), dec


def source_direction(ra: FloatLike, dec: FloatLike) -> Float[Array, "3"]:
    """Unit vector from the SSB towards the source, in ecliptic coordinates."""
    n_eq = jnp.array(
        [jnp.cos(dec) * jnp.cos(ra), jnp.cos(dec) * jnp.sin(ra), jnp.sin(dec)]
    )
    return EQUATORIAL_TO_ECLIPTIC @ n_eq


def polarization_tensors(
    ra: FloatLike, dec: FloatLike, psi: FloatLike, modes: str = "pc"
) -> dict[str, Float[Array, "3 3"]]:
    """Polarization tensors of the wave in ecliptic coordinates, in Jim's convention.

    Args:
        ra: Right ascension, in rad.
        dec: Declination, in rad.
        psi: Polarization angle, in rad.
        modes: Polarization modes, as for Jim's ground-based detectors.

    Returns:
        A dict mapping each polarization name (``"p"``, ``"c"``) to its 3x3 tensor.
    """
    tensors = {}
    for mode in modes:
        e_eq = Polarization(mode).tensor_from_sky(ra, dec, psi, 0.0)
        tensors[mode] = EQUATORIAL_TO_ECLIPTIC @ e_eq @ EQUATORIAL_TO_ECLIPTIC.T
    return tensors
