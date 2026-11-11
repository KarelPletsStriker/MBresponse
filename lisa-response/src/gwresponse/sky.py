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

Jim's ground-based detectors turn ``ra`` into an hour angle with the Greenwich
*mean* sidereal time, which is measured from the mean equinox of date. Their
``ra``, ``dec`` and ``psi`` are therefore effectively referred to the mean equator
and equinox at the trigger time, not to J2000 (the two differ by ~0.35 deg in
2025). :func:`sky_rotation` applies the same convention to LISA, so that both
detectors see the same source direction.
"""

from functools import lru_cache

import numpy as np

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


@lru_cache(maxsize=64)
def _sky_rotation(gps_time: float) -> tuple[tuple[float, ...], ...]:
    import erfa
    from astropy.time import Time

    tt = Time(gps_time, format="gps").tt
    # Bias-precession matrix (IAU 2006): ICRS -> mean equator and equinox of date
    precession = erfa.pmat06(tt.jd1, tt.jd2)
    return tuple(map(tuple, np.asarray(EQUATORIAL_TO_ECLIPTIC) @ precession.T))


def sky_rotation(gps_time: float | None) -> Float[Array, "3 3"]:
    """Rotation from Jim's equatorial sky frame at ``gps_time`` to the J2000 ecliptic frame.

    Jim's ground-based detectors refer ``ra``, ``dec`` and ``psi`` to the mean equator
    and equinox of date, see the module docstring. This undoes that precession and
    rotates to the ecliptic frame of the LISA orbits and the Earth ephemeris.

    Args:
        gps_time: Trigger time, in GPS seconds. ``None`` treats the sky angles as
            ICRS (J2000) coordinates, without precession.
    """
    if gps_time is None:
        return EQUATORIAL_TO_ECLIPTIC
    return jnp.array(_sky_rotation(float(gps_time)))


def source_direction(
    ra: FloatLike, dec: FloatLike, rotation: Float[Array, "3 3"] = EQUATORIAL_TO_ECLIPTIC
) -> Float[Array, "3"]:
    """Unit vector from the SSB towards the source, in ecliptic coordinates.

    Args:
        ra: Right ascension, in rad.
        dec: Declination, in rad.
        rotation: Equatorial-to-ecliptic rotation, see :func:`sky_rotation`. The
            default treats ``ra`` and ``dec`` as ICRS coordinates.
    """
    n_eq = jnp.array(
        [jnp.cos(dec) * jnp.cos(ra), jnp.cos(dec) * jnp.sin(ra), jnp.sin(dec)]
    )
    return rotation @ n_eq


def polarization_tensors(
    ra: FloatLike,
    dec: FloatLike,
    psi: FloatLike,
    modes: str = "pc",
    rotation: Float[Array, "3 3"] = EQUATORIAL_TO_ECLIPTIC,
) -> dict[str, Float[Array, "3 3"]]:
    """Polarization tensors of the wave in ecliptic coordinates, in Jim's convention.

    Args:
        ra: Right ascension, in rad.
        dec: Declination, in rad.
        psi: Polarization angle, in rad.
        modes: Polarization modes, as for Jim's ground-based detectors.
        rotation: Equatorial-to-ecliptic rotation, see :func:`sky_rotation`.

    Returns:
        A dict mapping each polarization name (``"p"``, ``"c"``) to its 3x3 tensor.
    """
    tensors = {}
    for mode in modes:
        e_eq = Polarization(mode).tensor_from_sky(ra, dec, psi, 0.0)
        tensors[mode] = rotation @ e_eq @ rotation.T
    return tensors
