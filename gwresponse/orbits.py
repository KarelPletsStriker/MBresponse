"""Orbits of the LISA spacecraft and of the Earth around the Solar System Barycentre.

The spacecraft follow the analytic, first-order-in-eccentricity orbits of
arXiv:2003.00357 (eqs. (35)-(36)): a rigid equilateral triangle whose guiding
centre moves on a circular, 1 AU orbit in the ecliptic plane.

The Earth position is taken from astropy's built-in ephemeris. It is only needed
to convert between geocentric arrival times (the ``t_c`` of Jim) and SSB
arrival times, and is therefore evaluated once per trigger time on the host.
"""

from functools import lru_cache

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from jimgw.typing import FloatLike

from .constants import AU, LISA_TRAILING_ANGLE, YEAR
from .sky import EQUATORIAL_TO_ECLIPTIC

#: Link indices (receiver, sender). ``y_rs`` is measured on spacecraft ``r`` for
#: light emitted by spacecraft ``s``.
LINKS = ((1, 2), (2, 3), (3, 1), (1, 3), (3, 2), (2, 1))


def spacecraft_positions(
    t: Float[Array, " n"],
    arm_length: FloatLike,
    kappa: FloatLike = 0.0,
    lambd: FloatLike = 0.0,
    t0: FloatLike = 0.0,
) -> tuple[Float[Array, "3 n 3"], Float[Array, "n 3"]]:
    """Positions of the three spacecraft and of the guiding centre in the SSB ecliptic frame.

    Args:
        t: Times at which to evaluate the orbits, in s.
        arm_length: Arm length of the constellation, in m.
        kappa: Phase of the guiding centre at ``t0``, alpha(t) = 2 pi (t - t0) / yr + kappa.
        lambd: Orientation of the constellation, beta_n = 2 pi (n - 1) / 3 + lambd.
        t0: Reference time of the orbit, in s.

    Returns:
        The spacecraft positions, of shape (3, N, 3) (one row per spacecraft), and the
        guiding-centre position, of shape (N, 3), in m.
    """
    alpha = (2 * jnp.pi / YEAR * (t - t0) + kappa)[jnp.newaxis, :]  # (1, N)
    s, c = jnp.sin(alpha), jnp.cos(alpha)
    beta = (2 * jnp.pi / 3 * jnp.arange(3) + lambd)[:, jnp.newaxis]  # (3, 1)

    # R * e, with e = L / (2 sqrt(3) R) the eccentricity of the spacecraft orbits
    re = arm_length / (2 * jnp.sqrt(3.0))
    x = re * (s * c * jnp.sin(beta) - (1 + s**2) * jnp.cos(beta))
    y = re * (s * c * jnp.cos(beta) - (1 + c**2) * jnp.sin(beta))
    z = -re * jnp.sqrt(3.0) * jnp.cos(alpha - beta)

    centre = AU * jnp.stack([c[0], s[0], jnp.zeros_like(c[0])], axis=-1)  # (N, 3)
    return centre[jnp.newaxis] + jnp.stack([x, y, z], axis=-1), centre


@lru_cache(maxsize=64)
def _earth_state(gps_time: float) -> tuple[tuple[float, ...], tuple[float, ...]]:
    from astropy import units as u
    from astropy.coordinates import get_body_barycentric_posvel
    from astropy.time import Time

    position, velocity = get_body_barycentric_posvel(
        "earth", Time(gps_time, format="gps")
    )
    position = np.asarray(EQUATORIAL_TO_ECLIPTIC) @ position.xyz.to_value(u.m)
    velocity = np.asarray(EQUATORIAL_TO_ECLIPTIC) @ velocity.xyz.to_value(u.m / u.s)
    return tuple(position.tolist()), tuple(velocity.tolist())


def earth_position_velocity(gps_time: float) -> tuple[np.ndarray, np.ndarray]:
    """Barycentric position (m) and velocity (m/s) of the geocentre, in the ecliptic frame.

    Args:
        gps_time: GPS time, in s.
    """
    position, velocity = _earth_state(float(gps_time))
    return np.array(position), np.array(velocity)


def lisa_orbit_trailing_earth(
    gps_time: float, trailing_angle: float = LISA_TRAILING_ANGLE
) -> dict[str, float]:
    """Orbit parameters placing the LISA guiding centre ``trailing_angle`` behind the Earth.

    Args:
        gps_time: GPS time at which the configuration is enforced, in s.
        trailing_angle: Angle between the Earth and the LISA guiding centre, seen from
            the SSB, in rad. Defaults to 20 degrees.

    Returns:
        ``orbit_kappa`` and ``orbit_t0``, to pass to
        :class:`~gwresponse.detector.SpaceBased` or :func:`~gwresponse.detector.get_LISA`.
    """
    position, _ = earth_position_velocity(gps_time)
    earth_longitude = float(np.arctan2(position[1], position[0]))
    return {
        "orbit_kappa": float(np.mod(earth_longitude - trailing_angle, 2 * np.pi)),
        "orbit_t0": float(gps_time),
    }
