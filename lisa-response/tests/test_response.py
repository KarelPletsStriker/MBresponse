import jax.numpy as jnp
import numpy as np
from jimgw.core.single_event.data import Data
from jimgw.core.single_event.polarization import Polarization

from conftest import GPS, LISA_PARAMS
from gwresponse import SpaceBased, spacecraft_positions, time_to_merger
from gwresponse.constants import C_SI
from gwresponse.sky import sky_rotation, source_direction


def _detector(channel, **kwargs):
    det = SpaceBased(f"LISA_{channel}", channel=channel, orbit_kappa=0.4, orbit_lambda=0.2, orbit_t0=GPS, **kwargs)
    det.data = Data.from_td(jnp.zeros(1024), 10.0, start_time=GPS - 9000.0, trigger_time=GPS)
    return det


def test_long_wavelength_limit_is_jims_antenna_pattern():
    """At low frequency, X is Jim's ground-based response of a 60-degree interferometer with LISA's arms."""
    # Deep in the long-wavelength regime, where arm and constellation delays are negligible
    f = jnp.array([1e-6, 3e-6])
    params = {**LISA_PARAMS, "t_c": 0.0}
    tdi = {c: _detector(c, time_reference="ssb").tdi_transfer(f, params) for c in "AET"}

    t = GPS + time_to_merger(f, params)
    positions, centre = spacecraft_positions(t, 2.5e9, 0.4, 0.2, GPS)
    rotation = sky_rotation(GPS)  # Jim's equatorial frame of date -> J2000 ecliptic
    k = -source_direction(params["ra"], params["dec"], rotation)
    fL = f * 2.5e9 / C_SI
    for pol in "pc":
        X = tdi["T"][pol] / jnp.sqrt(3) + tdi["E"][pol] / jnp.sqrt(6) - tdi["A"][pol] / jnp.sqrt(2)
        arm13 = (positions[0] - positions[2]) / jnp.linalg.norm(positions[0] - positions[2], axis=-1, keepdims=True)
        arm12 = (positions[0] - positions[1]) / jnp.linalg.norm(positions[0] - positions[1], axis=-1, keepdims=True)
        a, b = arm13 @ rotation, arm12 @ rotation
        detector_tensor = 0.5 * (jnp.einsum("ni,nj->nij", a, a) - jnp.einsum("ni,nj->nij", b, b))
        wave_tensor = Polarization(pol).tensor_from_sky(params["ra"], params["dec"], params["psi"], 0.0)
        antenna = jnp.einsum("nij,ij->n", detector_tensor, wave_tensor)
        expected = 16 * jnp.pi**2 * fL**2 * antenna * jnp.exp(-2j * jnp.pi * f * (centre @ k) / C_SI)
        np.testing.assert_allclose(X, expected, rtol=2e-3)


def test_geocentric_and_barycentric_time_references_agree():
    f = jnp.linspace(3e-3, 0.04, 50)
    h = {"p": jnp.ones_like(f) * (1 + 0.5j), "c": jnp.ones_like(f) * (0.3 - 1j)}
    geo, ssb = _detector("A"), _detector("A", time_reference="ssb")
    k = -source_direction(LISA_PARAMS["ra"], LISA_PARAMS["dec"], geo.sky_rotation())
    delay = geo.ssb_delay(k, LISA_PARAMS["t_c"])
    np.testing.assert_allclose(
        geo.fd_response(f, h, LISA_PARAMS),
        ssb.fd_response(f, h, {**LISA_PARAMS, "t_c": LISA_PARAMS["t_c"] + delay}),
        rtol=1e-9,
    )


def test_constellation_geometry():
    t = GPS + jnp.linspace(0, 3e7, 7)
    positions, centre = spacecraft_positions(t, 2.5e9)
    for i, j in ((0, 1), (1, 2), (2, 0)):
        np.testing.assert_allclose(jnp.linalg.norm(positions[i] - positions[j], axis=-1), 2.5e9, rtol=2e-2)
    np.testing.assert_allclose(jnp.linalg.norm(centre, axis=-1), 1.495978707e11, rtol=1e-12)


def test_roemer_delay_matches_astropy():
    """Geocentre-to-barycentre delay against astropy's barycentric light travel time."""
    import astropy.units as u
    from astropy.coordinates import BarycentricMeanEcliptic, CartesianRepresentation, EarthLocation, SkyCoord
    from astropy.time import Time

    det = _detector("A")
    n = np.asarray(source_direction(1.0, 0.3, det.sky_rotation()))
    sky = SkyCoord(CartesianRepresentation(*n), frame=BarycentricMeanEcliptic(equinox="J2000"))
    geocentre = EarthLocation.from_geocentric(0, 0, 0, unit="m")
    for t_c, tolerance in ((0.0, 1e-5), (3600.0, 1e-5), (86400.0, 1e-4)):
        expected = Time(GPS + t_c, format="gps").light_travel_time(sky, kind="barycentric", location=geocentre)
        # light_travel_time is t_SSB - t_geocentre for a wave arriving from the source
        np.testing.assert_allclose(float(det.ssb_delay(-jnp.asarray(n), t_c)), expected.to_value(u.s), atol=tolerance)


def test_sky_frame_matches_jims_equinox_of_date():
    """Jim's (ra, dec) are referred to the mean equinox of date; check against astropy."""
    import astropy.units as u
    from astropy.coordinates import BarycentricMeanEcliptic, PrecessedGeocentric, SkyCoord
    from astropy.time import Time

    time = Time(GPS, format="gps")
    for ra, dec in ((1.0, 0.3), (4.0, -1.1), (2.5, 1.4)):
        ours = np.asarray(source_direction(ra, dec, sky_rotation(GPS)))
        sky = SkyCoord(ra * u.rad, dec * u.rad, frame=PrecessedGeocentric(equinox=time, obstime=time))
        expected = sky.transform_to(BarycentricMeanEcliptic(equinox="J2000")).cartesian.xyz.value
        # astropy also removes annual aberration (< 1e-4 rad), which the GW analysis ignores
        assert np.arccos(np.clip(ours @ expected, -1, 1)) < 1.2e-4
        assert np.arccos(np.clip(ours @ np.asarray(source_direction(ra, dec)), -1, 1)) > 2e-3  # precession since J2000


def test_get_lisa_warns_without_trigger_time():
    import pytest

    from gwresponse import get_LISA

    with pytest.warns(UserWarning, match="trigger_time"):
        get_LISA("A")
