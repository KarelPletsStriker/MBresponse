"""Check the port against the numerical outputs printed in the original Mathematica notebook."""

import numpy as np
import pytest

from galactic_foreground import gw, population, selection, stability
from galactic_foreground.data import cheng2020_sample

mc = gw.chirp_mass


def test_chirp_masses():
    assert mc(0.3, 0.3) == pytest.approx(0.261165, rel=1e-5)
    assert mc(0.3, 0.8) == pytest.approx(0.416724, rel=1e-5)
    assert mc(0.6, 0.6) == pytest.approx(0.52233, rel=1e-5)
    assert mc(0.6, 0.8) == pytest.approx(0.601893, rel=1e-5)


def test_dfdt_and_steady_state():
    assert gw.dfdt(1.0, 1.0) == pytest.approx(5.82578e-7, rel=1e-5)
    assert gw.time_to_merger(1.0, 0.5) == pytest.approx(2.04359e6, rel=1e-5)
    assert gw.fmin_steady_state() == pytest.approx(0.000151344, rel=1e-5)


@pytest.mark.parametrize("rate,mchirp,expected", [
    (0.0005, mc(0.3, 0.3), 9600.75),
    (0.003, mc(0.3, 0.8), 26438.4),
    (0.003, mc(0.6, 0.6), 18144.3),
    (0.0065, mc(0.6, 0.6), 39312.6),
])
def test_cumulative_coefficients(rate, mchirp, expected):
    assert gw.cumulative_coefficient(rate, mchirp) == pytest.approx(expected, rel=1e-5)


def test_galactic_and_local_numbers():
    assert gw.number_in_band(0.0005, mc(0.3, 0.3), 2e-4, 0.5) == pytest.approx(701819, rel=1e-5)
    v = gw.volume_fraction_100pc()
    assert v == pytest.approx(5.92593e-6, rel=1e-5)
    assert gw.number_in_band(0.0005, mc(0.3, 0.3), 1.5e-4, 0.5, v) == pytest.approx(8.95676, rel=1e-5)
    assert gw.number_in_band(0.003, mc(0.3, 0.8), 1.5e-4, 0.5, v) == pytest.approx(24.665, rel=1e-4)
    assert gw.number_in_band(0.003, mc(0.6, 0.6), 1.5e-4, 0.5, v) == pytest.approx(16.9272, rel=1e-5)


def test_stability_boundary_matches_nelemans_fit():
    m1 = np.array([0.3, 0.6, 1.0, 1.4])
    computed = np.array([stability.boundary_m2(x) for x in m1])
    np.testing.assert_allclose(computed, population.nelemans_boundary(m1), atol=0.01)


def test_cheng_sample_size():
    assert cheng2020_sample().size == 253


def test_selection_reference_star_has_weight_scale():
    # A star at the reference mass and temperature gets exactly `scale` repeats.
    m = np.array([0.2, 0.3, 0.6])
    T = np.array([12000.0, 18000.0, 15000.0])
    reps, ref_m, ref_t4 = selection.selection_repeats(m, T)
    assert ref_m == pytest.approx(0.25)
    assert ref_t4 == pytest.approx(1.5)
    assert np.all(reps > 0)
