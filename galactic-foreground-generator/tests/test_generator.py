import numpy as np

from galactic_foreground import generator, gw


def _intrinsic():
    rng = np.random.default_rng(1)
    return np.concatenate([rng.normal(0.3, 0.05, 2000), rng.normal(0.6, 0.1, 4000)]).clip(0.12, 1.2)


def test_catalogue_columns_and_ranges():
    cat = generator.generate_catalogue(_intrinsic(), n=5000, f_lo=1e-4, f_hi=1e-2)
    assert len(cat) == 5000
    assert (cat.m1 >= cat.m2).all()
    assert cat.f_gw.between(1e-4, 1e-2).all()
    np.testing.assert_allclose(cat.f_orb, cat.f_gw / 2)
    assert cat.distance.max() < 15 + 8.2 + 1


def test_frequency_slope_is_minus_eight_thirds():
    cat = generator.generate_catalogue(_intrinsic(), n=50000, f_lo=1e-4, f_hi=1e-1)
    f = np.sort(cat.f_gw.to_numpy())
    n_above = np.arange(f.size, 0, -1)
    sel = (f > 2e-4) & (f < 2e-3)
    slope = np.polyfit(np.log(f[sel]), np.log(n_above[sel]), 1)[0]
    assert abs(slope + 8 / 3) < 0.1


def test_local_numbers_match_notebook_order_of_magnitude():
    exp = generator.expected_numbers(_intrinsic(), volume_fraction=gw.volume_fraction_100pc())
    assert 20 < sum(exp.values()) < 120  # notebook: ~50 within 100 pc
    cat = generator.generate_catalogue(_intrinsic(), geometry="local")
    assert (cat.distance <= 0.1).all()
