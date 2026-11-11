"""Draw a synthetic catalogue of Galactic double white dwarfs (the GW foreground).

Extension of Lucy McNeill's analysis: the notebook gives the steady-state number of
binaries N(>f) per composition; here individual binaries are drawn from the same model.

Model (steady state, constant merger rate R_c per composition c):
  * masses: pairs from the selection-corrected intrinsic mass distribution, split by
    composition (He+He, He+CO, CO+CO);
  * frequency: dN/df = R_c / (df/dt) ∝ Mchirp^(-5/3) f^(-11/3), so pairs are weighted by
    Mchirp^(-5/3) and f_GW is drawn from a truncated f^(-11/3) power law;
  * position: uniform in the notebook's Galactic disc (cylinder, radius 15 kpc, height 1 kpc),
    or uniform in a sphere around the Sun ("100pc" mode).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import gw, population
from .constants import YEAR

DEFAULT_RATES = {"HeHe": 0.0005, "HeCO": 0.003, "COCO": 0.003}  # per year, from the notebook


def _power_law_f(rng, n, f_lo, f_hi):
    """Draw f from p(f) ∝ f^(-11/3) on [f_lo, f_hi]."""
    a, b = f_lo ** (-8 / 3), f_hi ** (-8 / 3)
    return (a - rng.random(n) * (a - b)) ** (-3 / 8)


def _positions(rng, n, geometry, disc_radius, disc_height, r_sun, local_radius):
    if geometry == "galaxy":
        r = disc_radius * np.sqrt(rng.random(n))
        phi = 2 * np.pi * rng.random(n)
        x, y = r * np.cos(phi), r * np.sin(phi)
        z = disc_height * (rng.random(n) - 0.5)
        d = np.sqrt((x - r_sun) ** 2 + y**2 + z**2)
        return x, y, z, d
    if geometry == "local":
        v = rng.normal(size=(3, n))
        v /= np.linalg.norm(v, axis=0)
        d = local_radius * rng.random(n) ** (1 / 3)
        x, y, z = v * d
        return x + r_sun, y, z, d
    raise ValueError("geometry must be 'galaxy' or 'local'")


def expected_numbers(intrinsic, f_lo=1.5e-4, f_hi=0.5, rates=DEFAULT_RATES, volume_fraction=1.0,
                     n_pairs=200_000, seed=0):
    """Expected number of binaries per composition with f_lo < f_GW < f_hi (Hz)."""
    m1, m2 = population.draw_pairs(intrinsic, n=n_pairs, seed=seed)
    comp = population.composition(m1, m2)
    mc = gw.chirp_mass(m1, m2)
    return {c: rates[c] / YEAR * volume_fraction
            * np.mean(gw.time_to_merger(f_lo, mc[comp[c]]) - gw.time_to_merger(f_hi, mc[comp[c]]))
            for c in rates}


def generate_catalogue(intrinsic, n=None, f_lo=1.5e-4, f_hi=0.5, geometry="galaxy",
                       rates=DEFAULT_RATES, disc_radius=15.0, disc_height=1.0, r_sun=8.2,
                       local_radius=0.1, seed=42, pool_size=200_000):
    """Return a DataFrame of DWDs: m1, m2 [Msun], composition, mchirp, f_gw, f_orb [Hz],
    P_orb [min], x, y, z, distance [kpc].

    n=None draws a Poisson realisation of the full steady-state population in the chosen
    volume and frequency band (millions of rows for the whole Galaxy at f_lo=0.15 mHz);
    an integer n draws that many binaries with the same relative proportions.
    """
    rng = np.random.default_rng(seed)
    vf = 1.0 if geometry == "galaxy" else gw.volume_fraction_100pc(disc_radius, disc_height, local_radius)
    m1p, m2p = population.draw_pairs(intrinsic, n=pool_size, seed=rng.integers(2**32))
    comp = population.composition(m1p, m2p)
    mcp = gw.chirp_mass(m1p, m2p)

    expected = expected_numbers(intrinsic, f_lo, f_hi, rates, vf, seed=int(rng.integers(2**32)))
    total = sum(expected.values())
    if n is None:
        counts = {c: rng.poisson(lam) for c, lam in expected.items()}
    else:
        counts = dict(zip(expected, rng.multinomial(n, [expected[c] / total for c in expected])))

    frames = []
    for c, k in counts.items():
        idx = np.flatnonzero(comp[c])
        w = mcp[idx] ** (-5 / 3)
        pick = rng.choice(idx, size=k, p=w / w.sum())
        f = _power_law_f(rng, k, f_lo, f_hi)
        x, y, z, d = _positions(rng, k, geometry, disc_radius, disc_height, r_sun, local_radius)
        frames.append(pd.DataFrame({"m1": m1p[pick], "m2": m2p[pick], "composition": c,
                                    "mchirp": mcp[pick], "f_gw": f, "f_orb": f / 2,
                                    "P_orb_min": 2 / f / 60, "x": x, "y": y, "z": z, "distance": d}))
    return pd.concat(frames, ignore_index=True)
