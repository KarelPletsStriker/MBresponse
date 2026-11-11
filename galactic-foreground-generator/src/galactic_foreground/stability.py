"""Boundary between stable and unstable mass transfer for DWDs (direct-impact/Eddington limit).

Solves  Mdot_acc(m1, m2) = -Mdot_Edd(m1)  for m2 on a grid of m1, as in the notebook.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from .constants import C, G, MSOL, RSOL

ZETA_2_COLD = -1.0 / 3.0  # polytropic donor response (Soberman et al. 1997)


def radius_cold(m):
    """Eggleton's zero-temperature mass-radius relation (Verbunt & Rappaport 1988), Rsun."""
    mch, mp = 1.44, 5.7e-4
    return (0.0114 * np.sqrt((m / mch) ** (-2 / 3) - (m / mch) ** (2 / 3))
            * (1 + 3.5 * (m / mp) ** (-2 / 3) + mp / m) ** (-2 / 3))


def zeta_roche(m1, m2):
    """d ln R_L / d ln m2 for the Eggleton (1983)-based expression used in the notebook."""
    s = m1 + m2
    return 1.5 * 3 ** (1 / 3) * m2 ** (2 / 3) * s ** (1 / 3) * (
        -2 * m2 ** (1 / 3) / (9 * 3 ** (1 / 3) * s ** (4 / 3))
        + 2 / (9 * 3 ** (1 / 3) * m2 ** (2 / 3) * s ** (1 / 3))
    )


def jdot_over_j(m1, m2, r):
    """GW angular-momentum loss rate J̇/J (1/s); masses in g, donor radius r in cm (Paczynski R_L)."""
    a = (r / 0.46) * ((m1 + m2) / m2) ** (1 / 3)
    return -32 / 5 * G**3 / C**5 * m1 * m2 * (m1 + m2) / a**4


def mdot_eddington(m1):
    """Accretor Eddington-like limit used in the notebook (g/s); m1 in Msun."""
    return 5 * 4.4e-6 * m1**2 / (np.pi * 1e7) * MSOL


def mdot_transfer(m1, m2):
    """Mass-transfer rate (g/s, negative) for masses in g."""
    jj = jdot_over_j(m1, m2, radius_cold(m2 / MSOL) * RSOL)
    return jj * m2 / (1 + ZETA_2_COLD / 2 - zeta_roche(m1, m2) / 2 - m2 / m1)


def boundary_m2(m1: float, n_scan: int = 4000) -> float:
    """Smallest m2 (Msun) at which |Mdot| reaches the Eddington-like limit for accretor m1."""
    f = lambda m2: mdot_transfer(MSOL * m1, MSOL * m2) + mdot_eddington(m1)
    grid = np.linspace(1e-4, m1, n_scan)
    vals = np.array([f(x) for x in grid])
    idx = np.where(np.sign(vals[:-1]) != np.sign(vals[1:]))[0]
    if idx.size == 0:
        return np.nan
    i = idx[0]
    return brentq(f, grid[i], grid[i + 1])


def boundary_curve(m1_grid=None):
    m1_grid = np.arange(0.0001, 1.4, 0.01) if m1_grid is None else np.asarray(m1_grid)
    return m1_grid, np.array([boundary_m2(m) for m in m1_grid])


def fit_quarter_power(m1, m2):
    """Least-squares fit m2 = a + b m1^(1/4) (the notebook's LinearModelFit)."""
    ok = np.isfinite(m2)
    A = np.vstack([np.ones(ok.sum()), m1[ok] ** 0.25]).T
    (a, b), *_ = np.linalg.lstsq(A, m2[ok], rcond=None)
    return a, b
