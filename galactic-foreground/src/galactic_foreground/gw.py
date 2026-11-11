"""Gravitational-wave inspiral bookkeeping: chirp masses, residence times, Galactic numbers."""

from __future__ import annotations

import numpy as np

from .constants import C, G, MSOL, YEAR


def chirp_mass(m1, m2):
    return (np.asarray(m1) * m2) ** 0.6 / (np.asarray(m1) + m2) ** 0.2


def _k(mchirp):
    """dfdt = K f^(11/3) with f in Hz, Mchirp in Msun."""
    return 96 / 5 * np.pi ** (8 / 3) * G ** (5 / 3) * (mchirp * MSOL) ** (5 / 3) / C**5


def dfdt(f_gw, mchirp):
    """GW frequency derivative (Hz/s) for f_gw in Hz."""
    return _k(mchirp) * np.asarray(f_gw) ** (11 / 3)


def time_to_merger(f_gw, mchirp):
    """Integral of 1/(df/dt) from f_gw to infinity (s)."""
    return 3 / 8 / _k(mchirp) * np.asarray(f_gw) ** (-8 / 3)


def fmin_steady_state(t_steady=1e9 * YEAR, mchirp=0.5):
    """GW frequency (Hz) below which binaries take longer than t_steady to merge."""
    return (3 / 8 / _k(mchirp) / t_steady) ** (3 / 8)


def cumulative_coefficient(rate_per_yr, mchirp, volume_fraction=1.0):
    """Coefficient A in N(>f) = A f^(-8/3) with f in mHz, for a steady merger rate."""
    return rate_per_yr / YEAR * volume_fraction * time_to_merger(1e-3, mchirp)


def number_in_band(rate_per_yr, mchirp, f_lo, f_hi, volume_fraction=1.0):
    """Number of binaries with f_lo < f_gw < f_hi (Hz) for a steady merger rate."""
    return rate_per_yr / YEAR * volume_fraction * (time_to_merger(f_lo, mchirp) - time_to_merger(f_hi, mchirp))


def volume_fraction_100pc(disc_radius_kpc=15.0, disc_height_kpc=1.0, local_radius_kpc=0.1):
    """Fraction of a uniform cylindrical Galactic disc inside a local sphere."""
    v_mw = np.pi * disc_radius_kpc**2 * disc_height_kpc
    return 4 / 3 * np.pi * local_radius_kpc**3 / v_mw
