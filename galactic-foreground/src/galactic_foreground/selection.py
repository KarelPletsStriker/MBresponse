"""Luminosity-based selection function: observed -> intrinsic WD mass distribution."""

from __future__ import annotations

import numpy as np

from .constants import HE_CO_SPLIT


def radius_he(m, T4):
    """Analytic He-core WD radius fit (Rsun); m in Msun, T4 = T / 1e4 K."""
    s = np.sqrt(T4)
    return 0.0132 * 10 ** (-0.177 * s) * m ** (0.148 - 0.941 * s)


def radius_co(m, T4):
    """Analytic CO-core WD radius fit (Rsun); m in Msun, T4 = T / 1e4 K."""
    return 0.9 * 10 ** (-0.00063 * T4**2) * m ** (-0.5 - 0.13 * T4) / 100


def selection_repeats(masses, temps, split: float = HE_CO_SPLIT, scale: float = 1000.0):
    """Number of times each observed WD is repeated to undo the selection bias.

    Step 1: down-weight unusually hot WDs by L_ref / L, where L_ref uses the He-sample mean
    temperature in the radius. Step 2: up-weight WDs that are intrinsically fainter than a
    reference WD (mean He mass and temperature). Returns (repeats, ref_mass, ref_temp4)
    with repeats truncated to integers as Mathematica's Table does.
    """
    masses, temps = np.asarray(masses, float), np.asarray(temps, float)
    he = masses < split
    T4 = temps * 1e-4
    ref_temp4 = temps[he].mean() * 1e-4
    ref_mass = masses[he].mean()

    radius = np.where(he, radius_he(masses, T4), radius_co(masses, T4))
    radius_ref = np.where(he, radius_he(masses, ref_temp4), radius_co(masses, ref_temp4))
    lum = radius**2 * temps**4
    lum_ref = radius_ref**2 * temps**4

    l20 = radius_he(ref_mass, ref_temp4) ** 2 * (ref_temp4 * 1e4) ** 4
    lsize = l20 / (radius_ref**2 * (ref_temp4 * 1e4) ** 4)

    repeats = np.floor(scale * (lum_ref / lum) * lsize).astype(int)
    return repeats, ref_mass, ref_temp4


def intrinsic_masses(masses, temps, split: float = HE_CO_SPLIT):
    """Selection-corrected mass sample (the notebook's ``Join[Hescaled, COscaled]``)."""
    repeats, _, _ = selection_repeats(masses, temps, split)
    masses = np.asarray(masses, float)
    he = masses < split
    return np.concatenate([np.repeat(masses[he], repeats[he]), np.repeat(masses[~he], repeats[~he])])
