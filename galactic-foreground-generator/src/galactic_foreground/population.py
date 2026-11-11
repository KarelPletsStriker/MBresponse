"""Sampling binary pairs from the intrinsic mass distribution and classifying them."""

from __future__ import annotations

import numpy as np
from scipy.stats import gaussian_kde

from .constants import HE_CO_SPLIT, NELEMANS_FIT


def histogram_inverse_cdf(sample, bin_width: float = 0.05):
    """Inverse CDF of a histogram distribution (piecewise-uniform), like Mathematica's
    ``InverseCDF[HistogramDistribution[...]]``. Bin width is a free choice; Mathematica's
    automatic binning is not reproducible exactly."""
    sample = np.asarray(sample, float)
    lo = np.floor(sample.min() / bin_width) * bin_width
    hi = np.ceil(sample.max() / bin_width) * bin_width + bin_width
    edges = np.arange(lo, hi + 1e-12, bin_width)
    counts, edges = np.histogram(sample, bins=edges)
    cdf = np.concatenate([[0.0], np.cumsum(counts)]) / counts.sum()
    keep = np.concatenate([[True], np.diff(cdf) > 0])  # drop flat segments for a monotone inverse
    return lambda u: np.interp(u, cdf[keep], edges[keep])


def draw_pairs(intrinsic, n: int = 1000, seed: int | None = 42, bin_width: float = 0.05):
    """Draw n independent (m_a, m_b) pairs; return (m1, m2) sorted so m1 >= m2."""
    rng = np.random.default_rng(seed)
    inv = histogram_inverse_cdf(intrinsic, bin_width)
    a, b = inv(rng.random(n)), inv(rng.random(n))
    return np.maximum(a, b), np.minimum(a, b)


def composition(m1, m2, split: float = HE_CO_SPLIT):
    """Boolean masks for He+He, He+CO and CO+CO (m1 is the more massive component)."""
    m1, m2 = np.asarray(m1), np.asarray(m2)
    hehe = (m1 < split) & (m2 < split)
    heco = (m1 > split) & (m2 < split)
    coco = (m1 > split) & (m2 > split)
    return {"HeHe": hehe, "HeCO": heco, "COCO": coco}


def mass_sum_fractions(m1, m2):
    """Fractions of pairs by total mass: <0.8 (ELM merger), 0.8-1.1, 1.1-1.4, >1.4 (super-Chandra)."""
    s = np.asarray(m1) + np.asarray(m2)
    n = s.size
    return {
        "super_chandra (>1.4)": np.sum(s >= 1.4) / n,
        "elm_merger (<0.8)": np.sum(s <= 0.8) / n,
        "sub_chandra (0.8-1.4)": np.sum((s > 0.8) & (s < 1.4)) / n,
        "0.8-1.1": np.sum((s > 0.8) & (s < 1.1)) / n,
        "1.1-1.4": np.sum((s > 1.1) & (s < 1.4)) / n,
    }


def nelemans_boundary(m1, coeffs=NELEMANS_FIT):
    """m2 above which mass transfer is unstable (merger), fit a + b m1^(1/4)."""
    a, b = coeffs
    return a + b * np.asarray(m1) ** 0.25


def stable_fraction(m1, m2, coeffs=NELEMANS_FIT):
    """Return (n_unstable, n_stable) relative to the Nelemans et al. (2001) boundary."""
    boundary = nelemans_boundary(m1, coeffs)
    m2 = np.asarray(m2)
    return int(np.sum(m2 > boundary)), int(np.sum(m2 < boundary))


def kde_with_hpd_levels(m1, m2, levels=(0.68, 0.95, 0.99), grid_step: float = 0.005, extent=1.4):
    """Gaussian KDE (Silverman bandwidth, like ``SmoothKernelDistribution``) on a grid and the
    density thresholds enclosing the requested probability mass (highest-density regions)."""
    kde = gaussian_kde(np.vstack([m1, m2]), bw_method="silverman")
    x = np.arange(0, extent + 1e-12, grid_step)
    X, Y = np.meshgrid(x, x, indexing="xy")
    Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
    flat = np.sort(Z.ravel())[::-1]
    cdf = np.cumsum(flat) / flat.sum()
    thresholds = [flat[np.searchsorted(cdf, a)] for a in levels]
    return X, Y, Z, thresholds
