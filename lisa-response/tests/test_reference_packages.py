"""Cross-checks against the reference LISA packages this code is adapted from.

These tests are skipped unless the reference package is installed
(``pip install lisaanalysistools``).
"""

import warnings

import numpy as np
import pytest

from gwresponse.noise import tdi_noise_psd


@pytest.mark.parametrize("channel", ["X", "A", "E", "T"])
def test_tdi_noise_matches_lisatools_scird(channel):
    """First-generation TDI noise PSDs equal lisatools' ``scirdv1`` model."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sensitivity = pytest.importorskip("lisatools.sensitivity")
    reference = {
        "X": sensitivity.X1TDISens,
        "A": sensitivity.A1TDISens,
        "E": sensitivity.E1TDISens,
        "T": sensitivity.T1TDISens,
    }[channel]
    f = np.logspace(-5, np.log10(0.05), 400)
    expected = sensitivity.get_sensitivity(f, sens_fn=reference, model="scirdv1")
    np.testing.assert_allclose(np.asarray(tdi_noise_psd(f, channel)), expected, rtol=1e-8)
