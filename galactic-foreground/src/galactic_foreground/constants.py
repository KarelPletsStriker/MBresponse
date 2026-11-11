"""Constants exactly as used in the original notebook (cgs, rounded values)."""

import numpy as np

MSOL = 2e33          # g
RSOL = 6.955e10      # cm
G = 6.67e-8          # cgs
C = 3e10             # cm / s
YEAR = np.pi * 1e7   # s  (the notebook's "pi 10^7")

HE_CO_SPLIT = 0.45   # Msun: masses below are treated as He-core, above as CO-core

# Nelemans et al. (2001) stable / unstable mass-transfer boundary, fitted in the
# notebook as m2 = a + b * m1**(1/4).  Hard-coded values from the notebook.
NELEMANS_FIT = (-0.17871147963385745, 0.49072476542885)
