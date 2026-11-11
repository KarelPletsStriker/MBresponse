"""galactic_foreground: Python port of Lucy McNeill's Mathematica notebook
``paper-calculations.nb`` on double white dwarf (DWD) merger rates and the
Galactic DWD population.

Original analysis and notebook: Lucy McNeill.
"""

from . import constants, data, gw, population, selection, stability

__all__ = ["constants", "data", "gw", "population", "selection", "stability"]
__version__ = "0.1.0"
