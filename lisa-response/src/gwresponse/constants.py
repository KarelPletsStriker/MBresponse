"""Physical constants and default LISA configuration, in SI units.

Constants shared with ground-based detectors are taken from Jim, so that both
parts of a joint analysis use exactly the same values.
"""

from jimgw.core.constants import C_SI, MTSUN  # noqa: F401  (re-exported)

#: Astronomical unit, in m (IAU 2012). Radius of the guiding-centre orbit of LISA.
AU = 1.495978707e11

#: Light travel time across one astronomical unit, in s.
AU_LIGHT_S = AU / C_SI

#: Julian year, in s. Orbital period of the LISA guiding centre.
YEAR = 31557600.0

#: Nominal arm length of the LISA constellation, in m.
LISA_ARM_LENGTH = 2.5e9

#: Mean obliquity of the ecliptic at J2000 (84381.406 arcsec), in rad.
#: Used to rotate between the equatorial (ICRS) and ecliptic frames.
OBLIQUITY = 0.40909280422232897

#: Default angle by which LISA trails the Earth on its orbit, in rad (20 degrees).
LISA_TRAILING_ANGLE = 0.3490658503988659
