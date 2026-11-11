"""LISA as a Jim detector.

:class:`SpaceBased` is one first-generation TDI channel (A, E or T) of an equal-arm
LISA, implemented as a subclass of Jim's
:class:`~jimgw.core.single_event.detector.Detector`. It holds its own
:class:`~jimgw.core.single_event.data.Data` and
:class:`~jimgw.core.single_event.data.PowerSpectrum`, supports Jim's noise
simulation and signal injection, and works unchanged with Jim's likelihoods
(``TransientLikelihoodFD``, ``HeterodynedTransientLikelihoodFD``), priors,
transforms and samplers.

The response follows arXiv:2003.00357 without the long-wavelength approximation:
the six single-link responses are evaluated with the spacecraft positions at the
time-frequency map t(f) of the signal, and combined into the Michelson X, Y, Z
and the uncorrelated A = (Z - X)/sqrt(2), E = (X - 2Y + Z)/sqrt(6),
T = (X + Y + Z)/sqrt(3). The output is in fractional frequency, consistently with
:mod:`gwresponse.noise`. The Fourier convention is Jim's, exp(-2 i pi f t).

Source parameters are those of Jim's ground-based detectors: the waveform
parameters plus ``ra``, ``dec``, ``psi`` and ``t_c``. With the default
``time_reference="geocenter"``, ``t_c`` is, exactly as for ground-based detectors,
the arrival time of the peak at the geocentre relative to the trigger time of the
data, so LISA and ground-based detectors can be combined in one analysis.
"""

from collections.abc import Mapping
from typing import Optional

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Complex, Float

from jimgw.core.single_event.data import Data, PowerSpectrum
from jimgw.core.single_event.detector import Detector, GroundBased2G
from jimgw.core.single_event.polarization import Polarization
from jimgw.typing import FloatLike

from .constants import AU_LIGHT_S, C_SI, LISA_ARM_LENGTH, MTSUN
from .noise import tdi_noise_psd
from .orbits import (
    LINKS,
    earth_position_velocity,
    lisa_orbit_trailing_earth,
    spacecraft_positions,
)
from .sky import polarization_tensors, source_direction
from .waveform import check_waveform, time_to_merger

TDI_CHANNELS = ("A", "E", "T")
TIME_REFERENCES = ("geocenter", "ssb")

# Frequencies on which an analytic PSD is stored before any data are set
_PLACEHOLDER_FREQUENCIES = np.geomspace(1e-6, 1.0, 4096)


class SpaceBased(Detector):
    """One TDI channel (A, E or T) of a LISA-like space-based detector.

    Args:
        name: Name of the detector, e.g. ``"LISA_A"``.
        channel: TDI channel, ``"A"``, ``"E"`` or ``"T"``.
        arm_length: Arm length of the constellation, in m.
        orbit_kappa: Phase of the guiding centre at ``orbit_t0``, in rad, see
            :func:`gwresponse.orbits.spacecraft_positions`.
        orbit_lambda: Orientation of the constellation, in rad.
        orbit_t0: Reference time of the orbit, in GPS seconds.
        time_reference: ``"geocenter"`` (default): ``t_c`` is the geocentric arrival
            time of the peak, as for Jim's ground-based detectors. ``"ssb"``: ``t_c``
            is the arrival time of the peak at the Solar System Barycentre.
        modes: Polarization modes. Defaults to ``"pc"``.

    Notes:
        The response depends on the waveform model through its time-frequency map;
        only IMRPhenomD (Jim's ``RippleIMRPhenomD``) is currently supported, see
        :mod:`gwresponse.waveform`. Keep the likelihood ``f_max`` below
        c / (2 L) ~ 0.06 Hz, the first zero of the TDI response and noise.
    """

    name: str
    channel: str
    arm_length: float
    orbit_kappa: float
    orbit_lambda: float
    orbit_t0: float
    time_reference: str
    noise_model: Optional[dict[str, float]]

    _geometry_names = ("arm_length", "orbit_kappa", "orbit_lambda", "orbit_t0")

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        name: str,
        channel: str = "A",
        arm_length: float = LISA_ARM_LENGTH,
        orbit_kappa: float = 0.0,
        orbit_lambda: float = 0.0,
        orbit_t0: float = 0.0,
        time_reference: str = "geocenter",
        modes: str = "pc",
    ):
        super().__init__()
        self.name = name
        self.channel = channel
        self.arm_length = arm_length
        self.orbit_kappa = orbit_kappa
        self.orbit_lambda = orbit_lambda
        self.orbit_t0 = orbit_t0
        self.time_reference = time_reference
        self.polarization_mode = [Polarization(m) for m in modes]
        self.noise_model = None
        self.data = Data()
        self.psd = PowerSpectrum()
        self._validate_geometry()

    def _validate_geometry(self) -> None:
        if self.channel not in TDI_CHANNELS:
            raise ValueError(
                f"Invalid TDI channel {self.channel!r} for detector {self.name}, "
                f"valid options are {TDI_CHANNELS}."
            )
        if self.time_reference not in TIME_REFERENCES:
            raise ValueError(
                f"Invalid time_reference {self.time_reference!r} for detector "
                f"{self.name}, valid options are {TIME_REFERENCES}."
            )
        for name in self._geometry_names:
            value = getattr(self, name)
            if not (np.ndim(value) == 0 and np.isfinite(float(value))):
                raise ValueError(f"Detector {self.name} {name} must be a finite real scalar.")
        if not self.arm_length > 0:
            raise ValueError(f"Detector {self.name} arm_length must be positive.")

    def _freeze(self) -> None:
        if self.is_frozen:
            return
        self._validate_geometry()
        for name in self._geometry_names:
            setattr(self, name, float(getattr(self, name)))
        self.polarization_mode = tuple(self.polarization_mode)
        super()._freeze()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name}, channel={self.channel})"

    # ------------------------------------------------------------------
    # Times
    # ------------------------------------------------------------------

    def ssb_delay(self, k: Float[Array, "3"], t_c: FloatLike) -> FloatLike:
        """Arrival time at the SSB minus arrival time at the time reference, in s.

        Args:
            k: Propagation direction of the wave, in ecliptic coordinates.
            t_c: Arrival time of the peak at the time reference, relative to the trigger time.
        """
        if self.time_reference == "ssb":
            return 0.0
        # A plane wave reaches x at t_SSB + k.x / c
        position, velocity = earth_position_velocity(self.data.trigger_time)
        return -jnp.dot(k, jnp.asarray(position) + jnp.asarray(velocity) * t_c) / C_SI

    # ------------------------------------------------------------------
    # Response
    # ------------------------------------------------------------------

    def link_responses(
        self, frequency: Float[Array, " n"], params: Mapping[str, FloatLike]
    ) -> dict[str, Complex[Array, "6 n"]]:
        """Fractional frequency responses y_rs of the six links to each unit polarization.

        For the link from spacecraft s to r (unit vector n, positions p_s and p_r), and
        a plane wave h(t - k.x/c) at the SSB (eq. (3) of arXiv:2003.00357),

        y_rs(f) = -i pi f L/c sinc(f L (1 - k.n)/c) exp(-i pi f (L + k.(p_r + p_s))/c) n.e.n,

        with the positions evaluated at the time t(f) at which frequency f reaches
        the constellation. The links are ordered as :data:`gwresponse.orbits.LINKS`.

        Returns:
            A dict mapping each polarization name to an array of shape (6, N).
        """
        ra, dec, psi = params["ra"], params["dec"], params["psi"]
        k = -source_direction(ra, dec)
        tensors = polarization_tensors(ra, dec, psi, "".join(p.name for p in self.polarization_mode))

        t_peak_ssb = self.data.trigger_time + params["t_c"] + self.ssb_delay(k, params["t_c"])
        t_ssb = t_peak_ssb + time_to_merger(frequency, params)
        orbit = (self.arm_length, self.orbit_kappa, self.orbit_lambda, self.orbit_t0)
        # Evaluate the orbits when the wave reaches the guiding centre, not the SSB
        _, centre = spacecraft_positions(t_ssb, *orbit)
        positions, _ = spacecraft_positions(t_ssb + centre @ k / C_SI, *orbit)

        p_r = jnp.stack([positions[r - 1] for r, _ in LINKS])  # (6, N, 3)
        p_s = jnp.stack([positions[s - 1] for _, s in LINKS])
        n = (p_r - p_s) / jnp.linalg.norm(p_r - p_s, axis=-1, keepdims=True)

        fL = frequency * self.arm_length / C_SI
        k_dot_n = jnp.einsum("i,lki->lk", k, n)
        k_dot_r = jnp.einsum("i,lki->lk", k, p_r + p_s) / self.arm_length
        kernel = (
            -1j
            * jnp.pi
            * fL
            * jnp.sinc(fL * (1 - k_dot_n))
            * jnp.exp(-1j * jnp.pi * fL * (1 + k_dot_r))
        )
        return {
            pol: kernel * jnp.einsum("lki,ij,lkj->lk", n, e, n)
            for pol, e in tensors.items()
        }

    def tdi_transfer(
        self, frequency: Float[Array, " n"], params: Mapping[str, FloatLike]
    ) -> dict[str, Complex[Array, " n"]]:
        """Transfer function of this TDI channel for each polarization.

        The channel output is ``sum_pol T_pol(f) h_pol(f)`` with ``h_pol`` the
        polarizations at the SSB, shifted to the arrival time of the peak.
        """
        delay = jnp.exp(-2j * jnp.pi * frequency * self.arm_length / C_SI)
        transfer = {}
        for pol, y in self.link_responses(frequency, params).items():
            y12, y23, y31, y13, y32, y21 = y
            factor = 1 - delay**2
            X = factor * (y13 + delay * y31 - y12 - delay * y21)
            Y = factor * (y21 + delay * y12 - y23 - delay * y32)
            Z = factor * (y32 + delay * y23 - y31 - delay * y13)
            if self.channel == "A":
                transfer[pol] = (Z - X) / jnp.sqrt(2.0)
            elif self.channel == "E":
                transfer[pol] = (X - 2 * Y + Z) / jnp.sqrt(6.0)
            else:
                transfer[pol] = (X + Y + Z) / jnp.sqrt(3.0)
        return transfer

    def fd_response(
        self,
        frequency: Float[Array, " n_sample"],
        h_sky: dict[str, Complex[Array, " n_sample"]],
        params: Mapping[str, FloatLike],
    ) -> Complex[Array, " n_sample"]:
        """Project the sky-frame polarizations onto this TDI channel.

        Args:
            frequency: Frequencies, in Hz.
            h_sky: Polarizations, with keys ``"p"`` and ``"c"``, as returned by Jim's waveforms.
            params: Source parameters: IMRPhenomD's (``M_c``, ``eta``, ``s1_z``, ``s2_z``, ...),
                ``ra``, ``dec``, ``psi`` and ``t_c``. The trigger time and the start of
                the segment are read from the attached data.

        Returns:
            The TDI channel in the frequency domain, in fractional frequency.
        """
        transfer = self.tdi_transfer(frequency, params)
        strain = sum(transfer[pol] * h_sky[pol] for pol in transfer)
        k = -source_direction(params["ra"], params["dec"])
        time_shift = (
            self.data.trigger_time
            - self.data.start_time
            + params["t_c"]
            + self.ssb_delay(k, params["t_c"])
        )
        return strain * jnp.exp(-2j * jnp.pi * frequency * time_shift)

    def td_response(self, time, h_sky, params):
        raise NotImplementedError

    # ------------------------------------------------------------------
    # PSD
    # ------------------------------------------------------------------

    def noise_psd(self, frequencies: Float[Array, " n"]) -> Float[Array, " n"]:
        """The analytic noise PSD of this channel on ``frequencies``, see :mod:`gwresponse.noise`.

        The PSD diverges at 0 Hz, where it is set to its value at the next frequency.
        """
        if self.noise_model is None:
            raise ValueError(f"Detector {self.name} has no analytic noise model.")
        frequencies = jnp.asarray(frequencies)
        positive = frequencies > 0
        f_safe = jnp.where(positive, frequencies, jnp.min(jnp.where(positive, frequencies, jnp.inf)))
        return tdi_noise_psd(f_safe, self.channel, arm_length=self.arm_length, **self.noise_model)

    def _matched_psd(self, data: Data, psd: PowerSpectrum) -> PowerSpectrum:
        # An analytic PSD is evaluated exactly on the frequencies of the data
        if self.noise_model is not None and not data.is_empty:
            return PowerSpectrum(self.noise_psd(data.frequencies), data.frequencies, psd.name)
        return GroundBased2G._matched_psd(self, data, psd)

    def set_psd(self, psd: PowerSpectrum | Array, **kws) -> None:
        """Replace the PSD of the detector, as for Jim's ground-based detectors."""
        self._check_mutable()
        self.noise_model = None
        GroundBased2G.set_psd(self, psd, **kws)

    def load_and_set_psd(
        self, psd_file: str = "", asd_file: str = "", A: float = 3.0, P: float = 15.0
    ) -> PowerSpectrum:
        """Set the PSD of the detector from a file, or from the analytic LISA noise model.

        Args:
            psd_file: Path to a PSD file (1/Hz), in the units of the TDI output.
            asd_file: Path to an ASD file (Hz^-1/2). Values are squared.
            A: Without a file, test-mass acceleration noise in 1e-15 m s^-2 Hz^-1/2.
            P: Without a file, optical metrology noise in 1e-12 m Hz^-1/2.

        Returns:
            The PSD, already set on the detector. The analytic PSD is evaluated
            exactly on the frequencies of any data set later.
        """
        self._check_mutable()
        if psd_file and asd_file:
            raise ValueError("psd_file and asd_file are mutually exclusive; set at most one.")
        if psd_file or asd_file:
            psd = PowerSpectrum.from_file(psd_file or asd_file, is_asd=not psd_file)
            psd.name = f"{self.name}_psd"
            self.set_psd(psd)
            return self.psd
        self.noise_model = {"A": float(A), "P": float(P)}
        frequencies = (
            jnp.asarray(_PLACEHOLDER_FREQUENCIES) if self.data.is_empty else self.data.frequencies
        )
        self.psd = PowerSpectrum(self.noise_psd(frequencies), frequencies, f"{self.name}_psd")
        return self.psd

    # ------------------------------------------------------------------
    # Data, noise simulation and injection, shared with Jim's ground-based detectors
    # ------------------------------------------------------------------

    set_data = GroundBased2G.set_data
    _zero_data = GroundBased2G._zero_data
    set_zero_noise = GroundBased2G.set_zero_noise
    simulate_and_set_noise = GroundBased2G.simulate_and_set_noise
    get_whitened_frequency_domain_strain = GroundBased2G.get_whitened_frequency_domain_strain
    whitened_frequency_to_time_domain_strain = GroundBased2G.whitened_frequency_to_time_domain_strain
    whitened_frequency_domain_data = GroundBased2G.whitened_frequency_domain_data
    whitened_time_domain_data = GroundBased2G.whitened_time_domain_data

    def inject_signal(self, waveform_model, parameters: dict[str, float], allow_wraparound: bool = False):
        """Add a signal to the data, as :meth:`GroundBased2G.inject_signal`.

        Returns:
            The optimal SNR of the signal and its matched-filter SNR.
        """
        check_waveform(waveform_model)
        return GroundBased2G.inject_signal(self, waveform_model, parameters, allow_wraparound)

    def _check_signal_duration(self, waveform_model, params, trigger_time, allow_wraparound) -> None:
        """Check that the signal fits the segment, allowing for the light time across 1 AU."""
        eta = float(params["eta"])
        mass_seconds = float(params["M_c"]) * MTSUN / eta**0.6
        chi = max(abs(float(params["s1_z"])), abs(float(params["s2_z"])))
        f_start = self.data.frequency_bounds[0]
        v = (np.pi * mass_seconds * f_start) ** (1 / 3)
        # XLALSimInspiralChirpTimeBound in LAL's LALSimInspiral.c, as in Jim
        chirp_time = 5 * mass_seconds / (256 * eta * v**8) * (
            1
            + (743 / 252 + 11 * eta / 3) * v**2
            + (226 / 15) * chi * v**3
            + (3058673 / 508032 + 5429 * eta / 504 + 617 * eta**2 / 72) * v**4
        )
        padding = 1 / f_start + AU_LIGHT_S
        before = chirp_time + 200 * mass_seconds + padding
        after = 1000 * mass_seconds + padding
        peak = trigger_time - self.data.start_time + float(params["t_c"])
        if peak >= before and peak + after <= float(self.data.duration):
            return
        message = (
            f"The estimated signal duration does not fit the segment of detector "
            f"{self.name}: allow {before:.0f} s before and {after:.0f} s after the "
            f"peak, which is {peak:.0f} s into the {float(self.data.duration):.0f} s "
            "segment. Lengthen the segment, raise f_min, or pass allow_wraparound=True."
        )
        if not allow_wraparound:
            raise ValueError(message)
        import logging

        logging.getLogger(__name__).warning(message)


def get_LISA(
    channels: str = "AET",
    trigger_time: Optional[float] = None,
    **kwargs,
) -> list[SpaceBased]:
    """The TDI channels of LISA, as Jim detectors.

    Args:
        channels: Channels to return, any of ``"A"``, ``"E"``, ``"T"``. The T channel
            carries little signal below ~10 mHz and is often left out.
        trigger_time: If given, place the constellation 20 degrees behind the Earth at
            this GPS time (see :func:`gwresponse.orbits.lisa_orbit_trailing_earth`),
            unless ``orbit_kappa`` is passed explicitly.
        **kwargs: Passed to :class:`SpaceBased`, e.g. ``arm_length`` or ``time_reference``.
    """
    if trigger_time is not None and "orbit_kappa" not in kwargs:
        kwargs = {**lisa_orbit_trailing_earth(trigger_time), **kwargs}
    return [SpaceBased(f"LISA_{channel}", channel=channel, **kwargs) for channel in channels]
