"""
Frequency-domain response of a space-based detector (LISA) to a gravitational wave, written as a
Jim `Detector` (https://github.com/GW-JAX-Team/Jim).

The response follows arXiv:2003.00357: the six single-link transfer functions are evaluated at the
time-to-frequency map t(f) of the signal, and combined into the TDI A, E, T channels.
"""

from typing import Optional

import jax
from jax import config
config.update("jax_enable_x64", True)

import jax.numpy as np
from jaxtyping import Array, Complex, Float

from jimgw.core.single_event.data import PowerSpectrum, Data
from jimgw.core.single_event.detector import Detector, GroundBased2G

from . import globals as glob

#: Link indices (receiver, sender), in the order used for the transfer functions.
LINKS = ((1, 2), (2, 3), (3, 1), (1, 3), (3, 2), (2, 1))

#: Obliquity of the ecliptic (J2000), in radians, used to convert (ra, dec) to ecliptic coordinates.
OBLIQUITY = 0.40909280422232897


class SpaceBased(Detector):
    """
    One TDI channel (A, E or T) of a LISA-like space-based detector, compatible with Jim's
    `Data`, `PowerSpectrum` and likelihood classes.

    The spacecraft follow the first-order-in-eccentricity heliocentric orbits of arXiv:2003.00357,
    and the detector response is computed with the full (non long-wavelength) transfer functions,
    evaluated along the time-to-frequency map t(f) of the signal given by the stationary phase
    approximation.

    The source parameters read by `fd_response` are, on top of the waveform ones:
        - sky position: ecliptic ``lambda``, ``beta`` (rad) or, if absent, equatorial ``ra``, ``dec`` (rad)
        - ``psi``: polarization angle (rad)
        - ``trigger_time`` and ``t_c``: the peak of the signal reaches the SSB at ``trigger_time + t_c`` (s)

    The time-to-frequency map is obtained by differentiating the phase of ``waveform``, which must
    be the same waveform model used in the likelihood and expose ``waveform.phase(frequency, params)``
    (as all ripple ``AmplitudePhaseWaveform`` models, e.g. ``ripplegw.waveform("IMRPhenomD")``).

    Args:
        name (str): Name of the detector, e.g. ``"LISA_A"``.
        waveform: Waveform model used to compute the time-to-frequency map.
        channel (str): TDI channel, one of ``"A"``, ``"E"``, ``"T"``.
        arm_length (float): Arm length of the constellation, in m.
        orbit_kappa (float): Initial phase of the constellation guiding center, α(t) = ω(t − orbit_t0) + κ.
        orbit_lambda (float): Initial orientation of the constellation, β_n = 2π(n − 1)/3 + λ.
        orbit_t0 (float): Reference time of the orbit, in s (same time origin as ``trigger_time``).
        reduced_scale (bool): If True, apply the factor (−6iπfL)⁻¹ of eq. (31b) of arXiv:2003.00357,
            expressing the response in units of strain.
        rescaled (bool): If False, convert the TDI variables a, e, t to the rescaled A, E, T of
            eqs. (29a)-(29b) of arXiv:2003.00357. If True, return the unscaled combinations.
    """

    def __init__(
        self,
        name: str,
        waveform,
        channel: str = "A",
        arm_length: float = glob.L,
        orbit_kappa: float = 0.0,
        orbit_lambda: float = 0.0,
        orbit_t0: float = 0.0,
        reduced_scale: bool = False,
        rescaled: bool = True,
    ):
        super().__init__()
        if channel not in ("A", "E", "T"):
            raise ValueError(f"Invalid TDI channel '{channel}'. Valid options are: 'A', 'E', 'T'")

        self.name = name
        self.waveform = waveform
        self.channel = channel

        self.arm_length = arm_length
        self.orbit_kappa = orbit_kappa
        self.orbit_lambda = orbit_lambda
        self.orbit_t0 = orbit_t0
        self.reduced_scale = reduced_scale
        self.rescaled = rescaled

        self.data = Data()
        self.psd = PowerSpectrum()
        self.optimal_snr = None
        self.match_filtered_snr = None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name}, channel={self.channel})"

    # ------------------------------------------------------------------
    # Orbits
    # ------------------------------------------------------------------

    def spacecraft_positions(self, t: Float[Array, " n"]) -> tuple[Float[Array, "3 n 3"], Float[Array, "n 3"]]:
        """
        Positions of the three spacecraft and of the constellation center in the SSB frame.

        Args:
            t: Times at which to evaluate the orbits, in s.

        Returns:
            Positions of shape (3, N, 3), one row per spacecraft, and the constellation center, of shape (N, 3), in m.
        """
        # Angular velocity around the Sun
        T_orbit = 31557600
        omega = 2*np.pi/T_orbit
        alpha = (omega*(t - self.orbit_t0) + self.orbit_kappa)[np.newaxis, :]  # (1, N)
        s = np.sin(alpha)
        c = np.cos(alpha)

        beta = (2*np.arange(3)*np.pi/3 + self.orbit_lambda)[:, np.newaxis]  # (3, 1)
        Re = self.arm_length / (2*np.sqrt(3))  # R * eccentricity

        x = Re * (s*c*np.sin(beta) - (1 + s**2)*np.cos(beta))
        y = Re * (s*c*np.cos(beta) - (1 + c**2)*np.sin(beta))
        z = -Re * np.sqrt(3) * np.cos(alpha - beta)

        p0 = glob.R * np.stack([np.cos(alpha[0]), np.sin(alpha[0]), np.zeros_like(alpha[0])], axis=-1)  # (N, 3)
        return p0[np.newaxis] + np.stack([x, y, z], axis=-1), p0

    # ------------------------------------------------------------------
    # Source geometry
    # ------------------------------------------------------------------

    @staticmethod
    def ecliptic_sky_position(params: dict) -> tuple[Float, Float]:
        """
        Ecliptic longitude and latitude of the source, read from ``lambda``, ``beta`` or converted from ``ra``, ``dec``.
        """
        if "lambda" in params:
            return params["lambda"], params["beta"]
        ra, dec = params["ra"], params["dec"]
        beta = np.arcsin(np.sin(dec)*np.cos(OBLIQUITY) - np.cos(dec)*np.sin(OBLIQUITY)*np.sin(ra))
        lambd = np.arctan2(np.sin(ra)*np.cos(OBLIQUITY) + np.tan(dec)*np.sin(OBLIQUITY), np.cos(ra))
        return lambd, beta

    @staticmethod
    def polarization_tensors(lambd: Float, beta: Float, psi: Float) -> tuple[Float[Array, "3"], dict[str, Float[Array, "3 3"]]]:
        """
        Propagation direction and plus/cross polarization tensors of the wave in the SSB frame (eq. (14) of arXiv:2003.00357).
        """
        # Direction to the source and orthonormal basis of the sky
        n = np.array([np.cos(beta)*np.cos(lambd), np.cos(beta)*np.sin(lambd), np.sin(beta)])
        u = np.array([np.sin(lambd), -np.cos(lambd), 0.0])
        v = np.array([-np.sin(beta)*np.cos(lambd), -np.sin(beta)*np.sin(lambd), np.cos(beta)])

        # Polarization basis, rotated by psi
        p = np.cos(psi)*u + np.sin(psi)*v
        q = -np.sin(psi)*u + np.cos(psi)*v

        e_plus = np.outer(p, p) - np.outer(q, q)
        e_cross = np.outer(p, q) + np.outer(q, p)
        return -n, {"p": e_plus, "c": e_cross}

    def time_to_merger(self, frequency: Float[Array, " n"], params: dict) -> Float[Array, " n"]:
        """
        Time-to-frequency map of the signal relative to its peak, t(f) = −(1/2π) dΦ/df, from the stationary
        phase approximation, with Φ the phase of the waveform (h ∝ exp(iΦ), as in Jim and ripple).
        """
        _, dphase = jax.jvp(lambda f: self.waveform.phase(f, params), (frequency,), (np.ones_like(frequency),))
        return -dphase / (2*np.pi)

    # ------------------------------------------------------------------
    # Response
    # ------------------------------------------------------------------

    def tdi_transfer(self, frequency: Float[Array, " n"], params: dict) -> dict[str, Complex[Array, " n"]]:
        """
        Transfer function of the TDI channel for each polarization, such that the detector strain is
        sum_pol T_pol(f) h_pol(f), with h_pol the frequency-domain polarizations at the SSB.

        The transfer functions are derived in arXiv:2003.00357 with the Fourier convention exp(+2iπft);
        they are conjugated here to match the exp(−2iπft) convention of Jim and ripple.
        """
        lambd, beta = self.ecliptic_sky_position(params)
        k, e_pol = self.polarization_tensors(lambd, beta, params["psi"])

        # Time at which each frequency reaches the SSB
        t = params["trigger_time"] + params["t_c"] + self.time_to_merger(frequency, params)
        p, _ = self.spacecraft_positions(t)

        # Link unit vectors n_rs, pointing from sender s to receiver r, and receiver + sender positions
        pr = np.stack([p[r - 1] for r, _ in LINKS])  # (6, N, 3)
        ps = np.stack([p[s - 1] for _, s in LINKS])
        n = (pr - ps) / np.linalg.norm(pr - ps, axis=-1, keepdims=True)

        fL = frequency * self.arm_length / glob.c
        k_dot_n = np.einsum("i,lki->lk", k, n)
        k_dot_r = np.einsum("i,lki->lk", k, pr + ps)
        link_factor = 1j*np.pi*fL * np.sinc(fL*(1 - k_dot_n)) * np.exp(1j*np.pi*fL*(1 + k_dot_r/self.arm_length))  # (6, N)

        # Exponential delay factor z = exp(2iπfL/c)
        x = np.pi*fL
        z = np.exp(2j*x)

        if self.reduced_scale:
            scale = -1/(6j*x)
        else:
            scale = 1.
        if self.rescaled:
            scale_AE, scale_T = 1., 1.
        else:
            scale_AE = z * (1j*np.sqrt(2)*np.sin(2*x))
            scale_T = np.exp(3j*x) * (2*np.sqrt(2)*np.sin(2*x)*np.sin(x))

        transfer = {}
        for pol, e in e_pol.items():
            T_12, T_23, T_31, T_13, T_32, T_21 = link_factor * np.einsum("lki,ij,lkj->lk", n, e, n)
            if self.channel == "A":
                T = ((1 + z)*(T_31 + T_13) - T_23 - z*T_32 - T_21 - z*T_12) * scale_AE
            elif self.channel == "E":
                T = (1/np.sqrt(3)) * ((1 - z)*(T_13 - T_31) + (2 + z)*(T_12 - T_32) + (1 + 2*z)*(T_21 - T_23)) * scale_AE
            else:
                T = np.sqrt(2/3) * (T_21 - T_12 + T_32 - T_23 + T_13 - T_31) * scale_T
            transfer[pol] = np.conj(T * scale)
        return transfer

    def fd_response(
        self,
        frequency: Float[Array, " n_sample"],
        h_sky: dict[str, Complex[Array, " n_sample"]],
        params: dict,
    ) -> Complex[Array, " n_sample"]:
        """
        Modulate the waveform in the sky frame by the detector response in the frequency domain.

        Args:
            frequency: Array of frequency samples, in Hz.
            h_sky: Frequency-domain polarizations at the SSB, with keys ``"p"`` and ``"c"``, as returned by Jim's waveforms.
            params: Source parameters, see the class docstring.

        Returns:
            Complex strain measured in the TDI channel, with the signal peak at ``trigger_time + t_c`` relative
            to the start of the data segment.
        """
        transfer = self.tdi_transfer(frequency, params)
        strain = sum(transfer[pol] * h_sky[pol] for pol in transfer)

        time_shift = params["trigger_time"] - self.start_time + params["t_c"]
        return strain * np.exp(-2j*np.pi*frequency*time_shift)

    def td_response(self, time, h_sky, params):
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Data and PSD handling, shared with Jim's ground-based detectors
    # ------------------------------------------------------------------

    _equal_data_psd_frequencies = GroundBased2G._equal_data_psd_frequencies
    set_data = GroundBased2G.set_data
    set_psd = GroundBased2G.set_psd
    inject_signal = GroundBased2G.inject_signal
    get_whitened_frequency_domain_strain = GroundBased2G.get_whitened_frequency_domain_strain
    whitened_frequency_to_time_domain_strain = GroundBased2G.whitened_frequency_to_time_domain_strain
    whitened_frequency_domain_data = GroundBased2G.whitened_frequency_domain_data
    whitened_time_domain_data = GroundBased2G.whitened_time_domain_data

    def load_and_set_psd(self, psd_file: str = "", asd_file: str = "") -> PowerSpectrum:
        """
        Load a power spectral density from file and set it to the detector.

        Args:
            psd_file (str, optional): Path to a PSD file (Hz⁻¹).
            asd_file (str, optional): Path to an ASD file (Hz⁻¹/²). Values are squared.

        Returns:
            PowerSpectrum: The loaded PSD, already set on the detector.
        """
        if not (psd_file or asd_file):
            raise ValueError(f"No default PSD is available for {self.name}, provide psd_file or asd_file.")
        psd = PowerSpectrum.from_file(psd_file or asd_file, is_asd=not psd_file)
        psd.name = f"{self.name}_psd"
        self.set_psd(psd)
        return self.psd

    # ------------------------------------------------------------------
    # Instrumental noise model
    # ------------------------------------------------------------------

    @staticmethod
    def S_TM(f: Float[Array, " n"], A: float = 3.) -> Float[Array, " n"]:
        """
        Test-mass acceleration noise PSD of a single link, in fractional frequency units.

        Args:
            f: Frequencies, in Hz.
            A: Acceleration noise amplitude, in units of 1e-15 m s⁻² Hz⁻¹/².
        """
        return A**2 * 1e-30 * (1 + (4e-4/f)**2) * (1 + (f/8e-3)**4) / (2*np.pi*glob.c*f)**2

    @staticmethod
    def S_OMS(f: Float[Array, " n"], P: float = 15.) -> Float[Array, " n"]:
        """
        Optical metrology system noise PSD of a single link, in fractional frequency units.

        Args:
            f: Frequencies, in Hz.
            P: Displacement noise amplitude, in units of 1e-12 m Hz⁻¹/².
        """
        return P**2 * 1e-24 * (1 + (2e-3/f)**4) * (2*np.pi*f/glob.c)**2

    def noise_spectra(
        self, f: Float[Array, " n"], A: float = 3., P: float = 15., basis: str = "AET"
    ) -> tuple[list[Float[Array, " n"]], list[Complex[Array, " n"]]]:
        """
        Noise PSDs and CSDs of the first-generation TDI variables, for an equal-arm constellation in which every
        single link has the same test-mass (``A``) and optical metrology (``P``) noise amplitudes.

        The spectra are those of the standard Michelson variables X, Y, Z and of A = (Z − X)/√2,
        E = (X − 2Y + Z)/√6, T = (X + Y + Z)/√3, before the response normalization of `channel_psd`.

        Args:
            f: Frequencies, in Hz.
            A: Test-mass acceleration noise amplitude of each link, see `S_TM`.
            P: Optical metrology noise amplitude of each link, see `S_OMS`.
            basis: ``"XYZ"`` or ``"AET"``.

        Returns:
            The PSDs [XX, YY, ZZ] and CSDs [XY, YZ, ZX], or [AA, EE, TT] and [AE, ET, TA].
        """
        # Delay operator along one arm
        D = np.exp(-2j*np.pi*f*self.arm_length/glob.c)

        # Spectra of the single links y_ij and of their correlation with y_ji, through the shared test masses
        S_ij_ij = self.S_OMS(f, P) + 2*self.S_TM(f, A)
        S_ij_ji = (np.conj(D) + D) * self.S_TM(f, A)

        # All links are equal, so the three Michelson variables share the same spectra
        XX = 2 * np.abs(1 - D*D)**2 * (2*S_ij_ij + 2*D.real*S_ij_ji.real)
        XY = (1 - np.conj(D*D)) * (D*D - 1) * ((D + np.conj(D))*S_ij_ij + (1 + D*np.conj(D))*S_ij_ji)

        if basis == "XYZ":
            return [XX]*3, [XY]*3
        if basis != "AET":
            raise ValueError(f"Invalid basis '{basis}'. Valid options are: 'XYZ', 'AET'")

        XX, YY, ZZ = [XX]*3
        XY, YZ, ZX = [XY]*3
        AA = (ZZ + XX - 2*ZX.real) / 2
        EE = (XX + 4*YY + ZZ - 4*(XY + YZ - ZX/2).real) / 6
        TT = (XX + YY + ZZ + 2*(XY + YZ + ZX).real) / 3
        AE = (ZZ - XX + 2*ZX.imag + 2*(XY - YZ)) / np.sqrt(12)
        ET = (XX - 2*YY + ZZ + XY - 2*XY.conj() + 2*ZX.real + YZ.conj() - 2*YZ) / np.sqrt(18)
        TA = (ZZ - XX + 2*ZX.imag - XY.conj() + YZ)
        return [AA, EE, TT], [AE, ET, TA]

    def channel_psd(self, f: Float[Array, " n"], A: float = 3., P: float = 15.) -> Float[Array, " n"]:
        """
        Noise PSD of this detector's TDI channel, normalized as the output of `fd_response`.

        `fd_response` returns the TDI variables a, e, t of arXiv:2003.00357, related to the standard A, E, T by
        a = √2 A / (1 − z²), e = √2 E / (1 − z²), t = √2 T / ((1 − z²)(1 − z)) with z = exp(2iπfL), times the
        optional factors set by ``rescaled`` and ``reduced_scale``. The PSD diverges at f = 0, where it is set to infinity.

        Args:
            f: Frequencies, in Hz.
            A: Test-mass acceleration noise amplitude of each link, see `S_TM`.
            P: Optical metrology noise amplitude of each link, see `S_OMS`.
        """
        f_safe = np.where(f > 0, f, 1.)
        psds, _ = self.noise_spectra(f_safe, A, P, basis="AET")
        psd = psds["AET".index(self.channel)]

        x = np.pi * f_safe * self.arm_length / glob.c
        if self.rescaled:
            if self.channel == "T":
                psd = psd / (8 * np.sin(2*x)**2 * np.sin(x)**2)
            else:
                psd = psd / (2 * np.sin(2*x)**2)
        if self.reduced_scale:
            psd = psd / (36 * x**2)
        return np.where(f > 0, psd, np.inf)

    def load_and_set_lisa_psd(
        self, A: float = 3., P: float = 15., frequencies: Optional[Float[Array, " n"]] = None
    ) -> PowerSpectrum:
        """
        Compute the PSD of this detector's TDI channel from the LISA instrumental noise model, with the same
        test-mass and optical metrology noise in every link, and set it to the detector.

        This is an alternative to `load_and_set_psd`.

        Args:
            A: Test-mass acceleration noise amplitude of each link, in units of 1e-15 m s⁻² Hz⁻¹/².
            P: Optical metrology noise amplitude of each link, in units of 1e-12 m Hz⁻¹/².
            frequencies: Frequencies at which to evaluate the PSD, in Hz. Defaults to the frequencies of the data.

        Returns:
            PowerSpectrum: The PSD, already set on the detector.
        """
        if frequencies is None:
            if self.data.is_empty:
                raise ValueError(f"{self.name} has no data, provide the frequencies at which to evaluate the PSD.")
            frequencies = self.frequencies
        psd = PowerSpectrum(self.channel_psd(frequencies, A, P), frequencies, name=f"{self.name}_psd")
        self.set_psd(psd)
        return self.psd


def get_LISA(waveform, **kwargs) -> list[SpaceBased]:
    """
    Return the three TDI channels A, E, T of LISA as `SpaceBased` detectors.

    Args:
        waveform: Waveform model used to compute the time-to-frequency map, see `SpaceBased`.
        **kwargs: Additional arguments passed to `SpaceBased`, e.g. the orbit parameters.
    """
    return [SpaceBased(f"LISA_{channel}", waveform, channel=channel, **kwargs) for channel in ("A", "E", "T")]
