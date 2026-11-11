# GWResponse: LISA for Jim

`gwresponse` makes LISA a first-class detector in [Jim](https://github.com/GW-JAX-Team/Jim), the JAX-based gravitational-wave inference framework. The LISA TDI channels A, E and T are Jim `Detector`s. Jim's whole pipeline therefore works for space-based data as it does for LIGO/Virgo: data containers, noise simulation, injections, `TransientLikelihoodFD` / `HeterodynedTransientLikelihoodFD`, priors, transforms and samplers. A `MultibandLikelihood` combines LISA with ground-based detectors in one analysis.

The waveform model is **IMRPhenomD** (Jim's `RippleIMRPhenomD`, from [ripple](https://github.com/GW-JAX-Team/ripple)). It is currently the only supported model.

The response follows [arXiv:2003.00357](https://arxiv.org/abs/2003.00357) without the long-wavelength approximation. The single-link responses are evaluated along the time–frequency map t(f) of the signal, with the spacecraft on analytic heliocentric orbits, and combined into first-generation TDI.

## Installation

```bash
git clone https://github.com/KarelPletsStriker/MBresponse.git
cd MBresponse/lisa-response
pip install -e ".[test,examples]"
```

This installs `jimgw` (≥ 0.6.1) and `ripplegw`. JAX runs in 64-bit mode, which Jim requires.

## LISA parameter estimation

The workflow is the same as for ground-based detectors. Only the detectors change:

```python
import jax
from jimgw.core.single_event.likelihood import HeterodynedTransientLikelihoodFD
from jimgw.core.single_event.waveform import RippleIMRPhenomD
from gwresponse import get_LISA

waveform = RippleIMRPhenomD(f_ref=3e-4)
gps = 1.4e9
params = {"M_c": 4e5, "eta": 0.24, "s1_z": 0.3, "s2_z": -0.1, "d_L": 3e4, "iota": 0.5,
          "phase_c": 0.3, "ra": 1.0, "dec": 0.3, "psi": 0.7, "t_c": 100.0}

ifos = get_LISA("AE", trigger_time=gps)  # LISA trails the Earth by 20 deg at gps
for ifo in ifos:
    ifo.load_and_set_psd()               # analytic SciRD noise of the TDI channel
    ifo.simulate_and_set_noise(duration=2**21, sampling_frequency=0.1, trigger_time=gps,
                               f_min=3e-4, f_max=0.05, post_trigger_duration=2e4,
                               rng_key=jax.random.key(1))
    ifo.inject_signal(waveform, params)

likelihood = HeterodynedTransientLikelihoodFD(ifos, waveform, f_min=3e-4, f_max=0.05,
                                              n_bins=500, reference_parameters=params,
                                              phase_marginalization=True)
```

Then pass `likelihood` to `jimgw.core.jim.Jim` with a prior and transforms. [`examples/lisa_injection.py`](examples/lisa_injection.py) is the complete script, mirroring Jim's own `examples/injection.py`. Jim's detector-frame time and sky transforms are specific to ground-based detectors. So for LISA alone, use `get_default_sample_transforms(prior, ifos, detector_frame_time=False, detector_frame_sky=False)`.

## Joint LISA + ground-based likelihood

Jim requires every detector in a likelihood to share one frequency spacing, i.e. data of one duration. LISA data (weeks, sampled at ~0.1 Hz) and LIGO data (seconds, at kHz) cannot share a single `TransientLikelihoodFD`. `MultibandLikelihood` adds the log-likelihoods of one Jim likelihood per band, which is exact for independent noise:

```python
from jimgw.core.single_event.detector import get_H1, get_L1, get_V1
from gwresponse import MultibandLikelihood

likelihood = MultibandLikelihood([
    HeterodynedTransientLikelihoodFD(lisa_ifos, waveform, f_min=0.025, f_max=0.058, ...),
    HeterodynedTransientLikelihoodFD([get_H1(), get_L1(), get_V1()], waveform, f_min=10, f_max=1024, ...),
])
```

All bands see one set of source parameters. [`examples/joint_lisa_ground.py`](examples/joint_lisa_ground.py) analyses a multiband binary (𝓜 = 150 M☉ at 300 Mpc): LISA observes it ~6 weeks before merger with SNR ≈ 21, and HLV observe the merger with SNR ≈ 380.

`MultibandLikelihood` refuses per-band marginalization over `t_c`, `phase_c` or `d_L`. Those parameters are shared, and marginalizing them separately in each band would treat them as independent. It also requires one trigger time across bands, and IMRPhenomD for any band containing LISA.

## Conventions

These conventions are what make LISA and ground-based detectors consistent in one analysis.

| Quantity | Convention |
|---|---|
| Sky position | Jim's equatorial `ra`, `dec`. Converted internally to the ecliptic frame (J2000 obliquity). `ecliptic_to_equatorial` / `equatorial_to_ecliptic` are provided. |
| Polarization `psi` | Jim's own `Polarization` tensors, rotated to the ecliptic frame, so `psi` means the same as for H1/L1/V1. |
| Time `t_c` | Default `time_reference="geocenter"`: arrival time of the peak **at the geocentre** relative to the data's trigger time, exactly as in Jim. The Roemer delay to the barycentre uses astropy's built-in Earth ephemeris. `time_reference="ssb"` references `t_c` to the barycentre instead. |
| Fourier convention | Jim's, `exp(-2iπft)`. |
| TDI output | Fractional frequency. Michelson X, Y, Z → A = (Z−X)/√2, E = (X−2Y+Z)/√6, T = (X+Y+Z)/√3. The analytic PSDs in `gwresponse.noise` use the same normalization. |
| Orbits | Rigid equal-arm triangle on first-order-in-eccentricity heliocentric orbits (arXiv:2003.00357 eqs. 35–36). Orientation set by `orbit_kappa`, `orbit_lambda`, `orbit_t0`, or by `get_LISA(trigger_time=...)`, which puts LISA 20° behind the Earth. |

## Package layout

| Module | Contents |
|---|---|
| `gwresponse.detector` | `SpaceBased(Detector)`: one TDI channel; `get_LISA()` |
| `gwresponse.likelihood` | `MultibandLikelihood` |
| `gwresponse.waveform` | IMRPhenomD time–frequency map `time_to_merger`; `check_waveform` |
| `gwresponse.noise` | SciRD single-link noises, TDI PSDs (X, A, E, T), Robson et al. reference sensitivity |
| `gwresponse.orbits` | Spacecraft orbits, Earth ephemeris, `lisa_orbit_trailing_earth` |
| `gwresponse.sky` | Frame rotations and polarization tensors |

## Validation

`pytest` runs the test suite (about 6 minutes on a CPU). It checks:

- `time_to_merger` equals −(1/2π) dφ/df of ripple's IMRPhenomD phase to machine precision, on uniform and non-uniform grids.
- The A/E/T noise PSDs equal the combinations of the Michelson auto- and cross-spectra.
- The sky- and polarization-averaged sensitivity of the response plus noise reproduces Robson et al. (2019) eq. 13 (a factor of 2 for one channel against their two-channel curve) to within 4% for 0.1–10 mHz.
- In the long-wavelength limit, the X channel equals Jim's ground-based antenna-pattern formula for an interferometer with LISA's arms. This ties the LISA response and Jim's detectors to one polarization and sign convention.
- Geocentric and barycentric time references give identical responses after the Roemer delay.
- A zero-noise injection gives log L = SNR²/2 with zero gradient at the truth. The heterodyned likelihood matches the full one. `MultibandLikelihood` adds its bands and rejects invalid set-ups.
- Jim + flowMC run end-to-end on a joint LISA + LIGO likelihood.
- With `pip install -e ".[reference]"`: the TDI noise PSDs equal lisatools' SciRD `scirdv1` model to 1e-8.

[`VALIDATION.md`](VALIDATION.md) lists the planned cross-checks against BBHx, lisaorbits, LDC's lisagwresponse/pytdi, LAL and astropy.

## Approximations and limitations

- IMRPhenomD (2,2) only. Other models need their own time–frequency map in `gwresponse.waveform`.
- First-generation TDI, equal and constant arms, no light-travel time along the orbit within a link, no galactic confusion noise. Keep `f_max` below c/(2L) ≈ 0.06 Hz, the first zero of the TDI response and noise.
- Jim's time marginalization treats `t_c` as a pure phase shift. For LISA this neglects the change in orbital position over the `t_c` range, so prefer sampling `t_c`.
- Not a substitute for a full LISA data model (unequal arms, second-generation TDI, realistic orbits).

## Origin

The LISA response was developed by Mireia Egido for her Bachelor's thesis at the University of Barcelona, with ICE-CSIC, under the supervision of Dr. Carlos F. Sopuerta, and validated against BBHx. It has been rewritten here as a Jim detector. The previous stand-alone `LISA_response` API, the custom IMRPhenomD implementation and its data files have been removed in favour of ripple and Jim. They remain available in the git history.
