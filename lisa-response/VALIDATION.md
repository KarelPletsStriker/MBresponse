# Validation plan: correspondence with the reference packages

`gwresponse` adapts five things from other code or papers. Each one needs a test against an independent reference implementation, not just against its own formulas.

| Component | Adapted from | Reference implementation (pip) |
|---|---|---|
| TDI response | Marsat, Baker & Dal Canton 2021 (arXiv:2003.00357), as implemented in BBHx | `bbhx` 1.2 (CPU wheels available) |
| Orbits | arXiv:2003.00357 eqs. 35–36 | `lisaorbits` 3.x, `lisatools.detector.EqualArmlengthOrbits` |
| Instrument noise | SciRD, LDC, Robson et al. 2019 | `lisaanalysistools` (`lisatools.sensitivity`) |
| Waveform and t(f) | ripple IMRPhenomD | `lalsimulation` (`IMRPhenomD`), ripple's own tests |
| Detector API, likelihood | Jim `Detector`, `TransientLikelihoodFD` | `jimgw` |

The reference packages go in an optional extra (`pip install -e ".[reference]"`). Each test calls `pytest.importorskip`, so the default suite still runs without them. `tests/test_reference_packages.py` already holds test N1.

Tolerances below are the targets for passing. Use mismatch 1 − ⟨a|b⟩/√(⟨a|a⟩⟨b|b⟩) with the A-channel PSD, over the band analysed.

## Noise

**N1. TDI PSDs = lisatools SciRD (implemented, passing).**
- Compare `tdi_noise_psd(f, ch)` with `get_sensitivity(f, sens_fn={X,A,E,T}1TDISens, model="scirdv1")` for f ∈ [1e-5, 0.05] Hz.
- Result: agreement to `rtol=1e-8` in all four channels.

**N2. Cross-spectra = lisatools.**
- Compare the X–Y cross-spectrum S_XY with `XY1TDISens`.
- This pins the sign convention used to show that A, E and T are uncorrelated.

**N3. Noise realisation statistics.**
- Simulate noise with `SpaceBased.simulate_and_set_noise` and check that `mean(|n(f)|² · 4 df / S(f)) = 2` in each frequency band.
- Check that A and E are uncorrelated.
- This catches the one-sided/two-sided and df-normalisation conventions inherited from Jim.

## Response (the most important tests)

**R1. Transfer functions vs BBHx, with orbits held identical.**
- Purpose: isolate the response maths from the orbit model.
- Write our analytic spacecraft positions to a `lisaorbits`-format HDF5 file and load it into BBHx through `lisatools.detector.Orbits(filename)`.
- Feed BBHx's `LISATDIResponse(TDItag="AET", tdi2=False)` the same frequencies, IMRPhenomD phase and t(f) used by `SpaceBased`.
- Compare the complex transfer functions at ~10 sky positions × 3 polarisations × 3 masses (1e5, 1e6, 1e7 M☉).
- Target: |ΔT|/|T| < 1e-6, i.e. agreement to interpolation accuracy.
- Expected differences that must be mapped, not tolerated:
  - BBHx works in ecliptic λ, β; we use equatorial ra, dec.
  - BBHx's ψ is ecliptic; ours is Jim's equatorial ψ.
  - BBHx's t_ref is at the SSB; use `time_reference="ssb"`.
  - Check any overall `exp(±iπ/2)`, sign or rescaling factor in BBHx's TDI output; read it from the source and document it.
  - Any residual mismatch localises a bug to `link_responses` / `tdi_transfer`.

**R2. Full waveform vs BBHx `BBHWaveformFD`.**
- Use `PhenomHM` with `modes=[(2,2)]`, which is PhenomD's (2,2) mode, on the same orbit file.
- Compare it with `inject_signal` output of A and E on a ~1-month segment.
- Target: mismatch < 1e-4 and |ΔSNR|/SNR < 1e-3.
- This also tests the parameter mapping, including M_c/η → m1/m2, d_L in Mpc vs metres, phase_c ↔ phi_ref, and f_ref.

**R3. Time-domain cross-check with LDC tools (independent of the stationary-phase approximation).**
- Generate h₊, h× in the time domain by inverse FFT of ripple's IMRPhenomD for a 1e6 M☉ binary over ~1 week.
- Project with `lisagwresponse.ReadStrain` onto `lisaorbits` orbits, and form first-generation X, Y, Z → A, E, T with `pytdi.michelson.X1/Y1/Z1`.
- FFT the result and compare with our A and E. Target: mismatch < 1e-3.
- This is the only test that does not share the t(f) approximation with BBHx. It validates the SPA-based response near merger, where t(f) is least accurate.

**R4. Long-wavelength limit vs Cutler 1998 / Jim (implemented).**
- The existing `test_long_wavelength_limit_is_jims_antenna_pattern` already does this.
- Extend it to the low-frequency LISA antenna pattern with the √3/2 factor of a 60° interferometer, and with the annual modulation, checked against the analytic Cutler (1998) F₊ and F× over one year.

**R5. Exact sky-averaged response vs BBHx.**
- Average BBHx's |T_A|² over 10⁴ random sky positions and polarisations on the same orbit file as R1, and compare with our average.
- Target: within 1% (Monte Carlo noise) for 1e-4 to 5e-2 Hz.
- The existing checks cover only part of this. Robson's fit (factor 2, within 1% below 5 mHz, 7% off at 10 mHz) and lisatools' `stochastic_transform` are both low-frequency approximations: ours agrees with lisatools' shape to within 1% below 4 mHz and departs above, as expected. A sky-averaged BBHx response is the exact reference over the full band.

## Orbits and time

**O1. Orbits vs `lisaorbits.EqualArmlengthOrbits`.**
- Compare spacecraft positions over one year with the same initial phase and orientation.
- Expected agreement: O(e²·AU) ≈ 3,500 km, because ours are first order in eccentricity. Any O(e·AU) ≈ 7×10⁵ km difference is a convention mismatch: orientation κ/λ, spacecraft labelling, or the clockwise vs counter-clockwise sense of the triangle.
- Also compare arm lengths, which should be 2.5e9 m with no variation in our model, and the triangle's normal vector, which should be inclined 60° to the ecliptic.

**O2. Roemer delay vs astropy (implemented).**
- `test_roemer_delay_matches_astropy` compares `ssb_delay` with astropy's `light_travel_time(..., kind="barycentric")` at the geocentre for t_c = 0, 1 h and 1 day.
- Agreement: < 0.1 ms. This replaces the earlier test, which used `ssb_delay` to check itself.

**O3. Frame rotation vs astropy (implemented in part).**
- `test_sky_frame_matches_jims_equinox_of_date` checks that (ra, dec) taken in the mean equinox of date (as Jim's GMST-based ground detectors assume) map to astropy's J2000 ecliptic direction. Agreement: < 1.2e-4 rad, the size of annual aberration, which astropy includes and the analysis ignores.
- Still to do: the polarisation-angle mapping. Our rotated polarisation tensors at (ra, dec, ψ) must equal BBHx's ecliptic tensors at (λ, β, ψ_ecl), with ψ_ecl computed from the position angle of the ecliptic north pole.
- Still to do: an end-to-end check that Jim's H1 antenna pattern at (ra, dec, gmst) equals LAL's `ComputeDetAMResponse`, which pins the "equinox of date" reading of Jim's ra.

## Waveform

**W1. ripple vs LAL.**
- Compare `RippleIMRPhenomD` with `lalsimulation.SimInspiralChooseFDWaveform(..., IMRPhenomD)` at LISA masses (1e5–1e7 M☉, detector frame), down to f = 1e-4 Hz.
- Target: mismatch < 1e-6.
- ripple is tested against LAL for stellar masses. This check covers LISA masses, low frequencies, and `f_ref` far from merger.

**W2. t(f) vs LAL / BBHx.**
- Compare `time_to_merger` with −(1/2π) dΦ/df of LAL's phase (by finite difference), and with the `tf` that BBHx returns for PhenomHM (2,2).
- Target: < 1 s at 1e-4 Hz, i.e. weeks before merger. The time offset must equal the one in R2.

## Likelihood and Jim integration

**L1. Optimal SNR vs lisatools.**
- Compare Jim's ⟨h|h⟩ for an injection with `lisatools.diagnostic.snr` computed on the same A and E arrays and PSD. Target: < 1e-6.

**L2. Detector API conformance.**
- Parametrise Jim's own detector/likelihood tests over a `SpaceBased` channel to catch API drift between Jim releases.
- Covered tests: injection, `simulate_and_set_noise`, heterodyned vs full likelihood, and the Fisher information as the Hessian of the log-likelihood at the truth.

**L3. Fisher vs BBHx/lisatools Fisher.**
- Compare ∂² log L via `jax.hessian` of our likelihood with a finite-difference Fisher matrix from BBHx waveforms.
- Target: 1-σ errors on M_c, η, d_L, λ, β agree to 5%. This validates the derivatives that the samplers rely on.

## Order of work

1. **R1 and O1 first.** Together they pin every convention, so a disagreement elsewhere can be traced.
2. **Then R2 and W2.** They check the end-to-end parameter mapping.
3. **Then R3.** It is the slowest test but the only fully independent one.
4. **The rest are cheap**, at about 20 lines each.
