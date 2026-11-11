# galactic-foreground-generator

A Python and Jupyter port of **Lucy McNeill's** Mathematica notebook `paper-calculations.nb`. The notebook covers:

- the intrinsic mass distribution of white dwarfs in close double white dwarf (DWD) binaries
- how those binaries pair up and which ones merge
- the Galactic DWD population (the gravitational-wave foreground), and how much of it falls inside the 100 pc sample

> **Credit:** the science, methods and original notebook are by **Lucy McNeill**. This repository only translates them to Python.
> If you use it, please cite Lucy McNeill's work (see `CITATION.cff`) and the data and literature listed below.

## Contents

| Path | What |
|---|---|
| `notebooks/white_dwarf_rates.ipynb` | The full notebook, section by section as in the original, with outputs already run |
| `src/galactic_foreground/selection.py` | He/CO WD radius fits and the luminosity selection function (observed to intrinsic masses) |
| `src/galactic_foreground/population.py` | Pair sampling, He/CO composition, mass-sum fractions, KDE contours, Nelemans+2001 boundary |
| `src/galactic_foreground/stability.py` | Boundary between stable and unstable mass transfer (Eggleton mass–radius, Roche lobe, GW losses) |
| `src/galactic_foreground/gw.py` | Chirp mass, df/dt, residence times, N(>f) for the Milky Way and the 100 pc sample |
| `src/galactic_foreground/generator.py` | **Extension:** draws a synthetic catalogue of Galactic DWDs (m1, m2, f_GW, f_orb, position, distance) from the same model |
| `tests/` | Checks against every analytic number printed in the original notebook |
| `scripts/build_notebook.py` | Regenerates the notebook from source |

## Quick start

```bash
pip install -r requirements.txt
pytest -q
jupyter lab notebooks/white_dwarf_rates.ipynb
```

The notebook downloads the public DWD catalogue on first run (see `data/README.md`). To reproduce the original numbers exactly, put the original `munday-cut-4.csv` in `data/`.

## Fidelity to the original

- **Matches exactly** (to the printed precision, enforced by tests): chirp masses, df/dt, the 1 Gyr steady-state frequency (1.513×10⁻⁴ Hz), the N(>f) coefficients (9600.75, 26438.4, 18144.3, 39312.6), N_HeHe,MW = 701,819, and the 100 pc counts (8.96, 24.67, 16.93, about 50 in total).
- **Matches statistically:** the parts that depend on the data and random draws (composition and mass-sum fractions, stable fraction). The differences come from:
  - the public catalogue having grown since the February 2026 cut (141 binaries now, 134 then)
  - Mathematica's automatic histogram binning (a fixed 0.05 M☉ here)
  - the original's unseeded RNG (seeded here)
- **Stability boundary:** solved again with a bracketed root finder. It agrees with the notebook's m₂ = −0.1787 + 0.4907 m₁^¼ fit to within 0.01 M☉ for 0.2–1.4 M☉. Downstream counts use the notebook's coefficients.
- **Figure 4 secondary axes** (P_orb and N₁₀₀pc) are exact conversions. The original's tick values were placeholders.

## References

- L. McNeill: original analysis and Mathematica notebook; McNeill & Hirai (2025)
- J. Munday et al. 2024, MNRAS 532, 2534: close DWD catalogue, https://github.com/JamesMunday98/CloseDWDbinaries (GPL-3.0, not redistributed here)
- Cheng et al. 2020: merger-remnant mass distribution
- Nelemans et al. 2001: stable vs unstable DWD mass transfer
- Eggleton 1983; Verbunt & Rappaport 1988; Soberman et al. 1997: Roche lobe and mass–radius relations
- Lamberts et al. 2019: population synthesis comparison
