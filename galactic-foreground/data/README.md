# Data

The notebook needs the close double white dwarf sample of James Munday
(https://github.com/JamesMunday98/CloseDWDbinaries, Munday et al. 2024, MNRAS 532, 2534).
It is GPL-3.0 licensed and therefore **not committed here**.

Order of preference used by `galactic_foreground.data.load_sample()`:

1. `data/munday-cut-4.csv` — the exact cut used in Lucy McNeill's original notebook
   (columns 2–5 = m1, m2, T1, T2; unmeasured values < -50). Drop it here to reproduce the
   original numbers exactly.
2. Any local `data/CloseDoubleWhiteDwarfs_YYYYMMDD.csv`.
3. Otherwise the public catalogue is downloaded and the cut is rebuilt
   (numeric M1 and M2, measured T1). The catalogue grows over time, so counts differ
   slightly from the original 134 binaries / 72 secondary temperatures.
