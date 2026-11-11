"""Input samples: Cheng et al. (2020) remnant masses and the Munday DWD catalogue."""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

MISSING = -99.0  # sentinel for "not measured"; the notebook filters with  x > -50

MUNDAY_REPO = "https://raw.githubusercontent.com/JamesMunday98/CloseDWDbinaries/main/"
MUNDAY_DEFAULT_FILE = "CloseDoubleWhiteDwarfs_20261109.csv"


def cheng2020_sample() -> np.ndarray:
    """Merger-remnant masses reconstructed from the caption of Fig. 3 of Cheng et al. (2020).

    408 (0.07), 431 (0.15), 323 (0.27), 176 (0.35), 57 (0.18) -> 29, 65, 87, 62, 10.
    """
    centres = [0.85, 0.95, 1.07, 1.19, 1.28]
    counts = [29, 65, 87, 62, 10]
    return np.repeat(centres, counts)


def load_munday_cut(path: str | Path) -> dict[str, np.ndarray]:
    """Load the pre-cut sample used in the notebook (``munday-cut-4.csv``).

    Columns 2-5 (1-based) are m1, m2, T1, T2; unmeasured entries are numbers < -50.
    """
    df = pd.read_csv(path)
    cols = df.columns[1:5]
    out = {}
    for key, col in zip(("m1", "m2", "T1", "T2"), cols):
        out[key] = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
    keep = np.all([np.isfinite(v) for v in out.values()], axis=0)
    return {k: v[keep] for k, v in out.items()}


def fetch_munday_catalogue(dest_dir: str | Path = "data", filename: str = MUNDAY_DEFAULT_FILE) -> Path:
    """Download the public close-DWD catalogue (Munday et al. 2024; GPL-3.0, not redistributed here)."""
    dest = Path(dest_dir) / filename
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(MUNDAY_REPO + filename, dest)
    return dest


def build_munday_cut(catalogue_csv: str | Path) -> dict[str, np.ndarray]:
    """Rebuild the notebook's cut from the full catalogue.

    Keeps binaries with numeric (not limit) M1 and M2 and a measured T1; T2 may be
    missing and is then set to ``MISSING``. With the Feb-2026 catalogue this gave
    134 binaries / 72 secondary temperatures; later catalogue versions contain more.
    """
    df = pd.read_csv(catalogue_csv)
    num = {c: pd.to_numeric(df[c], errors="coerce") for c in ("M1", "M2", "T1", "T2")}
    keep = num["M1"].notna() & num["M2"].notna() & num["T1"].notna()
    return {
        "m1": num["M1"][keep].to_numpy(float),
        "m2": num["M2"][keep].to_numpy(float),
        "T1": num["T1"][keep].to_numpy(float),
        "T2": num["T2"][keep].fillna(MISSING).to_numpy(float),
    }


def load_sample(data_dir: str | Path = "data") -> tuple[dict[str, np.ndarray], str]:
    """Use ``munday-cut-4.csv`` if present, else rebuild the cut from the public catalogue."""
    data_dir = Path(data_dir)
    cut = data_dir / "munday-cut-4.csv"
    if cut.exists():
        return load_munday_cut(cut), str(cut)
    local = sorted(data_dir.glob("CloseDoubleWhiteDwarfs_*.csv"))
    path = local[-1] if local else fetch_munday_catalogue(data_dir)
    return build_munday_cut(path), str(path)


def masses_with_temperatures(sample: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Components with both a mass and a temperature (the notebook's ``bluemass``/``blueTemps``)."""
    ok1 = sample["T1"] > -50
    m1, T1 = sample["m1"][ok1], sample["T1"][ok1]
    ok2 = (sample["T2"] > -50) & (sample["m2"] > -50)
    m2, T2 = sample["m2"][ok2], sample["T2"][ok2]
    return np.concatenate([m1, m2]), np.concatenate([T1, T2])
