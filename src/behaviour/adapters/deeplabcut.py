"""DeepLabCut adapter: DLC CSV output -> canonical per-frame trajectory.

DLC CSVs have a three-row header (scorer / bodyparts / coords) and one row per
frame with x, y, likelihood for every body part.  Points below
``min_likelihood`` become NaN and are then handled by the shared cleaning
(jump rejection, gap bridging) exactly like any other tool's lost frames.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

TOOL = "dlc"


def video_id_from_filename(stem: str) -> str:
    """'FST_1_cropDeepCut_resnet50_...' -> 'FST_1_crop'; 'rat01DLC_resnet50_...' -> 'rat01'."""
    return re.split(r"_?(?:DeepCut|DLC)_", stem, maxsplit=1)[0]


def read_dlc(path: str | Path) -> pd.DataFrame:
    """DLC CSV with columns flattened to (bodypart, coord)."""
    df = pd.read_csv(path, header=[0, 1, 2], index_col=0)
    df.columns = df.columns.droplevel(0)
    return df


def trajectory(path: str | Path, fps: float, bodypart: str = "bodycentre",
               min_likelihood: float = 0.9) -> pd.DataFrame:
    df = read_dlc(path)
    parts = sorted(set(df.columns.get_level_values(0)))
    if bodypart not in parts:
        raise ValueError(f"{path}: no body part {bodypart!r}; available: {parts}")
    x = df[(bodypart, "x")].to_numpy(float).copy()
    y = df[(bodypart, "y")].to_numpy(float).copy()
    bad = df[(bodypart, "likelihood")].to_numpy(float) < min_likelihood
    x[bad] = np.nan
    y[bad] = np.nan
    frame = np.arange(len(df))
    return pd.DataFrame({"frame": frame, "time_s": frame / fps, "x_px": x, "y_px": y})


def landmarks(path: str | Path, names: list[str], min_likelihood: float = 0.95) -> np.ndarray:
    """Median position of static landmarks (e.g. arena corners) -> (len(names), 2)."""
    df = read_dlc(path)
    out = []
    for n in names:
        ok = df[(n, "likelihood")] > min_likelihood
        if not ok.any():
            raise ValueError(f"{path}: landmark {n!r} never detected with likelihood > {min_likelihood}")
        out.append([df.loc[ok, (n, "x")].median(), df.loc[ok, (n, "y")].median()])
    return np.array(out)
