"""Per-frame trajectories and the cleaning steps every tool's output goes through.

Canonical trajectory CSV columns (pixels, original video frame)::

    frame, time_s, x_px, y_px[, area_px, motion_px]

Missing detections are empty / NaN.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED = ["time_s", "x_px", "y_px"]


@dataclass
class CleanResult:
    t: np.ndarray            # s, 0 at window start
    x: np.ndarray            # px, NaN where unknown
    y: np.ndarray
    raw_valid: np.ndarray    # detection present in the tool's output
    n_jumps: int             # detections rejected as impossible jumps
    extra: dict = field(default_factory=dict)


def load_trajectory(path: str | Path, fps: float | None = None,
                    columns: dict[str, str] | None = None) -> pd.DataFrame:
    """Read a trajectory CSV.  ``columns`` maps canonical names to the file's names,
    e.g. ``{"x_px": "X", "y_px": "Y", "frame": "Frame"}`` for ezTrack-style output."""
    df = pd.read_csv(path)
    if columns:
        df = df.rename(columns={v: k for k, v in columns.items()})
    if "time_s" not in df.columns:
        if "frame" not in df.columns or not fps:
            raise ValueError(f"{path}: needs a time_s column, or a frame column plus fps")
        df["time_s"] = df["frame"] / fps
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    return df


def clean(df: pd.DataFrame, px_per_m: float, start_s: float = 0.0, duration_s: float | None = None,
          max_speed_m_s: float = 2.0, max_gap_s: float = 0.5, smooth_s: float = 0.2) -> CleanResult:
    """Cut to the analysis window, drop impossible jumps, bridge short gaps, smooth.

    Order matters and is the same for every tool:
      1. keep frames with start_s <= t < start_s + duration_s, re-zero time
      2. reject a detection if reaching it from the last accepted one needs a speed
         above ``max_speed_m_s`` (tracker jumped to a reflection, hand, ...)
      3. linearly interpolate gaps no longer than ``max_gap_s``
      4. centred rolling median over ``smooth_s`` (removes centroid jitter,
         which otherwise inflates distance)
    """
    t_all = df["time_s"].to_numpy(float)
    keep = t_all >= start_s - 1e-9
    if duration_s is not None:
        keep &= t_all < start_s + duration_s - 1e-9
    sub = df.loc[keep]
    t = sub["time_s"].to_numpy(float) - start_s
    x = sub["x_px"].to_numpy(float).copy()
    y = sub["y_px"].to_numpy(float).copy()
    raw_valid = np.isfinite(x) & np.isfinite(y)

    # 2. jump rejection
    max_px_s = max_speed_m_s * px_per_m
    n_jumps = 0
    last = None
    for i in np.flatnonzero(raw_valid):
        if last is not None:
            dt = max(t[i] - t[last], 1e-9)
            if np.hypot(x[i] - x[last], y[i] - y[last]) / dt > max_px_s:
                x[i] = y[i] = np.nan
                n_jumps += 1
                continue
        last = i

    # 3. bridge short gaps
    valid = np.isfinite(x)
    if valid.sum() >= 2 and max_gap_s > 0:
        idx = np.flatnonzero(valid)
        gaps = np.diff(idx)
        for a, b, g in zip(idx[:-1], idx[1:], gaps):
            if g > 1 and t[b] - t[a] <= max_gap_s:
                f = (t[a + 1:b] - t[a]) / (t[b] - t[a])
                x[a + 1:b] = x[a] + f * (x[b] - x[a])
                y[a + 1:b] = y[a] + f * (y[b] - y[a])

    # 4. smoothing
    dt_med = float(np.median(np.diff(t))) if len(t) > 1 else 0.0
    win = int(round(smooth_s / dt_med)) if dt_med > 0 else 1
    if win >= 2:
        win |= 1  # odd, so the median is centred
        known = np.isfinite(x)
        x = pd.Series(x).rolling(win, center=True, min_periods=1).median().to_numpy(copy=True)
        y = pd.Series(y).rolling(win, center=True, min_periods=1).median().to_numpy(copy=True)
        x[~known] = np.nan
        y[~known] = np.nan

    extra = {c: sub[c].to_numpy(float) for c in ("area_px", "motion_px") if c in sub.columns}
    return CleanResult(t=t, x=x, y=y, raw_valid=raw_valid, n_jumps=n_jumps, extra=extra)
