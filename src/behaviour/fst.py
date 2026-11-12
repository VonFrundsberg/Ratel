"""Forced swim test (FST): shared metric definitions and automated scoring.

Manual scoring (``adapters/boris.py``) and the automated scorer below both
build their FST metrics from time intervals with the helpers here, so totals,
latency and the 5-s bin counts are computed the same way.

Automated scoring uses the OpenCV tracker's per-frame ``motion_px`` (pixels
that changed since the previous frame inside the dilated animal blob):

  1. keep frames in the analysis window (``[window]`` / sessions CSV), re-zero time
  2. ratio = motion_px / median area_px over the window, so the threshold does
     not depend on camera distance or animal size
  3. centred rolling mean of the ratio over ``fst.smooth_s``
  4. a frame is still if ratio < ``fst.motion_threshold``; frames without a
     detection or motion value are never still
  5. immobile = still frames in runs of at least ``fst.min_immobile_s``

Motion is measured between consecutive frames, so the threshold depends on the
frame rate; calibrate it against manual scores at the lab's frame rate.  Only
immobile vs. mobile can be told apart: swimming vs. climbing needs manual scoring.

Bin counts follow the time-sampling method of Detke et al. (1995): the test is
cut into full ``fst.bin_s`` bins and each bin is credited to the behaviour that
takes up most of it.  Uncoded (manual) or mobile (automated) time competes as
"other", so a bin with 1 s of immobility and 4 s of anything else is not an
immobile bin.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config
from .metrics import METRICS, _frame_durations, to_long

TOOL = "opencv"
STATES = ("immobile", "swim", "climb")

Interval = tuple[float, float]


# --------------------------------------------------------------------------- intervals


def union(iv: list[Interval], tol: float = 1e-6) -> list[Interval]:
    """Sorted, merged intervals; ones that touch (within ``tol``) are joined."""
    out: list[list[float]] = []
    for a, b in sorted(iv):
        if b <= a:
            continue
        if out and a <= out[-1][1] + tol:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def clip(iv: list[Interval], start: float, end: float) -> list[Interval]:
    """Cut to [start, end) and re-zero so that ``start`` becomes 0."""
    out = []
    for a, b in iv:
        a, b = max(a, start), min(b, end)
        if b > a:
            out.append((a - start, b - start))
    return out


def total(iv: list[Interval]) -> float:
    return float(sum(b - a for a, b in union(iv)))


def overlap(iv: list[Interval], start: float, end: float) -> float:
    return float(sum(max(0.0, min(b, end) - max(a, start)) for a, b in iv))


def predominant_bins(states: dict[str, list[Interval]], duration_s: float, bin_s: float) -> dict[str, int]:
    """Number of full ``bin_s`` bins in which each state takes up the most time.

    Time not in any of ``states`` competes as "other" and is not reported.
    Ties go to the state listed first.
    """
    counts = dict.fromkeys(states, 0)
    if bin_s <= 0:
        return counts
    names = list(states)
    for k in range(int(np.floor(duration_s / bin_s + 1e-9))):
        a, b = k * bin_s, (k + 1) * bin_s
        t = [overlap(states[s], a, b) for s in names]
        other = max(bin_s - sum(t), 0.0)
        best = int(np.argmax(t))
        if t[best] > 0 and t[best] >= other:
            counts[names[best]] += 1
    return counts


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, stop) index pairs of the True runs in ``mask``."""
    m = np.r_[False, np.asarray(mask, bool), False]
    edges = np.flatnonzero(np.diff(m.astype(int)))
    return list(zip(edges[::2], edges[1::2]))


# --------------------------------------------------------------------------- automated scoring


def score_immobility(t: np.ndarray, motion_px: np.ndarray, area_px: np.ndarray, cfg: Config,
                     duration_s: float | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (frame durations, motion ratio after smoothing, immobile mask) for windowed frames."""
    p = cfg.fst
    durations = _frame_durations(t, duration_s)
    area = np.asarray(area_px, float)
    area = area[np.isfinite(area) & (area > 0)]
    ratio = np.asarray(motion_px, float) / (np.median(area) if area.size else np.nan)

    dt_med = float(np.median(np.diff(t))) if len(t) > 1 else 0.0
    win = int(round(p.smooth_s / dt_med)) if dt_med > 0 else 1
    if win >= 2:
        win |= 1  # odd, so the mean is centred
        known = np.isfinite(ratio)
        ratio = pd.Series(ratio).rolling(win, center=True, min_periods=1).mean().to_numpy(copy=True)
        ratio[~known] = np.nan

    still = np.nan_to_num(ratio, nan=np.inf) < p.motion_threshold
    immobile = np.zeros(len(t), bool)
    for a, b in runs(still):
        if durations[a:b].sum() >= p.min_immobile_s - 1e-9:
            immobile[a:b] = True
    return durations, ratio, immobile


def fst_rows(states: dict[str, list[Interval]], analysed_s: float, cfg: Config) -> list[dict]:
    """FST metric rows from per-state intervals (already clipped to the window, 0 = window start)."""
    rows = []
    for s, iv in states.items():
        rows.append(dict(metric=f"fst_{s}_s", zone="", value=total(iv)))
        if s == "immobile":
            u = union(iv)
            rows.append(dict(metric="fst_immobility_latency_s", zone="", value=u[0][0] if u else np.nan))
    if cfg.fst.bin_s > 0:
        for s, n in predominant_bins({s: union(iv) for s, iv in states.items()}, analysed_s, cfg.fst.bin_s).items():
            rows.append(dict(metric=f"fst_{s}_bins", zone="", value=float(n)))
    return rows


def fst_metrics(df: pd.DataFrame, cfg: Config, video_id: str, tool: str = TOOL) -> pd.DataFrame:
    """Automated FST metrics for one per-frame track (needs time_s, motion_px, area_px)."""
    missing = [c for c in ("time_s", "motion_px", "area_px") if c not in df.columns]
    if missing:
        raise ValueError(f"{video_id}: missing columns {missing}; FST scoring needs the output of `behav track`")
    s = cfg.session(video_id)
    t_all = df["time_s"].to_numpy(float)
    keep = t_all >= s.start_s - 1e-9
    if s.duration_s is not None:
        keep &= t_all < s.start_s + s.duration_s - 1e-9
    sub = df.loc[keep]
    t = t_all[keep] - s.start_s
    motion = sub["motion_px"].to_numpy(float)

    durations, _, immobile = score_immobility(t, motion, sub["area_px"].to_numpy(float), cfg, s.duration_s)
    analysed = float(durations.sum())
    imm_iv = [(float(t[a]), float(t[b - 1] + durations[b - 1])) for a, b in runs(immobile)]

    rows = [
        dict(metric="duration_s", zone="", value=analysed),
        dict(metric="tracked_fraction", zone="", value=float(np.isfinite(motion).mean()) if len(motion) else np.nan),
        *fst_rows({"immobile": imm_iv}, analysed, cfg),
    ]
    for r in rows:
        r["kind"] = METRICS[r["metric"]][1]
    return to_long(rows, video_id, tool)
