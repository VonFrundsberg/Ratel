"""Shared metric definitions.

Every tool that yields a trajectory gets its metrics from ``compute_metrics``,
so the formulas are identical.  Tools that only export summaries (TailTrail)
are mapped onto the same names by their adapter, with corrections where their
definitions differ (see ``adapters/tailtrail.py``).

Long-format output, one row per value::

    video_id, tool, metric, zone, value, kind

``zone`` is "" for whole-arena metrics.  ``kind`` is "behaviour" (compared
across tools) or "quality" (tracking diagnostics, tool-specific).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config
from .trajectory import CleanResult, clean
from .zones import assign_zones

# metric -> (unit, kind, definition)
METRICS: dict[str, tuple[str, str, str]] = {
    "total_distance_m": ("m", "behaviour",
                         "Path length of the cleaned trajectory; steps across untracked gaps are skipped."),
    "mean_speed_m_s": ("m/s", "behaviour", "total_distance_m / analysed duration."),
    "immobile_time_s": ("s", "behaviour",
                        "Time in bouts of >= immobile_min_s with centroid speed < immobile_speed_m_s. "
                        "Centroid-based, so grooming/rearing in place counts as immobile."),
    "time_s": ("s", "behaviour", "Time assigned to the zone (exclusive, priority order, hysteresis)."),
    "time_pct": ("%", "behaviour", "time_s as a percentage of the analysed duration."),
    "distance_m": ("m", "behaviour", "Path length of steps that end in the zone."),
    "entries": ("count", "behaviour",
                "Transitions into the zone from anywhere else. The zone the animal starts in is NOT counted."),
    "latency_s": ("s", "behaviour", "Time of the first frame in the zone (0 if it starts there; NaN if never)."),
    "duration_s": ("s", "quality", "Analysed duration."),
    "tracked_fraction": ("", "quality", "Fraction of frames with a detection in the tool's raw output."),
    "n_jumps": ("count", "quality", "Detections rejected as faster than max_speed_m_s."),
    "unassigned_time_s": ("s", "quality", "Time outside every zone."),
    "losses": ("count", "quality", "TailTrail's own count of lost-object events."),
    # forced swim test (behaviour/fst.py, adapters/boris.py)
    "fst_immobile_s": ("s", "behaviour",
                       "Time immobile. Manual: coded immobility. Automated: bouts >= fst.min_immobile_s "
                       "with motion_px / median area < fst.motion_threshold."),
    "fst_swim_s": ("s", "behaviour", "Time coded as swimming (manual only)."),
    "fst_climb_s": ("s", "behaviour", "Time coded as climbing (manual only)."),
    "fst_immobility_latency_s": ("s", "behaviour",
                                 "Time to the start of the first immobility bout (0 if immobile at the "
                                 "window start; NaN if never)."),
    "fst_immobile_bins": ("count", "behaviour",
                          "Full fst.bin_s bins in which immobility predominates (Detke et al. 1995 "
                          "time sampling; automated: immobile vs. mobile, manual: vs. swim and climb)."),
    "fst_swim_bins": ("count", "behaviour", "Full fst.bin_s bins in which swimming predominates (manual only)."),
    "fst_climb_bins": ("count", "behaviour", "Full fst.bin_s bins in which climbing predominates (manual only)."),
    "fst_uncoded_s": ("s", "quality", "Manual FST scoring: analysed time with no FST state coded."),
}

LONG_COLUMNS = ["video_id", "tool", "metric", "zone", "value", "kind"]


def _frame_durations(t: np.ndarray, duration_s: float | None) -> np.ndarray:
    if len(t) == 0:
        return np.zeros(0)
    if len(t) == 1:
        return np.array([duration_s or 0.0])
    dt = np.diff(t)
    last = float(np.median(dt))
    if duration_s is not None:
        last = float(np.clip(duration_s - t[-1], 0.0, last))
    return np.append(dt, last)


def _immobile_time(speed: np.ndarray, durations: np.ndarray, thr: float, min_s: float) -> float:
    """speed[i] is the speed over frame i -> i+1 (NaN if unknown)."""
    still = np.nan_to_num(speed, nan=np.inf) < thr
    total = 0.0
    run = 0.0
    for s, d in zip(still, durations):
        if s:
            run += d
        else:
            if run >= min_s:
                total += run
            run = 0.0
    if run >= min_s:
        total += run
    return total


def compute_metrics(cr: CleanResult, cfg: Config, duration_s: float | None = None) -> list[dict]:
    """Return metric rows (without video_id/tool) for one cleaned trajectory."""
    m_per_px = 1.0 / cfg.px_per_m
    durations = _frame_durations(cr.t, duration_s)
    total_time = float(durations.sum())

    step = np.hypot(np.diff(cr.x), np.diff(cr.y)) * m_per_px  # NaN across gaps
    step_ok = np.isfinite(step)
    total_distance = float(step[step_ok].sum())
    speed = np.append(step / np.maximum(np.diff(cr.t), 1e-9), np.nan)

    rows = [
        dict(metric="duration_s", zone="", value=total_time),
        dict(metric="total_distance_m", zone="", value=total_distance),
        dict(metric="mean_speed_m_s", zone="", value=total_distance / total_time if total_time else np.nan),
        dict(metric="immobile_time_s", zone="",
             value=_immobile_time(speed, durations, cfg.immobile_speed_m_s, cfg.immobile_min_s)),
        dict(metric="tracked_fraction", zone="", value=float(cr.raw_valid.mean()) if len(cr.raw_valid) else np.nan),
        dict(metric="n_jumps", zone="", value=float(cr.n_jumps)),
    ]

    zone_idx = assign_zones(cfg.zones, cr.x, cr.y, cfg.hysteresis_mode)
    rows.append(dict(metric="unassigned_time_s", zone="", value=float(durations[zone_idx == -1].sum())))
    entered = np.zeros(len(zone_idx), bool)
    entered[1:] = zone_idx[1:] != zone_idx[:-1]
    end_zone = zone_idx[1:]
    for k, z in enumerate(cfg.zones):
        in_z = zone_idx == k
        t_in = float(durations[in_z].sum())
        first = np.flatnonzero(in_z)
        rows += [
            dict(metric="time_s", zone=z.name, value=t_in),
            dict(metric="time_pct", zone=z.name, value=100.0 * t_in / total_time if total_time else np.nan),
            dict(metric="distance_m", zone=z.name, value=float(step[(end_zone == k) & step_ok].sum())),
            dict(metric="entries", zone=z.name, value=float((entered & in_z).sum())),
            dict(metric="latency_s", zone=z.name, value=float(cr.t[first[0]]) if first.size else np.nan),
        ]
    for r in rows:
        r["kind"] = METRICS[r["metric"]][1]
    return rows


def to_long(rows: list[dict], video_id: str, tool: str) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df.insert(0, "tool", tool)
    df.insert(0, "video_id", video_id)
    return df[LONG_COLUMNS]


def metrics_for_trajectory(df: pd.DataFrame, cfg: Config, video_id: str, tool: str) -> pd.DataFrame:
    s = cfg.session(video_id)
    cr = clean(df, cfg.px_per_m, s.start_s, s.duration_s,
               cfg.max_speed_m_s, cfg.max_gap_s, cfg.smooth_s)
    return to_long(compute_metrics(cr, cfg, s.duration_s), video_id, tool)
