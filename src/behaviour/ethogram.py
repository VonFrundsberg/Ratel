"""Shared metrics for named behaviours, independent of the scoring software.

Canonical events use video_id, behaviour, start, stop (absolute media seconds),
and event_type (state or point). Optional observer keeps human raters separate.
States are merged within behaviour, never across behaviours or observers.
The scoring window is half-open: a point at its end belongs to the next window.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .config import Config
from .fst import clip, union
from .metrics import LONG_COLUMNS

EVENT_COLUMNS = ["video_id", "behaviour", "start", "stop", "event_type"]
ALIASES = {
    "groom": "grooming", "body_groom": "body_grooming", "face_groom": "face_grooming",
    "rear": "rearing", "stand_up": "rearing", "stand_ups": "rearing",
    "supported": "supported_rearing", "rearing_supported": "supported_rearing",
    "unsupported": "unsupported_rearing", "rearing_unsupported": "unsupported_rearing",
    "defecating": "defecation", "defecate": "defecation",
    "urinating": "urination", "urinate": "urination",
}


def behaviour_name(name: str) -> str:
    name = re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")
    if not name:
        raise ValueError("Behaviour names cannot be empty")
    return ALIASES.get(name, name)


def normalize_events(events: pd.DataFrame, mapping: dict[str, str] | None = None) -> pd.DataFrame:
    missing = {"video_id", "behaviour", "start", "stop"} - set(events)
    if missing:
        raise ValueError(f"Behaviour events missing columns {sorted(missing)}")
    out = events.copy()
    mapping = {str(k).strip().lower(): v for k, v in (mapping or {}).items()}
    out["behaviour"] = [behaviour_name(mapping.get(str(b).strip().lower(), b)) for b in out.behaviour]
    out["video_id"] = out.video_id.astype(str)
    for c in ("start", "stop"):
        out[c] = pd.to_numeric(out[c], errors="raise")
    if "event_type" not in out:
        out["event_type"] = np.where(out.stop == out.start, "point", "state")
    out["event_type"] = out.event_type.astype(str).str.lower()
    if not out.event_type.isin(["state", "point"]).all():
        raise ValueError("event_type must be state or point")
    if not np.isfinite(out[["start", "stop"]].to_numpy(float)).all():
        raise ValueError("Event start and stop times must be finite")
    if (out.start < 0).any() or (out.stop < out.start).any():
        raise ValueError("Events require 0 <= start <= stop")
    if ((out.event_type == "state") & (out.stop <= out.start)).any():
        raise ValueError("State events require a positive duration")
    if ((out.event_type == "point") & (out.stop != out.start)).any():
        raise ValueError("Point events require start == stop")
    return out


def behaviour_rows(events: pd.DataFrame, start_s: float, end_s: float,
                   behaviours: dict[str, str] | None = None) -> list[dict]:
    """Eight summaries per state; count, frequency and latency per point behaviour.

    Explicit ``behaviours`` declares which absent behaviours were actually scored;
    undeclared behaviours are never silently turned into zero observations.
    Bouts intersecting the window are counted and their durations are clipped.
    """
    duration = max(end_s - start_s, 0.0)
    kinds = dict(behaviours or {})
    for name, g in events.groupby("behaviour", sort=True):
        types = set(g.event_type)
        if len(types) != 1:
            raise ValueError(f"{name}: cannot combine point and state events")
        kind = next(iter(types))
        if name in kinds and kinds[name] != kind:
            raise ValueError(f"{name}: event type differs from the declared ethogram")
        kinds[name] = kind
    out = []
    for name, kind in sorted(kinds.items()):
        g = events[events.behaviour == name]
        if kind == "point":
            t = g.loc[(g.start >= start_s) & (g.start < end_s), "start"].to_numpy(float)
            values = {"count": len(t), "latency_s": t.min() - start_s if len(t) else np.nan}
            count = len(t)
        elif kind == "state":
            iv = union(clip(list(zip(g.start, g.stop)), start_s, end_s))
            lengths = np.array([b - a for a, b in iv])
            count = len(iv)
            time = float(lengths.sum())
            values = {
                "time_s": time, "time_pct": 100 * time / duration if duration else np.nan,
                "bouts": count, "mean_bout_s": lengths.mean() if count else np.nan,
                "median_bout_s": np.median(lengths) if count else np.nan,
                "max_bout_s": lengths.max() if count else np.nan,
                "latency_s": iv[0][0] if count else np.nan,
            }
        else:
            raise ValueError(f"Unknown event type {kind!r}")
        values["frequency_per_min"] = 60 * count / duration if duration else np.nan
        out += [dict(metric=f"{name}_{key}", zone="", value=float(value), kind="behaviour")
                for key, value in values.items()]
    return out


def events_to_metrics(events: pd.DataFrame, cfg: Config, tool: str,
                      behaviours: dict[str, str] | None = None, video_id: str | None = None) -> pd.DataFrame:
    events = normalize_events(events)
    declared = {behaviour_name(k): v for k, v in (behaviours or {}).items()}
    keys = ["video_id"] + (["observer"] if "observer" in events else [])
    out = []
    groups = events.groupby(keys, dropna=False, sort=True)
    if events.empty:
        vid = video_id or events.attrs.get("video_id")
        if not vid:
            raise ValueError("An empty event table needs video_id to report scored zero counts")
        groups = [((vid,), events)]
    for key, g in groups:
        if not isinstance(key, tuple):
            key = (key,)
        vid = str(key[0])
        observer = str(key[1]) if len(key) > 1 and pd.notna(key[1]) else ""
        label = f"{tool}_{observer}" if observer else tool
        s = cfg.session(vid)
        end = s.start_s + s.duration_s if s.duration_s is not None else np.inf
        media_dur = pd.to_numeric(g.get("media_duration", pd.Series(dtype=float)), errors="coerce").max()
        if not np.isfinite(media_dur):
            media_dur = events.attrs.get("media_duration", np.nan)
        if np.isfinite(media_dur):
            end = min(end, media_dur)
        if not np.isfinite(end):
            raise ValueError(f"{vid}: an analysis duration or media_duration is required")
        rows = behaviour_rows(g, s.start_s, end, declared)
        rows.append(dict(metric="duration_s", zone="", value=max(end - s.start_s, 0), kind="quality"))
        out += [dict(video_id=vid, tool=label, **r) for r in rows]
    return pd.DataFrame(out, columns=LONG_COLUMNS)
