"""BORIS (Behavioral Observation Research Interactive Software) adapter: manual scoring.

Reads BORIS event exports (Observations > Export events), as CSV, TSV or XLSX:

* aggregated events: one row per state bout with ``Start (s)`` / ``Stop (s)``
* tabular events: one row per event with ``Time`` and ``Behavior type``
  (BORIS 8) or ``Status`` (BORIS 7) = START / STOP / POINT; BORIS 7 puts
  observation metadata above the table, which is skipped (and used for the
  observation id)

Point events are ignored by default; include_points preserves them and any
explicit observer field for the named-behaviour pipeline. Times are seconds from the media start,
then cut to the same analysis window as the trackers (``[window]`` / sessions
CSV).  The video id is the media file name without folder and extension (as
for the trackers), or the BORIS observation id.

Behaviours are mapped to

* config zones, for open field zone-crossing coding: a behaviour named like a
  zone maps to it, others via ``zone_map`` (e.g. ``{"Center zone": "center"}``).
  The zone states should be mutually exclusive in the ethogram.
* FST states ``immobile`` / ``swim`` / ``climb``: ``DEFAULT_FST_MAP`` plus
  ``fst_map``.  A state is reported only if the export contains it, so a file
  in which only immobility was coded yields no swim/climb zeros.

Zone metrics follow ``behaviour/metrics.py``: the zone the animal is in at the
window start (= the first zone coded) is not an entry and has latency 0.  An
uncoded gap counts towards ``unassigned_time_s``; returning to the same zone
after it is not an entry (as for untracked frames, the animal is assumed to be
where it was last seen).  Distance needs positions, so there is none.
FST metrics come from ``behaviour/fst.py``; manual bouts are used as coded,
without a minimum duration.
"""

from __future__ import annotations

import csv
import io
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
from ..fst import STATES, clip, fst_rows, total, union
from ..metrics import LONG_COLUMNS, METRICS

TOOL = "manual"
DEFAULT_FST_MAP = {
    "immobility": "immobile", "immobile": "immobile", "floating": "immobile",
    "swimming": "swim", "swim": "swim",
    "climbing": "climb", "climb": "climb",
}
EVENT_COLUMNS = ["video_id", "behaviour", "start", "stop", "media_duration"]
_TOL = 1e-6


# --------------------------------------------------------------------------- reading


def _read_rows(path: Path) -> list[list[str]]:
    if path.suffix.lower() in (".xlsx", ".xls", ".ods"):
        raw = pd.read_excel(path, header=None, dtype=object)
        return [["" if pd.isna(v) else str(v) for v in r] for r in raw.itertuples(index=False)]
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() == ".tsv":
        delim = "\t"
    else:
        try:
            delim = csv.Sniffer().sniff(text[:4096], delimiters=",\t;").delimiter
        except csv.Error:
            delim = ","
    return list(csv.reader(io.StringIO(text), delimiter=delim))


def read_table(path: str | Path) -> tuple[pd.DataFrame, dict[str, str]]:
    """Return (event table, metadata) from a BORIS export.

    The header is the first row with a "Behavior" cell; rows above it are
    ``key, value`` metadata (BORIS 7 tabular export)."""
    path = Path(path)
    rows = _read_rows(path)
    for i, r in enumerate(rows[:100]):
        if {c.strip().lower() for c in r} & {"behavior", "behaviour"}:
            header = [c.strip() for c in r]
            body = [(list(b) + [""] * len(header))[: len(header)] for b in rows[i + 1:]
                    if any(str(c).strip() for c in b)]
            meta = {str(m[0]).strip().lower(): str(m[1]).strip() for m in rows[:i] if len(m) >= 2 and str(m[0]).strip()}
            return pd.DataFrame(body, columns=header), meta
    raise ValueError(f"{path}: no header row with a 'Behavior' column; export events from BORIS "
                     "as aggregated or tabular events")


def _col(df: pd.DataFrame, *names: str) -> str | None:
    lower = {c.lower(): c for c in reversed(df.columns)}
    for n in names:
        if n in lower:
            return lower[n]
    return None


def _stem(name: str) -> str:
    base = re.split(r"[\\/]", str(name).strip())[-1]
    return Path(base).stem if Path(base).suffix else base


def read_events(path: str | Path, video_id: str = "media", include_points: bool = False) -> pd.DataFrame:
    """State bouts from one BORIS export: columns ``EVENT_COLUMNS``.

    ``video_id`` is "media" (media file name stem) or "observation" (BORIS
    observation id); if that is missing the other one, then the file name, is used."""
    path = Path(path)
    df, meta = read_table(path)
    beh = _col(df, "behavior", "behaviour")
    media = _col(df, "media file name", "media file path", "media file")
    obs = _col(df, "observation id")
    dur = _col(df, "media duration (s)", "media duration", "total length", "media total length")

    def ids(frame: pd.DataFrame) -> pd.Series:
        def get(c, key):
            if c is not None:
                return frame[c].astype(str).str.strip()
            return pd.Series(meta.get(key, ""), index=frame.index)

        m = get(media, "media file name").map(lambda s: _stem(s) if s else "")
        o = get(obs, "observation id")
        first, second = (m, o) if video_id == "media" else (o, m)
        out = first.where(first != "", second)
        return out.where(out != "", path.stem)

    df = df[df[beh].astype(str).str.strip() != ""].copy()
    df["_vid"] = ids(df)
    df["_beh"] = df[beh].astype(str).str.strip()
    observer = _col(df, "observer")
    keep_observer = include_points and (observer is not None or bool(meta.get("observer")))
    if keep_observer:
        df["_observer"] = df[observer].astype(str).str.strip() if observer else meta["observer"]
    if dur:
        df["_dur"] = pd.to_numeric(df[dur], errors="coerce")
    else:
        df["_dur"] = pd.to_numeric(pd.Series(meta.get("media duration (s)", meta.get("total length", "")),
                                             index=df.index), errors="coerce")
    subj = _col(df, "subject")
    if subj:
        for vid, g in df.groupby("_vid"):
            s = set(g[subj].astype(str).str.strip()) - {"", "No focal subject"}
            if len(s) > 1:
                warnings.warn(f"{path}: {vid} has several subjects {sorted(s)}; their events are pooled",
                              stacklevel=2)

    start, stop = _col(df, "start (s)", "start"), _col(df, "stop (s)", "stop")
    if start and stop:  # aggregated events
        btype = _col(df, "behavior type", "behaviour type")
        if btype and not include_points:
            df = df[df[btype].astype(str).str.strip().str.upper() != "POINT"]
        out = pd.DataFrame({
            "video_id": df["_vid"], "behaviour": df["_beh"],
            "start": pd.to_numeric(df[start], errors="coerce"),
            "stop": pd.to_numeric(df[stop], errors="coerce"),
            "media_duration": df["_dur"],
        })
        if include_points:
            if keep_observer:
                out["observer"] = df["_observer"]
            points = df[btype].astype(str).str.strip().str.upper().eq("POINT") if btype else out.stop.eq(out.start)
            out["event_type"] = np.where(points, "point", "state")
            out.loc[points, "stop"] = out.loc[points, "start"]
            out = out[(out.stop > out.start) | points]
            return out[[*EVENT_COLUMNS, "event_type", *(["observer"] if keep_observer else [])]].reset_index(drop=True)
        out = out[out["stop"] > out["start"]]
        return out[EVENT_COLUMNS].reset_index(drop=True)

    time = _col(df, "time")
    status = _col(df, "behavior type", "behaviour type", "status", "event type")
    if not (time and status):
        raise ValueError(f"{path}: neither aggregated (Start (s), Stop (s)) nor tabular (Time, "
                         f"Behavior type / Status) events; columns are {list(df.columns)}")
    df["_t"] = pd.to_numeric(df[time], errors="coerce")
    df["_s"] = df[status].astype(str).str.strip().str.upper()
    rows = []
    for key, g in df.groupby(["_vid", "_beh", *(["_observer"] if keep_observer else [])], sort=False):
        vid, b = key[:2]
        suffix = (key[2],) if keep_observer else ()
        if include_points:
            rows += [(vid, b, t, t, g["_dur"].max(), "point", *suffix) for t in g.loc[g["_s"] == "POINT", "_t"]]
        g = g[g["_s"].isin(["START", "STOP"])].sort_values("_t", kind="stable")
        media_dur = g["_dur"].max()
        opened = None
        for t, s in zip(g["_t"], g["_s"]):
            if s == "START" and opened is None:
                opened = t
            elif s == "STOP" and opened is not None:
                rows.append((vid, b, opened, t, media_dur, "state", *suffix) if include_points else (vid, b, opened, t, media_dur))
                opened = None
            else:
                warnings.warn(f"{path}: {vid} '{b}': unpaired {s} at {t} s ignored", stacklevel=2)
        if opened is not None:
            end = media_dur if np.isfinite(media_dur) else np.inf
            warnings.warn(f"{path}: {vid} '{b}': START at {opened} s never stopped; "
                          "closed at the end of the observation", stacklevel=2)
            rows.append((vid, b, opened, end, media_dur, "state", *suffix) if include_points else (vid, b, opened, end, media_dur))
    columns = [*EVENT_COLUMNS, "event_type", *(["observer"] if keep_observer else [])] if include_points else EVENT_COLUMNS
    return pd.DataFrame(rows, columns=columns)


# --------------------------------------------------------------------------- metrics


def _mapping(m: dict[str, str] | None) -> dict[str, str]:
    return {str(k).strip().lower(): str(v).strip().lower() for k, v in (m or {}).items()}


def events_to_long(events: pd.DataFrame, cfg: Config, zone_map: dict[str, str] | None = None,
                   fst_map: dict[str, str] | None = None) -> pd.DataFrame:
    """Convert state bouts (from ``read_events``) into the long metric table."""
    zone_names = cfg.zone_names()
    zmap = {z: z for z in zone_names} | _mapping(zone_map)
    fmap = DEFAULT_FST_MAP | _mapping(fst_map)
    bad = {v for v in zmap.values() if v not in zone_names}
    if bad:
        raise ValueError(f"zone map targets {sorted(bad)} are not zones in the config {zone_names}")
    bad = {v for v in fmap.values() if v not in STATES}
    if bad:
        raise ValueError(f"FST map targets {sorted(bad)} are not FST states {list(STATES)}")

    key = events["behaviour"].str.lower()
    events = events.assign(zone=key.map(zmap), state=key.map(fmap))
    ignored = sorted(set(events.loc[events["zone"].isna() & events["state"].isna(), "behaviour"]))
    if ignored:
        warnings.warn(f"BORIS behaviours {ignored} map to no zone or FST state and are ignored; "
                      "use the zone / FST maps to include them", stacklevel=2)
    states = [s for s in STATES if events["state"].eq(s).any()]

    rows = []
    for vid, g in events.groupby("video_id", sort=True):
        vid = str(vid)
        s = cfg.session(vid)
        end = s.start_s + s.duration_s if s.duration_s is not None else np.inf
        media_dur = g["media_duration"].max()
        if np.isfinite(media_dur):
            end = min(end, media_dur)
        if not np.isfinite(end):
            end = float(g["stop"][np.isfinite(g["stop"])].max())
        analysed = max(end - s.start_s, 0.0)

        def bouts(mask):
            sub = g[mask]
            return clip(list(zip(sub["start"], sub["stop"])), s.start_s, end)

        out = [("duration_s", "", analysed)]
        if g["zone"].notna().any():
            out += _zone_rows({z: union(bouts(g["zone"] == z)) for z in zone_names}, analysed, vid, s.start_zone)
        if g["state"].notna().any():
            iv = {st: union(bouts(g["state"] == st)) for st in states}
            coded = total([b for v in iv.values() for b in v])
            if sum(total(v) for v in iv.values()) > coded + _TOL:
                warnings.warn(f"{vid}: FST states overlap in time; they should be mutually exclusive",
                              stacklevel=2)
            out += [(r["metric"], r["zone"], r["value"]) for r in fst_rows(iv, analysed, cfg)]
            out.append(("fst_uncoded_s", "", max(analysed - coded, 0.0)))
        for metric, zone, value in out:
            rows.append(dict(video_id=vid, tool=TOOL, metric=metric, zone=zone, value=float(value),
                             kind=METRICS[metric][1]))
    return pd.DataFrame(rows, columns=LONG_COLUMNS)


def _zone_rows(iv: dict[str, list], analysed: float, vid: str, start_zone: str | None) -> list[tuple]:
    covered = total([b for v in iv.values() for b in v])
    if sum(total(v) for v in iv.values()) > covered + _TOL:
        warnings.warn(f"{vid}: zone states overlap in time; they should be mutually exclusive", stacklevel=2)
    # zone sequence; a gap keeps the previous zone, so a return to it is not an entry
    seq = []
    for _, _, z in sorted((a, b, z) for z, v in iv.items() for a, b in v):
        if not seq or seq[-1] != z:
            seq.append(z)
    if seq and start_zone and seq[0] != start_zone:
        warnings.warn(f"{vid}: first coded zone is {seq[0]!r}, not the start zone {start_zone!r}", stacklevel=3)
    out = [("unassigned_time_s", "", max(analysed - covered, 0.0))]
    for z, v in iv.items():
        t_in = total(v)
        latency = (0.0 if seq[0] == z else v[0][0]) if v else np.nan
        out += [
            ("time_s", z, t_in),
            ("time_pct", z, 100.0 * t_in / analysed if analysed else np.nan),
            ("entries", z, float(seq[1:].count(z))),
            ("latency_s", z, latency),
        ]
    return out


def read_boris(paths: list[str | Path], cfg: Config, zone_map: dict[str, str] | None = None,
               fst_map: dict[str, str] | None = None, video_id: str = "media") -> pd.DataFrame:
    """BORIS exports -> long metric table (tool = "manual")."""
    events = pd.concat([read_events(p, video_id) for p in paths], ignore_index=True)
    dup = events.duplicated()
    if dup.any():
        warnings.warn(f"{int(dup.sum())} duplicate BORIS bouts (same video, behaviour and times) dropped",
                      stacklevel=2)
        events = events[~dup]
    return events_to_long(events, cfg, zone_map, fst_map)
