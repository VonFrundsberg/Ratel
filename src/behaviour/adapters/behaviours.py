"""Behaviour exports from SimBA, DeepEthogram, LabGym and other classifiers.

CSV formats are explicit: intervals, frame-wise binary/probability columns,
or one categorical label per frame. Native DeepEthogram HDF5 is also supported.
No installed classifier is needed to import its results. These adapters do not
claim to run or train the producing software.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..ethogram import EVENT_COLUMNS, normalize_events
from ..fst import runs


def _read(path: str | Path, columns: dict[str, str] | None) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_excel(path) if path.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(
        path, sep="\t" if path.suffix.lower() == ".tsv" else ",")
    if columns:
        missing = set(columns.values()) - set(df)
        if missing:
            raise ValueError(f"{path}: missing source columns {sorted(missing)}")
        df = df.rename(columns={v: k for k, v in columns.items()})
    return df


def read_intervals(path: str | Path, video_id: str | None = None,
                   columns: dict[str, str] | None = None,
                   mapping: dict[str, str] | None = None) -> pd.DataFrame:
    df = _read(path, columns)
    if video_id is not None or "video_id" not in df:
        df["video_id"] = video_id or Path(path).stem
    return normalize_events(df, mapping)


def frame_events(df: pd.DataFrame, video_id: str, fps: float | None,
                 behaviours: list[str] | None = None, threshold: float = 0.5,
                 label_column: str | None = None,
                 mapping: dict[str, str] | None = None) -> pd.DataFrame:
    if not 0 <= threshold <= 1:
        raise ValueError("Probability threshold must be between 0 and 1")
    if "time_s" in df:
        t = df.time_s.to_numpy(float)
    else:
        if fps is None or fps <= 0:
            raise ValueError("A positive fps is required for frame-indexed behaviour exports")
        t = df["frame"].to_numpy(float) / fps if "frame" in df else np.arange(len(df)) / fps
    if not len(t):
        raise ValueError("An empty frame export does not establish that a behaviour was absent")
    if not np.isfinite(t).all() or t[0] < 0 or (np.diff(t) <= 0).any():
        raise ValueError("Frame times must be finite, nonnegative and strictly increasing")
    dt = 1 / fps if fps else float(np.median(np.diff(t))) if len(t) > 1 else np.nan
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("fps is required for a single-frame export")
    # Missing frame indices are unobserved time, never bridged into one bout.
    stop = np.minimum(np.r_[t[1:], t[-1] + dt], t + dt)
    breaks = np.r_[False, np.diff(t) > dt * 1.5]
    if label_column:
        labels = df[label_column].astype(str)
        names = behaviours or sorted(set(labels) - {"nan", "", "NA"})
        masks = {name: (labels == name).to_numpy() for name in names}
    else:
        if not behaviours:
            raise ValueError("Specify the scored behaviour columns; probability/features columns are not guessed")
        masks = {}
        for name in behaviours:
            values = pd.to_numeric(df[name], errors="raise").to_numpy(float)
            if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
                raise ValueError(f"{name}: predictions must be finite binary values or probabilities in [0, 1]")
            masks[name] = values >= threshold
    out = []
    for name, mask in masks.items():
        for a, b in runs(mask):
            edges = [a, *(np.flatnonzero(breaks[a + 1:b]) + a + 1).tolist(), b]
            out += [dict(video_id=video_id, behaviour=name, start=float(t[x]), stop=float(stop[y - 1]),
                         event_type="state") for x, y in zip(edges[:-1], edges[1:])]
    events = normalize_events(pd.DataFrame(out, columns=EVENT_COLUMNS), mapping)
    events.attrs.update(video_id=video_id, media_duration=float(stop[-1]))
    return events


def read_frames(path: str | Path, fps: float | None, behaviours: list[str] | None,
                threshold: float = 0.5, video_id: str | None = None,
                columns: dict[str, str] | None = None, label_column: str | None = None,
                mapping: dict[str, str] | None = None) -> pd.DataFrame:
    return frame_events(_read(path, columns), video_id or Path(path).stem, fps,
                        behaviours, threshold, label_column, mapping)


def read_deepethogram(path: str | Path, fps: float, behaviours: list[str],
                      dataset: str | None = None, threshold: float = 0.5,
                      video_id: str | None = None) -> pd.DataFrame:
    import h5py

    with h5py.File(path, "r") as f:
        names = []
        f.visititems(lambda n, obj: names.append(n) if isinstance(obj, h5py.Dataset) and n.split("/")[-1] == "P" else None)
        if dataset is None:
            if len(names) != 1:
                raise ValueError(f"Choose an HDF5 probability dataset; available: {names}")
            dataset = names[0]
        p = np.asarray(f[dataset])
    if p.ndim != 2 or p.shape[1] != len(behaviours):
        raise ValueError(f"Prediction shape {p.shape} does not match {len(behaviours)} behaviour names")
    return frame_events(pd.DataFrame(p, columns=behaviours), video_id or Path(path).stem,
                        fps, behaviours, threshold)
