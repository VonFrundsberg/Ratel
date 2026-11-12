"""Pose features for local, explicitly named classifier baselines.

These are our features, not a reimplementation claimed to be SimBA or LabGym.
Coordinates are scaled by the arena side; camera translation and rotation are
removed from shape features. Temporal features use the current video only.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from .adapters.deeplabcut import read_dlc

PARTS = ["nose", "headcentre", "neck", "earl", "earr", "bodycentre", "bcl", "bcr",
         "hipl", "hipr", "tailbase", "tailcentre", "tailtip"]
SIMBA_PARTS = {"Ear_left": "earl", "Ear_right": "earr", "Nose": "nose", "Center": "bodycentre",
               "Lat_left": "bcl", "Lat_right": "bcr", "Tail_base": "tailbase", "Tail_end": "tailtip"}


def read_pose(path, likelihood=0.9, max_gap_frames=12):
    raw = read_dlc(path)
    parts = [p for p in PARTS if (p, "x") in raw]
    if len(parts) != len(PARTS):
        raise ValueError(f"{path}: pose baseline needs body parts {sorted(set(PARTS) - set(parts))}")
    pose, prob = {}, {}
    for p in parts:
        prob[p] = raw[(p, "likelihood")].to_numpy(float)
        xy = raw.loc[:, [(p, "x"), (p, "y")]].copy()
        xy.columns = ["x", "y"]
        xy.loc[prob[p] < likelihood] = np.nan
        # Interpolate whole short gaps, leave long gaps for training-fold imputation.
        missing = xy.isna().any(axis=1)
        run = missing.ne(missing.shift()).cumsum()
        sizes = missing.groupby(run).transform("size")
        filled = xy.interpolate(limit_area="inside")
        xy.loc[missing & (sizes <= max_gap_frames)] = filled.loc[missing & (sizes <= max_gap_frames)]
        pose[p] = xy.to_numpy(float)
    corners = np.array([raw.loc[raw[(p, "likelihood")] > 0.95, [(p, "x"), (p, "y")]].median().values
                        for p in ["tl", "tr", "br", "bl"]], float)
    if not np.isfinite(corners).all():
        raise ValueError(f"{path}: invalid arena landmarks")
    return pose, prob, corners, raw


def pose_features(pose, prob, corners, fps=25.0):
    side = float(np.mean(np.linalg.norm(corners - np.roll(corners, -1, axis=0), axis=1)))
    if not np.isfinite(side) or side <= 0:
        raise ValueError("Invalid arena scale")
    features = {}
    for a, b in combinations(PARTS, 2):
        features[f"distance_{a}_{b}"] = np.linalg.norm(pose[a] - pose[b], axis=1) / side
    temporal = []
    for p in PARTS:
        xy = pose[p]
        speed = np.r_[np.nan, np.linalg.norm(np.diff(xy, axis=0), axis=1) * fps / side]
        features[f"speed_{p}"] = speed
        features[f"acceleration_{p}"] = np.r_[np.nan, np.diff(speed) * fps]
        features[f"likelihood_{p}"] = prob[p]
        temporal += [f"speed_{p}"]
    # Distance to arena edges separates wall-supported posture from central posture.
    for p in ("nose", "headcentre", "bodycentre", "tailbase"):
        xy = pose[p]
        distances = []
        for a, b in zip(corners, np.roll(corners, -1, axis=0)):
            ab = b - a
            q = np.clip(((xy - a) @ ab) / (ab @ ab), 0, 1)
            distances.append(np.linalg.norm(xy - (a + q[:, None] * ab), axis=1) / side)
        features[f"wall_distance_{p}"] = np.min(distances, axis=0)
        temporal.append(f"wall_distance_{p}")
    for name, a, b, c in [("body_bend", "tailbase", "bodycentre", "neck"),
                           ("neck_bend", "bodycentre", "neck", "headcentre"),
                           ("head_bend", "neck", "headcentre", "nose")]:
        u, v = pose[a] - pose[b], pose[c] - pose[b]
        den = np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1)
        features[name] = np.sum(u * v, axis=1) / np.maximum(den, 1e-9)
        temporal.append(name)
    temporal += ["distance_nose_tailbase", "distance_bcl_bcr", "distance_nose_bodycentre"]
    shape = pd.DataFrame(features).replace([np.inf, -np.inf], np.nan)
    roll = {}
    for seconds in (0.2, 0.5, 1.0):
        w = max(3, int(round(seconds * fps)) | 1)
        r = shape[temporal].rolling(w, center=True, min_periods=1)
        for stat, values in [("mean", r.mean()), ("std", r.std(ddof=0))]:
            for c in temporal:
                roll[f"{c}_{stat}_{seconds}s"] = values[c].to_numpy()
    full = pd.concat([shape, pd.DataFrame(roll)], axis=1)
    return shape.astype(np.float32), full.astype(np.float32), side


def rater_labels(table: pd.DataFrame, video_id: str, frame_times: np.ndarray,
                  names=("Grooming", "Supported", "Unsupported")):
    """Labels are -1 for overlapping conflicting annotations, 0 for none.

    Returns the intersection of the raters' explicitly scored start/end windows.
    Camera frames are never split randomly between training and evaluation.
    """
    sub = table[table.ID == video_id]
    observers = sorted(sub.Experimenter.unique())
    y, windows = [], []
    for rater in observers:
        g = sub[sub.Experimenter == rater]
        bounds = g.loc[g.type.str.match(r"Start.?End"), "from"]
        if len(bounds) == 1 and float(bounds.iloc[0]) < 5:
            # Two source sessions lack an end marker. Do not infer absence after
            # the final annotated bout; stop the shared window there instead.
            end = float(g.loc[~g.type.str.match(r"Start.?End"), "to"].max())
            windows.append((float(bounds.iloc[0]), end))
        elif len(bounds) >= 2:
            windows.append((float(bounds.min()), float(bounds.max())))
        else:
            raise ValueError(f"{video_id}/{rater}: no usable scoring window")
        labels = np.zeros((len(frame_times), len(names)), bool)
        for j, name in enumerate(names):
            for a, b in g.loc[g.type == name, ["from", "to"]].itertuples(index=False, name=None):
                labels[:, j] |= (frame_times >= a) & (frame_times < b)
        target = np.where(labels.any(axis=1), labels.argmax(axis=1) + 1, 0).astype(np.int8)
        target[labels.sum(axis=1) > 1] = -1
        y.append(target)
    start, end = max(a for a, b in windows), min(b for a, b in windows)
    if end <= start:
        raise ValueError(f"{video_id}: raters have no shared scored window")
    matrix = np.column_stack(y)
    votes = np.stack([(matrix == k).sum(axis=1) for k in range(len(names) + 1)], axis=1)
    consensus = votes.argmax(axis=1).astype(np.int8)
    consensus[votes.max(axis=1) < len(observers) // 2 + 1] = -1
    return matrix, consensus, observers, start, end
