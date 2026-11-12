"""RATEL's trainable offline pose classifier and its versioned model files.

Only the pose CSV format comes from DLC. Feature engineering, training,
postprocessing and summaries run here without SimBA, LabGym or DeepEthogram.
Models use the thirteen markers and four arena landmarks in pose_features.py.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .ethogram import behaviour_name, normalize_events
from .branding import SOFTWARE_NAME, SOFTWARE_FULL_NAME
from .fst import runs
from .pose_features import PARTS, pose_features, read_pose

MODEL_VERSION = 1
DEFAULT_CLASSES = ["none", "grooming", "supported_rearing", "unsupported_rearing"]
SEED = 20261017


def feature_signature() -> str:
    """Invalidate saved models when the feature or pose-reader code changes."""
    from .adapters import deeplabcut
    from . import pose_features as features_module

    digest = hashlib.sha256()
    for module in (features_module, deeplabcut):
        digest.update(Path(module.__file__).read_bytes())
    return digest.hexdigest()


def fit_model(x, y, feature_names, classes=None, trained_on=(), train_stride=5, seed=SEED):
    import sklearn
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from threadpoolctl import threadpool_limits

    classes = list(classes or DEFAULT_CLASSES)
    x, y = np.asarray(x, np.float32), np.asarray(y)
    if len(classes) < 2 or classes[0] != "none" or len(set(classes)) != len(classes):
        raise ValueError("Classes must start with 'none' and contain unique behaviour names")
    if x.ndim != 2 or x.shape[1] != len(feature_names) or len(y) != len(x):
        raise ValueError("Training features, column names and labels do not match")
    if not len(y) or not np.isin(y, np.arange(len(classes))).all():
        raise ValueError("Training labels must be declared class indices; exclude ambiguous labels first")
    if set(np.unique(y)) != set(range(len(classes))):
        raise ValueError("Every declared class, including 'none', needs training examples")
    model = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15, max_depth=6,
                                       learning_rate=0.08, class_weight="balanced",
                                       early_stopping=False, random_state=seed))
    with threadpool_limits(limits=8):
        model.fit(x, y)
    return dict(format_version=MODEL_VERSION, estimator=model, classes=classes,
                software=SOFTWARE_NAME, software_full_name=SOFTWARE_FULL_NAME,
                feature_names=list(feature_names), feature_signature=feature_signature(),
                trained_on=list(trained_on), training_frames=len(y), train_stride=train_stride,
                seed=seed, sklearn_version=sklearn.__version__,
                created_utc=datetime.now(timezone.utc).isoformat(),
                algorithm="histogram gradient boosting", input="13-marker overhead pose + 4 arena corners",
                use="offline behaviour proposals; review against annotations on the target setup")


def save_model(model: dict, path: str | Path):
    import joblib

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path, compress=3)
    card = {k: v for k, v in model.items() if k != "estimator"}
    path.with_suffix(".json").write_text(json.dumps(card, indent=2), encoding="utf-8")


def load_model(path: str | Path) -> dict:
    import joblib
    import sklearn

    model = joblib.load(path)
    if model.get("format_version") != MODEL_VERSION:
        raise ValueError("Unsupported model format; train a model with this version of RATEL")
    if model.get("sklearn_version") != sklearn.__version__:
        raise ValueError(f"Model needs scikit-learn {model.get('sklearn_version')}; installed {sklearn.__version__}")
    if model.get("feature_signature") != feature_signature():
        raise ValueError("Pose feature code changed since training; retrain the model")
    return model


def predict_features(model: dict, features: pd.DataFrame) -> np.ndarray:
    from threadpoolctl import threadpool_limits

    if list(features) != model["feature_names"]:
        raise ValueError("Pose feature columns differ from the saved training schema")
    with threadpool_limits(limits=8):
        values = model["estimator"].predict_proba(features.to_numpy(np.float32))
    p = np.zeros((len(features), len(model["classes"])), np.float32)
    p[:, model["estimator"].classes_.astype(int)] = values
    return p


def predict_pose(path, fps: float, model: dict, max_frames: int | None = None):
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Pose inference needs a positive fps")
    pose, prob, corners, raw = read_pose(path)
    if max_frames is not None:
        if max_frames < 2 or max_frames > len(raw):
            raise ValueError("max_frames must be between two and the pose row count")
        pose = {k: v[:max_frames] for k, v in pose.items()}
        prob = {k: v[:max_frames] for k, v in prob.items()}
    _, features, _ = pose_features(pose, prob, corners, fps)
    p = predict_features(model, features)
    # Long missing-pose intervals cannot establish absence of a behaviour.
    known = np.mean([np.isfinite(pose[k]).all(axis=1) for k in PARTS], axis=0) >= 0.5
    frame = np.arange(len(features))
    table = pd.DataFrame({"frame": frame, "time_s": frame / fps,
                          "pose_valid": known, "feature_missing_fraction": features.isna().mean(axis=1)})
    for j, name in enumerate(model["classes"]):
        table[f"p_{name}"] = p[:, j]
    return table


def prediction_labels(table: pd.DataFrame, classes: list[str], smooth_s=0.2, min_bout_s=0.2):
    """Smooth each observed segment; short named bouts become background.

    Unknown frames stay -1 and split bouts. Smoothing never crosses a missing
    frame or a missing-pose interval. No gaps are filled between named bouts.
    """
    if smooth_s < 0 or min_bout_s < 0:
        raise ValueError("Smoothing and minimum bout duration must be nonnegative")
    t = table.time_s.to_numpy(float)
    if len(t) < 2 or not np.isfinite(t).all() or (np.diff(t) <= 0).any():
        raise ValueError("Predictions need at least two strictly increasing finite timestamps")
    dt = float(np.median(np.diff(t)))
    p = table[[f"p_{name}" for name in classes]].to_numpy(float)
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any() or not np.allclose(p.sum(1), 1, atol=1e-5):
        raise ValueError("Class probabilities must be finite, in [0, 1], and sum to one")
    known = table.get("pose_valid", pd.Series(True, index=table.index)).to_numpy(bool)
    label = np.full(len(t), -1, int)
    w = max(1, int(round(smooth_s / dt))) | 1
    gaps = np.flatnonzero(np.diff(t) > 1.5 * dt) + 1
    for a, b in runs(known):
        edges = [a, *gaps[(gaps > a) & (gaps < b)].tolist(), b]
        for lo, hi in zip(edges[:-1], edges[1:]):
            values = pd.DataFrame(p[lo:hi]).rolling(w, center=True, min_periods=1).mean().to_numpy()
            local = values.argmax(axis=1)
            lengths = np.minimum(np.r_[t[lo + 1:hi], t[hi - 1] + dt] - t[lo:hi], dt)
            for j in range(1, len(classes)):
                for x, y in runs(local == j):
                    if lengths[x:y].sum() < min_bout_s - 1e-9:
                        local[x:y] = 0
            label[lo:hi] = local
    return label


def labels_from_events(events, video_id, times, classes):
    """Strict majority of available observers; conflicting labels are unknown.

    The manifest, not the last positive annotation, defines the scored window.
    Every listed observer must have scored all declared classes in that window.
    """
    ev = events[events.video_id == video_id]
    observers = sorted(ev.observer.dropna().unique()) if "observer" in ev else []
    groups = [ev[ev.observer == name] for name in observers] if observers else [ev]
    targets = []
    for g in groups:
        masks = np.zeros((len(times), len(classes) - 1), bool)
        for j, name in enumerate(classes[1:]):
            for a, b in g.loc[g.behaviour == name, ["start", "stop"]].itertuples(index=False, name=None):
                masks[:, j] |= (times >= a) & (times < b)
        y = np.where(masks.any(1), masks.argmax(1) + 1, 0)
        y[masks.sum(1) > 1] = -1
        targets.append(y)
    votes = np.stack([(np.column_stack(targets) == j).sum(1) for j in range(len(classes))], axis=1)
    y = votes.argmax(1)
    y[votes.max(1) <= len(groups) // 2] = -1
    return y


def train_manifest(manifest, events_path, behaviours, output, stride=5):
    """Train on explicitly scored recordings; deployment fitting is not a benchmark."""
    if stride < 1:
        raise ValueError("Training stride must be positive")
    manifest = Path(manifest)
    records = pd.read_csv(manifest, dtype={"video_id": str})
    required = {"video_id", "pose_path", "fps", "duration_s"}
    if not required.issubset(records) or records.empty or records.video_id.duplicated().any():
        raise ValueError(f"Manifest needs unique recordings and columns {sorted(required)}")
    events = normalize_events(pd.read_csv(events_path, dtype={"video_id": str}))
    classes = ["none", *[behaviour_name(b) for b in behaviours]]
    if len(set(classes)) != len(classes) or len(classes) < 2:
        raise ValueError("Specify unique named behaviours, excluding 'none'")
    if not events.event_type.eq("state").all() or not set(events.behaviour).issubset(classes[1:]):
        raise ValueError("Training events must be states with names from --behaviours")
    if "observer" in events and events.observer.isna().any():
        raise ValueError("Observer IDs must be present for every training annotation")
    if not set(events.video_id).issubset(set(records.video_id)):
        raise ValueError("Annotations include video IDs absent from the training manifest")
    chunks, targets, names = [], [], None
    for rec in records.to_dict("records"):
        path = Path(rec["pose_path"])
        if not path.is_absolute():
            path = manifest.parent / path
        fps, duration = float(rec["fps"]), float(rec["duration_s"])
        start = float(rec.get("start_s", 0.0))
        if not np.isfinite([fps, start, duration]).all() or fps <= 0 or start < 0 or duration <= 0:
            raise ValueError(f"{rec['video_id']}: invalid frame rate or scored window")
        pose, prob, corners, _ = read_pose(path)
        _, features, _ = pose_features(pose, prob, corners, fps)
        t = np.arange(len(features)) / fps
        if start + duration > len(features) / fps + 1e-6:
            raise ValueError(f"{rec['video_id']}: scored window exceeds pose data")
        keep = np.flatnonzero((t >= start) & (t < start + duration))[::stride]
        y = labels_from_events(events, rec["video_id"], t[keep], classes)
        known = np.mean([np.isfinite(pose[k][keep]).all(1) for k in PARTS], axis=0) >= 0.5
        good = (y >= 0) & known
        chunks.append(features.iloc[keep[good]].to_numpy(np.float32))
        targets.append(y[good])
        names = list(features)
    model = fit_model(np.concatenate(chunks), np.concatenate(targets), names, classes,
                      records.video_id.tolist(), stride)
    model.update(training_manifest=str(manifest.resolve()),
                 annotation_sha256=hashlib.sha256(Path(events_path).read_bytes()).hexdigest(),
                 validation="Not evaluated by this training command; evaluate on separate recordings")
    save_model(model, output)
    return model
