"""Open-field benchmark: prepare labelled videos, classify, and analyse movement.

Whole original recordings are held out, never random frames. Native external
tools run in separate environments (tool_worker.py); custom baseline names
explicitly identify our own models. Cached results make every step resumable.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")

import numpy as np
import pandas as pd

from behaviour.adapters.deeplabcut import trajectory
from behaviour.config import load_config
from behaviour.metrics import metrics_for_trajectory
from behaviour.pose_features import PARTS, SIMBA_PARTS, pose_features, rater_labels, read_pose

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = Path(r"E:\Diploma\BehaviourBenchmarks")
DEFAULT_OUT = ROOT / "output" / "open_field_benchmark"
NAMES = ["none", "grooming", "supported_rearing", "unsupported_rearing"]
SEED = 20261017


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str, allow_nan=False), encoding="utf-8")


def catalog(data):
    df = pd.read_csv(data / "catalog.csv", keep_default_na=False)
    return df[df.assay == "open_field"].copy()


def labels(data):
    path = data / "datasets/eth_zurich_3608658/references/DLCAnalyzer/data/OFT/Labels/AllLabDataOFT_final.csv"
    return pd.read_csv(path, sep=";")


def config_file(path, corners, start, duration):
    center = corners.mean(axis=0)
    side = np.mean(np.linalg.norm(corners - np.roll(corners, -1, axis=0), axis=1))
    text = ["# Generated geometry; arena side 0.42 m is an explicit scaling assumption.",
            "[video]", f"px_per_m = {side / 0.42:.8f}", "mask_margin_px = 10.0", "[window]",
            f"start_s = {start:.8f}", f"duration_s = {duration:.8f}", "[zone_rules]",
            'hysteresis_mode = "outer"', "hysteresis_m = 0.01", "[tracker]",
            "min_area_px = 150", "max_area_px = 5000"]
    for name, factor in [("center", 0.5), ("periphery", 1.1)]:
        points = center + factor * (corners - center)
        pts = ", ".join(f"[{x:.4f}, {y:.4f}]" for x, y in points)
        text += ["[[zones]]", f'name = "{name}"', 'shape = "polygon"', f"points = [{pts}]"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(text) + "\n", encoding="utf-8")


def prepare(data, out):
    from sklearn.model_selection import KFold

    table = labels(data)
    entries = catalog(data)
    entries = entries[entries.dataset == "eth_zurich_3608658"].sort_values("source_video_id")
    project = out / "simba_project"
    for d in ["cache", "configs", "predictions", "simba_project/csv/outlier_corrected_movement_location",
              "simba_project/csv/features_extracted", "simba_project/logs/measures/pose_configs/bp_names"]:
        (out / d).mkdir(parents=True, exist_ok=True)
    metadata, quality = [], []
    for rec in entries.to_dict("records"):
        vid = rec["source_video_id"]
        source = table.loc[table.ID == vid, "DLCFile"].unique()
        if len(source) != 1:
            raise ValueError(f"{vid}: ambiguous DLC mapping")
        track = data / "datasets/eth_zurich_3608658/references/DLCAnalyzer/data/OFT/Output_DLC" / source[0]
        pose, prob, corners, raw = read_pose(track)
        n = min(int(rec["frames"]), len(raw))
        t = np.arange(n) / float(rec["fps"])
        raters, y, observers, start, end = rater_labels(table, vid, t)
        keep = (t >= start) & (t < end)
        shape, full, side = pose_features(pose, prob, corners, float(rec["fps"]))
        shape, full = shape.iloc[:n], full.iloc[:n]
        np.savez_compressed(out / "cache" / f"{vid}.npz", shape=shape.to_numpy()[keep],
                            temporal=full.to_numpy()[keep], y=y[keep], raters=raters[keep],
                            t=t[keep], frame=np.flatnonzero(keep), observers=np.array(observers))
        config_file(out / "configs" / f"{vid}.toml", corners, start, end - start)
        simba = {}
        for name, source_name in SIMBA_PARTS.items():
            xy = pd.DataFrame(pose[source_name]).ffill().bfill().fillna(0).to_numpy()
            simba[f"{name}_x"], simba[f"{name}_y"] = xy[:, 0], xy[:, 1]
            simba[f"{name}_p"] = prob[source_name]
        simba = pd.DataFrame(simba)
        header_file = ROOT / "output/tool_envs/simba/Lib/site-packages/simba/assets/lookups/feature_extraction_headers.csv"
        if header_file.exists():
            headers = pd.read_csv(header_file)["1 animal 8 body-parts"].dropna().tolist()
            simba = simba[headers]
        simba.to_csv(project / "csv/outlier_corrected_movement_location" / f"{vid}.csv")
        metadata.append(dict(video_id=vid, video_path=str(data / rec["video_path"]), pose_path=str(track),
                             sha256=rec["sha256"], recording_key=rec["reference_recording_key"],
                             fps=float(rec["fps"]), width=int(rec["width"]), height=int(rec["height"]),
                             px_per_mm=float(side / 420), start_s=start, end_s=end,
                             frames=int(keep.sum()), observers=observers,
                             pose_frames=len(raw), video_header_frames=int(rec["frames"])))
        quality.append(dict(video_id=vid, scored_s=end-start, frames=int(keep.sum()),
                            incomplete_marker_observers=",".join(r for r in observers if
                                len(table[(table.ID == vid) & (table.Experimenter == r) &
                                          table.type.str.match(r"Start.?End")]) < 2),
                            consensus_ambiguous_fraction=float((y[keep] < 0).mean()),
                            feature_missing_fraction=float(np.isnan(full.to_numpy()[keep]).mean()),
                            raw_pose_low_confidence_fraction=float(np.mean([
                                np.mean(prob[p][:n][keep] < 0.9) for p in PARTS])),
                            pose_minus_video_frames=len(raw)-int(rec["frames"])))
        print(f"Prepared {vid}: {keep.sum()} frames, {len(observers)} observers", flush=True)
    ids = np.array([r["video_id"] for r in metadata])
    folds = []
    for fold, (train, test) in enumerate(KFold(5, shuffle=True, random_state=SEED).split(ids)):
        train_ids, test_ids = ids[train].tolist(), ids[test].tolist()
        for field in ("sha256", "recording_key"):
            a = {r[field] for r in metadata if r["video_id"] in train_ids}
            b = {r[field] for r in metadata if r["video_id"] in test_ids}
            assert not a & b, f"{field}: same original recording in training and evaluation"
        folds.append(dict(fold=fold, train=train_ids, test=test_ids))
    save_json(out / "recordings.json", metadata)
    save_json(out / "splits.json", dict(seed=SEED, unit="whole original video", folds=folds))
    save_json(out / "features.json", {"shape": list(shape), "temporal": list(full)})
    pd.DataFrame(quality).to_csv(out / "data_quality.csv", index=False)
    pd.Series(simba.columns[::3].str.removesuffix("_x")).to_csv(
        project / "logs/measures/pose_configs/bp_names/project_bp_names.csv", header=False, index=False)
    pd.DataFrame([dict(Video=r["video_id"], fps=r["fps"], Resolution_width=r["width"],
                       Resolution_height=r["height"], Distance_in_mm=420, **{"pixels/mm": r["px_per_mm"]})
                  for r in metadata]).to_csv(project / "logs/video_info.csv", index=False)
    ini = f"""[General settings]
project_path = {project.resolve()}
project_name = open_field_benchmark
workflow_file_type = csv
animal_no = 1
[SML settings]
no_targets = 3
target_name_1 = grooming
target_name_2 = supported_rearing
target_name_3 = unsupported_rearing
[create ensemble settings]
pose_estimation_body_parts = 8
"""
    (project / "project_config.ini").write_text(ini, encoding="utf-8")
    print(f"Prepared {len(metadata)} unique labelled videos and five fixed video folds", flush=True)


def models():
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    imp = lambda: SimpleImputer(strategy="median", keep_empty_features=True)
    forest = lambda seed: RandomForestClassifier(n_estimators=160, max_depth=18, min_samples_leaf=3,
                                                  class_weight="balanced_subsample", n_jobs=8, random_state=seed)
    return {
        "majority": ("shape", lambda: make_pipeline(imp(), DummyClassifier(strategy="most_frequent"))),
        "pose_logistic": ("shape", lambda: make_pipeline(imp(), StandardScaler(), LogisticRegression(
            C=1, class_weight="balanced", max_iter=300, random_state=SEED))),
        "pose_rf": ("shape", lambda: make_pipeline(imp(), forest(SEED))),
        "temporal_rf": ("temporal", lambda: make_pipeline(imp(), forest(SEED))),
        "temporal_rf_seed2": ("temporal", lambda: make_pipeline(imp(), forest(SEED+1))),
        "temporal_extra_trees": ("temporal", lambda: make_pipeline(imp(), ExtraTreesClassifier(
            n_estimators=160, min_samples_leaf=3, class_weight="balanced", n_jobs=8, random_state=SEED))),
        "temporal_hist_gradient": ("temporal", lambda: make_pipeline(imp(), HistGradientBoostingClassifier(
            max_iter=100, max_leaf_nodes=15, max_depth=6, learning_rate=0.08, class_weight="balanced",
            early_stopping=False, random_state=SEED))),
        "temporal_mlp": ("temporal", lambda: make_pipeline(imp(), StandardScaler(), MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=60, early_stopping=False, random_state=SEED))),
        "simba_rf": ("simba", lambda: make_pipeline(imp(), forest(SEED))),
        "simba_extra_trees": ("simba", lambda: make_pipeline(imp(), ExtraTreesClassifier(
            n_estimators=160, min_samples_leaf=3, class_weight="balanced", n_jobs=8, random_state=SEED))),
    }


def load_cache(out, vid, source):
    with np.load(out / "cache" / f"{vid}.npz") as z:
        data = {k: z[k] for k in ["y", "t", "frame", "raters", "observers"]}
        if source != "simba":
            data["x"] = z[source]
    if source == "simba":
        path = out / "simba_project/csv/features_extracted" / f"{vid}.csv"
        features = pd.read_csv(path, index_col=0)
        # Retain engineered features; exclude raw camera coordinates/likelihoods.
        cols = [c for c in features if not c.endswith(("_x", "_y", "_p"))]
        data["x"] = features.iloc[data["frame"]][cols].replace([np.inf, -np.inf], np.nan).to_numpy(np.float32)
    return data


def classify(out, selected=None, train_stride=5):
    from threadpoolctl import threadpool_limits

    registry = models()
    selected = selected or list(registry)
    unknown = set(selected) - set(registry)
    if unknown:
        raise ValueError(f"Unknown models {sorted(unknown)}; available: {sorted(registry)}")
    if train_stride < 1:
        raise ValueError("Training stride must be positive")
    folds = json.loads((out / "splits.json").read_text())["folds"]
    for name in selected:
        timing_path = out / "timing" / f"{name}.csv"
        timing_path.parent.mkdir(exist_ok=True)
        timings = pd.read_csv(timing_path).to_dict("records") if timing_path.exists() else []
        source, make_model = registry[name]
        if source == "simba" and not list((out / "simba_project/csv/features_extracted").glob("*.csv")):
            print(f"SKIPPED {name}: native SimBA features not generated yet", flush=True)
            continue
        print(f"Loading features for {name}", flush=True)
        ids = sorted({v for f in folds for v in f["test"]})
        cache = {vid: load_cache(out, vid, source) for vid in ids}
        for fold in folds:
            dest = out / "predictions" / name
            dest.mkdir(parents=True, exist_ok=True)
            if all((dest / f"{vid}.npz").exists() for vid in fold["test"]):
                continue
            chunks, targets = [], []
            for vid in fold["train"]:
                c = cache[vid]
                ix = np.arange(0, len(c["y"]), train_stride)
                ix = ix[c["y"][ix] >= 0]
                chunks.append(c["x"][ix])
                targets.append(c["y"][ix])
            x, y = np.concatenate(chunks), np.concatenate(targets)
            model = make_model()
            started = time.perf_counter()
            with threadpool_limits(limits=8):
                model.fit(x, y)
                for vid in fold["test"]:
                    c = cache[vid]
                    p = np.zeros((len(c["y"]), len(NAMES)), np.float32)
                    p[:, model.classes_.astype(int)] = model.predict_proba(c["x"])
                    np.savez_compressed(dest / f"{vid}.npz", p=p, t=c["t"], y=c["y"],
                                        fold=fold["fold"], trained_on=np.array(fold["train"]))
            elapsed = time.perf_counter() - started
            timings.append(dict(tool=name, fold=fold["fold"], training_frames=len(y),
                                test_videos=len(fold["test"]), feature_count=x.shape[1], seconds=elapsed))
            pd.DataFrame(timings).drop_duplicates(["tool", "fold"], keep="last").to_csv(timing_path, index=False)
            print(f"{name} fold {fold['fold'] + 1}/5: {len(y)} training frames, {elapsed:.1f}s", flush=True)
        del cache


def movement(data, out, limit=None):
    from behaviour.adapters.deeplabcut import landmarks

    entries = catalog(data)
    entries["priority"] = np.where(entries.dataset == "eth_zurich_3608658", 0, 1)
    entries = entries.sort_values(["priority", "recording_id"]).drop_duplicates("sha256")
    # Missing recording keys are distinct, not one shared empty recording.
    entries["dedup_key"] = entries.reference_recording_key.where(
        entries.reference_recording_key != "", entries.sha256)
    entries = entries.drop_duplicates("dedup_key")
    if limit:
        entries = entries.head(limit)
    rows, status = [], []
    (out / "movement_configs").mkdir(parents=True, exist_ok=True)
    table = labels(data)
    for k, rec in enumerate(entries.to_dict("records")):
        paths = [data / p for p in json.loads(rec["tracking_paths"]) if p.lower().endswith(".csv")]
        if len(paths) != 1:
            status.append(dict(recording_id=rec["recording_id"], status="unsupported_tracking_format", paths=len(paths)))
            continue
        vid = rec["source_video_id"] if rec["dataset"] == "eth_zurich_3608658" else rec["recording_id"]
        try:
            corners = landmarks(paths[0], ["tl", "tr", "br", "bl"])
            duration = float(rec["duration_s"])
            start, end = 5.0, max(5.0, duration - 5.0)
            if rec["dataset"] == "eth_zurich_3608658":
                _, _, _, start, end = rater_labels(
                    table, vid, np.arange(int(rec["frames"])) / float(rec["fps"]))
            if end <= start:
                raise ValueError("No positive shared analysis window")
            cfg_path = out / "movement_configs" / f"{vid}.toml"
            config_file(cfg_path, corners, start, end-start)
            traj = trajectory(paths[0], float(rec["fps"]))
            traj = traj.iloc[:int(rec["frames"])]
            metrics = metrics_for_trajectory(traj, load_config(cfg_path), vid, "dlc")
            if metrics.loc[metrics.metric == "duration_s", "value"].iloc[0] <= 0:
                raise ValueError("No trajectory frames in the scoring window")
            metrics["dataset"] = rec["dataset"]
            rows.append(metrics)
            status.append(dict(recording_id=rec["recording_id"], video_id=vid, status="analysed", frames=len(traj)))
        except Exception as e:
            status.append(dict(recording_id=rec["recording_id"], video_id=vid, status="failed", reason=f"{type(e).__name__}: {e}"))
        if (k + 1) % 20 == 0:
            print(f"Movement metrics: {k+1}/{len(entries)} recordings", flush=True)
            pd.DataFrame(status).to_csv(out / "movement_status.csv", index=False)
    if rows:
        pd.concat(rows, ignore_index=True).to_csv(out / "movement_metrics.csv", index=False)
    pd.DataFrame(status).to_csv(out / "movement_status.csv", index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["prepare", "classify", "movement", "report", "all"])
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--train-stride", type=int, default=5)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    if a.command in ("prepare", "all"):
        prepare(a.data, a.output)
    if a.command in ("classify", "all"):
        classify(a.output, a.models, a.train_stride)
    if a.command in ("movement", "all"):
        movement(a.data, a.output, a.limit)
    if a.command in ("report", "all"):
        from open_field_report import report
        report(a.output)


if __name__ == "__main__":
    main()
