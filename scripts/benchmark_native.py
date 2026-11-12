"""Benchmark RATEL against the existing held-out software-component trials.

Training uses the prepared pose features; evaluation recomputes features from
the source pose CSV through the deployment API. Existing benchmark outputs and
source recordings are read-only. Run from the workspace root with .venv Python.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")

import numpy as np
import pandas as pd

from behaviour.branding import SOFTWARE_FULL_NAME, display_name
from behaviour.classifier import (DEFAULT_CLASSES, SEED, feature_signature, fit_model, load_model,
                                  predict_pose, prediction_labels, save_model)
from behaviour.compare import agreement, compare
from behaviour.config import Session, load_config
from behaviour.ethogram import behaviour_rows, normalize_events
from behaviour.metrics import metrics_for_trajectory
from behaviour.native import analyse_track, file_hash, metric_catalog, save_analysis

ROOT = Path(__file__).resolve().parents[1]
BEHAVIOURS = [*DEFAULT_CLASSES[1:], "rearing"]
VARIANTS = {"raw": (0.0, 0.0), "default": (0.2, 0.2)}


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=str), encoding="utf-8")


def validate_folds(records, folds):
    ids = {r["video_id"] for r in records}
    tested = []
    if len(ids) != len(records) or len({f["fold"] for f in folds}) != len(folds):
        raise ValueError("Duplicate recording or fold IDs")
    for fold in folds:
        train, test = set(fold["train"]), set(fold["test"])
        if train & test or train | test != ids or len(train) != len(fold["train"]) or len(test) != len(fold["test"]):
            raise ValueError("Folds must partition all unique recordings without overlap")
        for field in ("sha256", "recording_key"):
            a = {r[field] for r in records if r["video_id"] in train and r.get(field)}
            b = {r[field] for r in records if r["video_id"] in test and r.get(field)}
            if a & b:
                raise ValueError(f"Duplicate original recording crosses a fold: {field}")
        tested += fold["test"]
    if set(tested) != ids or len(tested) != len(ids):
        raise ValueError("Every recording must be tested exactly once")


def mask(y, name):
    return np.isin(y, [2, 3]) if name == "rearing" else y == DEFAULT_CLASSES.index(name)


def counts(truth, predicted):
    return dict(tp=int((truth & predicted).sum()), fp=int((~truth & predicted).sum()),
                fn=int((truth & ~predicted).sum()), tn=int((~truth & ~predicted).sum()))


def rates(c):
    tp, fp, fn, tn = (c[k] for k in ("tp", "fp", "fn", "tn"))
    divide = lambda a, b: float(a / b) if b else 0.0
    return dict(precision=divide(tp, tp + fp), recall=divide(tp, tp + fn),
                f1=divide(2 * tp, 2 * tp + fp + fn))


def manual_values(events, rec):
    result = {}
    for observer in rec["observers"]:
        ev = events[(events.video_id == rec["video_id"]) & (events.observer == observer)].copy()
        rear = ev[ev.behaviour.isin(DEFAULT_CLASSES[2:])].copy()
        rear["behaviour"] = "rearing"
        rows = behaviour_rows(pd.concat([ev, rear], ignore_index=True), rec["start_s"], rec["end_s"],
                              dict.fromkeys(BEHAVIOURS, "state"))
        result[observer] = {r["metric"]: r["value"] for r in rows}
    return result


def run_oft(reference, out, stride):
    from behaviour.adapters.deeplabcut import trajectory

    records = json.loads((reference / "recordings.json").read_text())
    folds = json.loads((reference / "splits.json").read_text())["folds"]
    validate_folds(records, folds)
    metadata = {r["video_id"]: r for r in records}
    feature_names = json.loads((reference / "features.json").read_text())["temporal"]
    cache = {}
    for vid in metadata:
        with np.load(reference / "cache" / f"{vid}.npz") as z:
            cache[vid] = {k: z[k] for k in ("temporal", "y", "t", "frame")}
    manual = normalize_events(pd.read_csv(reference / "manual_events.csv", dtype={"video_id": str}))
    frame_rows, summary_rows, metric_parts, tracker_parts, positions, timings, checks = [], [], [], [], [], [], []

    def training(ids):
        xs, ys = [], []
        for vid in ids:
            c = cache[vid]
            ix = np.arange(0, len(c["y"]), stride)
            ix = ix[c["y"][ix] >= 0]
            xs.append(c["temporal"][ix])
            ys.append(c["y"][ix])
        return np.concatenate(xs), np.concatenate(ys)

    for fold in folds:
        x, y = training(fold["train"])
        started = time.perf_counter()
        model = fit_model(x, y, feature_names, trained_on=fold["train"], train_stride=stride)
        model["validation"] = dict(fold=fold["fold"], held_out=fold["test"], unit="original video")
        model_path = out / "models" / f"fold_{fold['fold']}.joblib"
        save_model(model, model_path)
        model = load_model(model_path)
        train_s = time.perf_counter() - started
        timings.append(dict(stage="training", fold=fold["fold"], video_id="", frames=len(y), seconds=train_s))
        print(f"Fold {fold['fold'] + 1}/{len(folds)}: fitted and saved on {len(y)} frames in {train_s:.2f}s", flush=True)
        for vid in fold["test"]:
            rec, c = metadata[vid], cache[vid]
            cfg = load_config(reference / "configs" / f"{vid}.toml")
            track_path = reference / "tracks/opencv" / f"{vid}.csv"
            track = pd.read_csv(track_path).iloc[:rec["video_header_frames"]]
            if len(track) != rec["video_header_frames"]:
                raise ValueError(f"{vid}: cached native track does not cover the video header")
            started = time.perf_counter()
            predictions = predict_pose(rec["pose_path"], rec["fps"], model, len(track))
            infer_s = time.perf_counter() - started
            timings.append(dict(stage="pose_features_and_inference", fold=fold["fold"], video_id=vid,
                                frames=len(predictions), seconds=infer_s))
            if not np.allclose(predictions.time_s.to_numpy()[c["frame"]], c["t"], rtol=0, atol=1e-8):
                raise ValueError(f"{vid}: source pose clock differs from prepared evaluation clock")
            if vid in model["trained_on"] or set(model["trained_on"]) != set(fold["train"]):
                raise ValueError("Deployment artifact leaks an evaluation video")
            truth_values = manual_values(manual, rec)
            for variant, (smooth, minimum) in VARIANTS.items():
                started = time.perf_counter()
                result = analyse_track(track, cfg, vid, rec["fps"], predictions, DEFAULT_CLASSES,
                                       smooth_s=smooth, min_bout_s=minimum)
                result["metadata"].update(tracking_source="RATEL (OpenCV; existing full-video cache)",
                                          evaluation_fold=fold["fold"], trained_on=fold["train"],
                                          pose_path=rec["pose_path"], model=str(model_path.resolve()),
                                          config_sha256=file_hash(reference / "configs" / f"{vid}.toml"))
                timings.append(dict(stage=f"metrics_{variant}", fold=fold["fold"], video_id=vid,
                                    frames=len(predictions), seconds=time.perf_counter() - started))
                if variant == "default":
                    save_analysis(result, out / "recordings" / vid)
                labels = prediction_labels(predictions, DEFAULT_CLASSES, smooth, minimum)[c["frame"]]
                common = np.arange(len(c["y"])) % 10 == 0
                good = (c["y"] >= 0) & common
                for name in BEHAVIOURS:
                    tally = counts(mask(c["y"][good], name), mask(labels[good], name))
                    frame_rows.append(dict(video_id=vid, tool="native", variant=variant, behaviour=name,
                                           frames=int(good.sum()), **tally, **rates(tally)))
                metrics = result["metrics"]
                metric_parts.append(metrics.assign(variant=variant))
                predicted = metrics.set_index("metric").value
                for observer, truth in truth_values.items():
                    for metric, value in truth.items():
                        summary_rows.append(dict(video_id=vid, variant=variant, observer=observer,
                                                 metric=metric, manual=value, predicted=predicted[metric]))
                if variant == "raw":
                    dlc = trajectory(rec["pose_path"], rec["fps"]).iloc[:len(track)]
                    reference_metrics = metrics_for_trajectory(dlc, cfg, vid, "dlc")
                    movement_keys = set(reference_metrics.metric)
                    tracker_parts.extend([metrics[metrics.metric.isin(movement_keys)], reference_metrics])
                    keep = (track.time_s >= rec["start_s"]) & (track.time_s < rec["end_s"])
                    error = np.hypot(track.x_px - dlc.x_px, track.y_px - dlc.y_px).to_numpy()[keep]
                    positions.append(dict(video_id=vid, valid_position_pairs=int(np.isfinite(error).sum()),
                                          compared_frames=len(error), median_error_mm=float(np.nanmedian(error) / rec["px_per_mm"]),
                                          p95_error_mm=float(np.nanpercentile(error, 95) / rec["px_per_mm"])))
            checks.append(dict(video_id=vid, fold=fold["fold"], trained_on=fold["train"],
                               source_pose_sha256=file_hash(rec["pose_path"]), source_track_sha256=file_hash(track_path),
                               predictions=len(predictions), scored_frames=len(c["y"]),
                               feature_schema_checked=True, clock_checked=True, disjoint_training_checked=True))
            print(f"  {vid}: {len(predictions)} frames; source pose + inference {infer_s:.2f}s; both analysis variants saved", flush=True)
        pd.DataFrame(timings).to_csv(out / "timing.csv", index=False)
    frames, summaries = pd.DataFrame(frame_rows), pd.DataFrame(summary_rows)
    frames.to_csv(out / "frame_scores.csv", index=False)
    summaries.to_csv(out / "summary_scores_vs_each_observer.csv", index=False)
    pd.concat(metric_parts, ignore_index=True).to_csv(out / "metrics.csv", index=False)
    trackers = pd.concat(tracker_parts, ignore_index=True)
    trackers.to_csv(out / "tracker_metrics.csv", index=False)
    compare(trackers, "dlc").to_csv(out / "tracker_agreement.csv", index=False)
    pd.DataFrame(positions).to_csv(out / "tracker_positions.csv", index=False)
    means = summaries.groupby(["video_id", "variant", "metric"], as_index=False)[["manual", "predicted"]].mean()
    agr = [dict(variant=variant, metric=metric, **agreement(g.manual, g.predicted))
           for (variant, metric), g in means.groupby(["variant", "metric"])]
    pd.DataFrame(agr).to_csv(out / "metric_agreement.csv", index=False)
    save_json(out / "evaluation_checks.json", checks)
    # This all-recording fit is for deployment and never supplies benchmark scores.
    x, y = training(list(metadata))
    started = time.perf_counter()
    model = fit_model(x, y, feature_names, trained_on=list(metadata), train_stride=stride)
    model["validation"] = dict(scores="Separate five-fold models; see benchmark report", species="public mouse recordings",
                               independence="known original video identities; animal identity not established",
                               default_postprocessing=dict(smooth_s=.2, min_bout_s=.2),
                               deployment_fit="all development videos; do not evaluate this artifact on those videos")
    save_model(model, out / "models/native_pose.joblib")
    save_model(model, out / "models/ratel_pose.joblib")
    timings.append(dict(stage="deployment_training", fold=-1, video_id="", frames=len(y), seconds=time.perf_counter()-started))
    pd.DataFrame(timings).to_csv(out / "timing.csv", index=False)
    pd.DataFrame([dict(video_id=r["video_id"], pose_path=r["pose_path"], fps=r["fps"], start_s=r["start_s"],
                       duration_s=r["end_s"]-r["start_s"]) for r in records]).to_csv(out / "training_manifest.csv", index=False)
    return records


def comparisons(reference, out):
    frames = pd.read_csv(out / "frame_scores.csv")
    legacy = pd.read_csv(reference / "frame_scores.csv")
    legacy = legacy[(legacy.variant == "raw") & (legacy.cadence == "common_2.5hz")].copy()
    legacy["configuration"] = legacy.tool
    native = frames.copy()
    native["configuration"] = "native_" + native.variant
    columns = ["configuration", "video_id", "behaviour", "frames", "tp", "fp", "fn", "tn"]
    all_frames = pd.concat([legacy[columns], native[columns]], ignore_index=True)
    all_frames.to_csv(out / "all_frame_counts.csv", index=False)
    reference_sizes = legacy[legacy.tool == "temporal_hist_gradient"].set_index(["video_id", "behaviour"]).frames
    for variant, g in frames.groupby("variant"):
        sizes = g.set_index(["video_id", "behaviour"]).frames.reindex(reference_sizes.index)
        if not sizes.equals(reference_sizes):
            raise ValueError(f"{variant}: compared frames differ from existing benchmark")
    ids = sorted(frames.video_id.unique())
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(ids), size=(2000, len(ids)))
    board, boot_macro = [], {}
    for configuration, g in all_frames.groupby("configuration"):
        row = dict(configuration=configuration, videos=len(ids))
        boot = []
        for name in BEHAVIOURS:
            b = g[g.behaviour == name].set_index("video_id").reindex(ids)
            if b[["tp", "fp", "fn", "tn"]].isna().any().any():
                raise ValueError(f"{configuration}: missing held-out videos")
            c = b[["tp", "fp", "fn", "tn"]].to_numpy(float)
            pooled = c.sum(0)
            row[name + "_f1"] = rates(dict(zip(["tp", "fp", "fn", "tn"], pooled)))["f1"]
            sampled = c[draws].sum(1)
            den = 2 * sampled[:, 0] + sampled[:, 1] + sampled[:, 2]
            score = np.divide(2 * sampled[:, 0], den, out=np.zeros(len(den)), where=den > 0)
            if name != "rearing":
                boot.append(score)
        row["macro_f1"] = float(np.mean([row[b + "_f1"] for b in DEFAULT_CLASSES[1:]]))
        bootstrap = np.mean(boot, axis=0)
        row["macro_ci_low"], row["macro_ci_high"] = np.quantile(bootstrap, [.025, .975])
        boot_macro[configuration] = bootstrap
        board.append(row)
    board = pd.DataFrame(board).sort_values("macro_f1", ascending=False)
    board.to_csv(out / "comparison.csv", index=False)
    point = board.set_index("configuration").macro_f1
    delta = []
    for name, values in boot_macro.items():
        if name.startswith("native_"):
            continue
        diff = boot_macro["native_default"] - values
        lo, hi = np.quantile(diff, [.025, .975])
        delta.append(dict(reference=name, macro_f1_difference=point["native_default"]-point[name],
                          paired_ci_low=lo, paired_ci_high=hi, video_bootstrap_draws=len(draws)))
    pd.DataFrame(delta).to_csv(out / "paired_differences.csv", index=False)
    return board


def run_fst(out):
    from fst_calibration import FST_CFG, OUT, find_track, kappa, mask_from_bouts, read_labels
    from behaviour.fst import clip, score_immobility, total

    windows, bouts, raters = read_labels()
    sweep = pd.read_csv(OUT / "sweep.csv")
    videos = sorted(sweep.video_id.unique())
    keys = ["motion_threshold", "smooth_s", "min_immobile_s"]
    rows = []
    for vid in videos:
        # Reproduce the existing calibration with the held-out recording excluded.
        chosen = sweep[sweep.video_id != vid].groupby(keys).mean_kappa.mean().idxmax()
        cfg = load_config(FST_CFG)
        cfg.fst = replace(cfg.fst, motion_threshold=float(chosen[0]), smooth_s=float(chosen[1]), min_immobile_s=float(chosen[2]))
        start, duration = windows[vid]
        cfg.sessions = {vid: Session(start, duration, None)}
        track = pd.read_csv(find_track(vid))
        result = analyse_track(track, cfg, vid, assay="fst")
        result["metadata"].update(tracking_source="RATEL (OpenCV; existing FST cache)",
                                  calibration_videos=[v for v in videos if v != vid],
                                  fst=dict(motion_threshold=float(chosen[0]), smooth_s=float(chosen[1]), min_immobile_s=float(chosen[2])))
        save_analysis(result, out / "fst" / vid)
        df = result["trajectory"]
        w = df[(df.time_s >= start) & (df.time_s < start + duration)]
        t = w.time_s.to_numpy()
        _, _, predicted = score_immobility(t - start, w.motion_px.to_numpy(), w.area_px.to_numpy(), cfg, duration)
        values = result["metrics"].set_index("metric").value
        scores = [kappa(predicted, mask_from_bouts(t, bouts.get((vid, r), []))) for r in raters]
        manual = [total(clip(bouts.get((vid, r), []), start, start + duration)) for r in raters]
        rows.append(dict(video_id=vid, threshold=chosen[0], smooth_s=chosen[1], min_bout_s=chosen[2],
                         mean_rater_kappa=float(np.mean(scores)), tool_s=values.fst_immobile_s,
                         raters_mean_s=float(np.mean(manual)), latency_s=values.fst_immobility_latency_s,
                         calibration_excludes_test=True))
    pd.DataFrame(rows).to_csv(out / "fst_scores.csv", index=False)
    save_json(out / "fst_provenance.json", dict(sweep_sha256=file_hash(OUT / "sweep.csv"),
              design="leave one original video out; parameters chosen on the other four from the existing sweep",
              scope="five public mouse videos, four raters; immobility versus activity only"))


def table_md(table):
    values = [list(table.columns), ["---"] * len(table.columns), *table.astype(str).values.tolist()]
    return "\n".join("| " + " | ".join(row) + " |" for row in values)


def report(reference, out, dest, refresh_scores=False):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dest.mkdir(parents=True, exist_ok=True)
    board = comparisons(reference, out) if refresh_scores or not (out / "comparison.csv").exists() else pd.read_csv(out / "comparison.csv")
    agr = pd.read_csv(out / "metric_agreement.csv")
    movement = pd.read_csv(out / "tracker_agreement.csv", keep_default_na=False, na_values=[""])
    fst = pd.read_csv(out / "fst_scores.csv")
    timing = pd.read_csv(out / "timing.csv")
    positions = pd.read_csv(out / "tracker_positions.csv")
    summary = pd.read_csv(out / "summary_scores_vs_each_observer.csv")
    catalog_count = len(metric_catalog(["center", "periphery"]))
    compact = board[["configuration", "grooming_f1", "supported_rearing_f1", "unsupported_rearing_f1", "macro_f1"]].copy()
    for column in compact.columns[1:]:
        compact[column] = compact[column].map(lambda v: f"{v:.3f}")
    compact["configuration"] = compact.configuration.map(display_name)
    selected = [f"{b}_{m}" for b in BEHAVIOURS for m in ("time_s", "bouts", "latency_s")]
    accuracy = agr[(agr.variant == "default") & agr.metric.isin(selected)][["metric", "n", "bias", "mae", "ccc"]].round(3)
    bout_means = summary[summary.metric.str.endswith("_bouts")].groupby(["variant", "metric"])[["manual", "predicted"]].mean().round(2)
    move = movement[movement.metric.isin(["total_distance_m", "time_s", "entries", "immobile_time_s"])][
        ["metric", "zone", "n", "mae", "ccc"]].fillna("").round(4)
    native = board.set_index("configuration").loc["native_default"]
    tracking = movement[movement.metric == "total_distance_m"].iloc[0]
    inference = timing[timing.stage == "pose_features_and_inference"]
    train = timing[timing.stage == "training"]
    video_run = out / "full_video_run.json"
    video_note = ""
    if video_run.exists():
        run = json.loads(video_run.read_text())
        video_note = (f"A separate full-video CLI run processed {run['frames']:,} original frames "
                      f"({run['duration_s']:.2f} s of video) in {run['seconds']:.2f} s, using RATEL tracking "
                      "and supplied pose. This demonstration uses the all-development-video deployment "
                      "model and is not a held-out accuracy measurement.")
    text = f"""# RATEL 0.2.0

**{SOFTWARE_FULL_NAME}**

Built and measured on 2026-10-17. Reproduce with:

```powershell
.venv/Scripts/python scripts/benchmark_ratel.py
```

RATEL combines its OpenCV tracker, temporal pose classifier, shared
trajectory/zone calculations, named state and point-event summaries, and FST
immobility scoring. A two-zone catalogue contains **{catalog_count} metric/zone definitions**,
including tracking diagnostics and legacy import-only values. This is coverage
of this workspace's OFT/FST workflow; it is not a claim to reproduce every feature
of every commercial or research platform. Arbitrary additional scored states
and point events use the same summary engine.

Automatic named behaviours are grooming, supported/unsupported rearing, and
their combined rearing union. They require supplied thirteen-marker pose CSVs
and four arena landmarks. Video input can run the RATEL tracker without pose.
Sniffing, stretch-attend, freezing, jumping, defecation and urination need actual
annotations. FST swimming/climbing also need annotations. The analyzer preserves
these gaps in coverage.csv and retains manual observer labels.

## Held-out behaviour results

Twenty unique public mouse recordings, three human observers, five fixed folds
of sixteen training and four test videos. No original-video hash or recording
key crosses a fold. The classifier and fixed postprocessing settings were frozen
in benchmark_plan.json before fitting. Every evaluation recomputes features from
the source pose file through the saved-model deployment API. The all-video
deployment fit is separate and is never used for these scores.

All configurations use the same unambiguous frames at 2.5 Hz. F1 is pooled across
recordings; macro F1 averages the three named classes, excluding background and
the combined rearing union. Previous native package trials are limited component
experiments, not full-package rankings. These recordings already informed the
choice of the RATEL model; this remains exploratory development evaluation.

{table_md(compact)}

RATEL (filtered) macro F1: **{native.macro_f1:.3f}**, video-bootstrap 95% interval
{native.macro_ci_low:.3f}–{native.macro_ci_high:.3f} (2,000 resamples). Paired
differences against every existing configuration are in paired_differences.csv.
RATEL (raw) has no probability smoothing or minimum bout filter. RATEL (filtered) uses
0.2 s centered probability smoothing and removes named bouts shorter than 0.2 s.
These fixed settings are not a tuned or independently validated optimum.
Unusable-pose frames predict unknown; they remain in evaluation as missed named
behaviours and never disappear from the denominator.

## Agreement on actual measurements

Reference values use original continuous human intervals, clipped to the shared
scored window, with the mean observer as reference. Latency agreement includes
only paired finite observations; n is reported. Bias is native minus human.

{table_md(accuracy)}

Bout counts before and after the fixed filter:

{table_md(bout_means.reset_index())}

Grooming and bout counts still need review. A strong frame F1 does not establish
accurate bout counts or clinically meaningful rat behaviour classification.

## Movement and FST

Movement uses existing full-video RATEL (OpenCV) detections, rerun through the
analyzer, against DLC trajectories. It is agreement between two methods, not
independent position ground truth. All twenty videos are included; median of
per-video position errors is {positions.median_error_mm.median():.2f} mm.
Total-distance CCC is {tracking.ccc:.3f}, with MAE {tracking.mae:.3f} m.

{table_md(move)}

FST uses five labelled public mouse videos and four raters. Motion thresholds,
smoothing and minimum bout settings are selected on the other four videos from
the previously saved calibration sweep, then evaluated through the new analyzer.
Mean held-out frame kappa against raters is **{fst.mean_rater_kappa.mean():.3f}**.
Immobile-time MAE against the mean rater is {np.abs(fst.tool_s-fst.raters_mean_s).mean():.2f} s.
The supplied fixed default threshold is not represented as independently held out.

{table_md(fst.round(3))}

## Runtime and saved program

Python {platform.python_version()} on {platform.platform()}, up to eight classifier
threads. Five fold fits took {train.seconds.sum():.2f} s total. Recomputing pose
features and predicting {int(inference.frames.sum()):,} full-video frames took
{inference.seconds.sum():.2f} s total ({inference.frames.sum()/inference.seconds.sum():,.0f} frames/s).
These are single-run wall times. They exclude pose estimation, original video
tracking, disk report export and source-file hashing. OpenCV tracking was reused
from the earlier full-video run. Detailed stages are in timing.csv.

{video_note}

- Model: output/native_benchmark/models/ratel_pose.joblib, with a JSON model card.
- Per-recording reports: output/native_benchmark/recordings/VIDEO_ID/report.html.
- Metrics, frame counts, coverage, events, paired bootstrap comparisons and provenance
  stay in output/native_benchmark. Existing benchmark results were not overwritten.
- CLI help: `.venv/Scripts/python -m behaviour analyze --help`.
- User guide: reports/native_analyzer/workflow.md.
- TeX/PDF report: output/pdf/ratel_report.pdf; source bundle output/pdf/ratel_tex.zip.

Known limits: externally supplied pose is required for the RATEL named-behaviour model;
the present marker schema is specific; source pose geometry is not an independently
trained keypoint detector; only known video identities establish independence;
rat recordings and the actual laboratory camera remain unvalidated; motion FST
cannot separate swimming from climbing; pixel movement cannot establish freezing,
pellet counts or urination. Add scored target recordings, benchmark a fresh model
version on reserved animals/sessions, then compare with this frozen baseline.
"""
    (dest / "report.md").write_text(text, encoding="utf-8")
    presentation = board.copy()
    presentation.insert(0, "display_name", presentation.configuration.map(display_name))
    presentation.to_csv(dest / "comparison.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    plotted = board.iloc[::-1]
    colors = ["#118575" if c.startswith("native_") else "#7791ad" for c in plotted.configuration]
    ax.barh(plotted.configuration.map(display_name), plotted.macro_f1, color=colors)
    ax.set(xlim=(0, 1), xlabel="Macro F1 (three named behaviours)", title="Shared held-out videos · common 2.5 Hz grid")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(dest / "comparison.png", dpi=160)
    plt.close(fig)
    import html

    body = html.escape(text)
    report_html = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>RATEL benchmark</title>
<style>body{{font:16px system-ui;max-width:1150px;margin:40px auto;padding:0 24px;color:#183047}}img{{width:100%}}
table{{border-collapse:collapse;font-size:14px;width:100%}}td,th{{padding:7px;text-align:left;border-bottom:1px solid #ddd}}
pre{{white-space:pre-wrap;line-height:1.6;font:14px ui-monospace,monospace;background:#f4f7fa;padding:20px}}</style>
<h1>RATEL 0.2.0</h1><p>{SOFTWARE_FULL_NAME}</p>
<p>20 held-out public mouse recordings · 5 video folds · {catalog_count} catalogue definitions</p>
<img src="comparison.png" alt="Macro F1 by classifier configuration">
{compact.to_html(index=False, border=0)}<h2>Methods, measurements and limits</h2><pre>{body}</pre></html>'''
    (dest / "report.html").write_text(report_html, encoding="utf-8")
    return board


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", nargs="?", choices=["run", "report"], default="run")
    ap.add_argument("--reference", type=Path, default=ROOT / "output/open_field_benchmark")
    ap.add_argument("--output", type=Path, default=ROOT / "output/native_benchmark")
    ap.add_argument("--report-dir", type=Path, default=ROOT / "reports/native_analyzer")
    ap.add_argument("--train-stride", type=int, default=5)
    a = ap.parse_args()
    if a.train_stride < 1:
        ap.error("--train-stride must be positive")
    if a.output.resolve() == a.reference.resolve():
        ap.error("Use a separate output directory to preserve the reference benchmark")
    a.output.mkdir(parents=True, exist_ok=True)
    if a.command == "run":
        paths = [a.reference / p for p in ("recordings.json", "splits.json", "features.json", "manual_events.csv")]
        paths += [ROOT / "src/behaviour" / p for p in ("classifier.py", "native.py", "pose_features.py", "metrics.py", "fst.py",
                  "cli.py", "config.py", "trajectory.py", "ethogram.py", "compare.py", "adapters/deeplabcut.py",
                  "adapters/behaviours.py", "adapters/boris.py")]
        paths.append(Path(__file__))
        plan = dict(generated_utc=datetime.now(timezone.utc).isoformat(), seed=SEED, train_stride=a.train_stride,
                    feature_signature=feature_signature(), variants=VARIANTS,
                    estimator="HistGradientBoosting: 100 iterations, 15 leaves, depth 6, learning rate .08, balanced weights",
                    evaluation="five existing original-video folds; no hyperparameter search; common 2.5 Hz grid",
                    reference=str(a.reference.resolve()), sha256={str(p.resolve()): file_hash(p) for p in paths})
        save_json(a.output / "benchmark_plan.json", plan)
        metric_catalog(["center", "periphery"]).to_csv(a.output / "metric_catalog.csv", index=False)
        run_oft(a.reference, a.output, a.train_stride)
        run_fst(a.output)
    board = report(a.reference, a.output, a.report_dir, refresh_scores=a.command == "run")
    if a.command == "run":
        save_json(a.output / "completion.json", dict(completed_utc=datetime.now(timezone.utc).isoformat(),
                  videos=20, folds=5, fst_videos=5,
                  result_sha256={p.name: file_hash(p) for p in a.output.glob("*.csv")},
                  model_sha256={p.name: file_hash(p) for p in (a.output / "models").glob("*.joblib")}))
    print(board[["configuration", "macro_f1"]].round(4).to_string(index=False))
    print(f"Report: {(a.report_dir / 'report.html').resolve()}")


if __name__ == "__main__":
    main()
