"""RATEL's integrated tracker, pose classifier and scored-event pipeline.

Unmeasured behaviours are listed in coverage.csv, never represented as zeros.
Saved frame labels and events make every reported behaviour bout reviewable.
"""

from __future__ import annotations

import hashlib
import html
import json
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .branding import SOFTWARE_NAME, SOFTWARE_FULL_NAME, display_name
from .adapters.behaviours import frame_events
from .config import Config, Session
from .ethogram import EVENT_COLUMNS, behaviour_rows, events_to_metrics, normalize_events
from .fst import clip, fst_metrics, fst_rows, runs
from .metrics import LONG_COLUMNS, METRICS, compute_metrics, to_long
from .trajectory import clean

STATES = ["grooming", "supported_rearing", "unsupported_rearing", "rearing",
          "sniffing", "stretch_attend", "freezing"]
POINTS = ["jumping", "defecation", "urination"]
STATE_SUMMARIES = {"time_s": "s", "time_pct": "%", "bouts": "count", "mean_bout_s": "s",
                   "median_bout_s": "s", "max_bout_s": "s", "latency_s": "s", "frequency_per_min": "1/min"}
POINT_SUMMARIES = {"count": "count", "latency_s": "s", "frequency_per_min": "1/min"}
EXTRA_METRICS = {
    "max_speed_m_s": ("m/s", "behaviour", "Maximum finite step speed of the cleaned trajectory."),
    "median_speed_m_s": ("m/s", "behaviour", "Median finite step speed of the cleaned trajectory."),
    "p95_speed_m_s": ("m/s", "behaviour", "95th percentile of finite cleaned step speeds."),
    "lost_detection_bouts": ("count", "quality", "Runs of missing raw detections, including leading/trailing loss."),
    "behaviour_scored_fraction": ("", "quality", "Fraction of analysis frames with sufficient pose for behaviour proposals."),
    "fst_motion_valid_fraction": ("", "quality", "Fraction of analysis frames with valid FST motion and area measurements."),
}


def file_hash(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def metric_catalog(zone_names=(), states=None, points=None) -> pd.DataFrame:
    rows = []
    for name, (unit, kind, definition) in {**METRICS, **EXTRA_METRICS}.items():
        if name in {"time_s", "time_pct", "distance_m", "entries", "latency_s"}:
            zones, source = list(zone_names), "trajectory"
        else:
            zones = [""]
            source = "fst" if name.startswith("fst_") else "trajectory"
        if name == "losses":
            source = "vendor_export"
        if name == "behaviour_scored_fraction":
            source = "pose_model"
        if name in {"fst_swim_s", "fst_climb_s", "fst_swim_bins", "fst_climb_bins", "fst_uncoded_s"}:
            source = "fst_annotations"
        rows += [dict(metric=name, zone=z, unit=unit, kind=kind, input=source, definition=definition) for z in zones]
    for names, summaries, source in [(states if states is not None else STATES, STATE_SUMMARIES, "state"),
                                      (points if points is not None else POINTS, POINT_SUMMARIES, "point")]:
        for name in names:
            for suffix, unit in summaries.items():
                rows.append(dict(metric=f"{name}_{suffix}", zone="", unit=unit, kind="behaviour", input=source,
                                 definition=f"{suffix} for explicitly scored {name}; observation window is required."))
    return pd.DataFrame(rows)


def complete_track(df: pd.DataFrame, fps: float | None = None) -> tuple[pd.DataFrame, float]:
    """Restore absent frame rows before cleaning, so gaps cannot become movement."""
    if not {"time_s", "x_px", "y_px"}.issubset(df) or len(df) < 2:
        raise ValueError("A trajectory needs time_s, x_px, y_px and at least two frames")
    t = df.time_s.to_numpy(float)
    if not np.isfinite(t).all() or t[0] < 0 or (np.diff(t) <= 0).any():
        raise ValueError("Trajectory times must be finite, nonnegative and strictly increasing")
    fps = float(fps) if fps is not None else 1 / float(np.median(np.diff(t)))
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("A positive fps is required")
    frame = np.rint(t * fps).astype(int)
    if not np.allclose(t, frame / fps, atol=1e-5, rtol=0) or (np.diff(frame) <= 0).any():
        raise ValueError("Trajectory timestamps must follow a regular video clock; pass its original --fps")
    if "frame" in df and not np.array_equal(df.frame.to_numpy(), frame):
        raise ValueError("Frame indices do not match time_s and fps")
    out = df.copy()
    out["frame"] = frame
    out = out.set_index("frame").reindex(np.arange(frame[-1] + 1)).rename_axis("frame").reset_index()
    out["time_s"] = out.frame / fps
    xy = out[["x_px", "y_px"]].to_numpy(float)
    out.loc[~np.isfinite(xy).all(1), ["x_px", "y_px"]] = np.nan
    return out, fps


def prediction_events(predictions, classes, video_id, fps, smooth_s=0.2, min_bout_s=0.2):
    from .classifier import prediction_labels

    labels = prediction_labels(predictions, classes, smooth_s, min_bout_s)
    table = predictions.copy()
    table["label"] = np.array([classes[j] if j >= 0 else "unknown" for j in labels])
    ev = frame_events(table, video_id, fps, classes[1:], label_column="label")
    return table, ev


def _add_rearing(events, declared):
    if {"supported_rearing", "unsupported_rearing"}.issubset(declared) and "rearing" not in declared:
        rear = events[events.behaviour.isin(["supported_rearing", "unsupported_rearing"])].copy()
        rear["behaviour"] = "rearing"
        events = pd.concat([events, rear], ignore_index=True)
        declared = {**declared, "rearing": "state"}
    return events, declared


def analyse_track(df, cfg: Config, video_id: str, fps=None, predictions=None, classes=None,
                  events=None, declared=None, assay="oft", smooth_s=0.2, min_bout_s=0.2):
    if assay not in {"oft", "fst"}:
        raise ValueError("Assay must be oft or fst")
    if not np.isfinite(cfg.px_per_m) or cfg.px_per_m <= 0:
        raise ValueError("Video scale px_per_m must be positive")
    df, fps = complete_track(df, fps)
    session = cfg.session(video_id)
    if not np.isfinite(session.start_s) or (session.duration_s is not None and
            (not np.isfinite(session.duration_s) or session.duration_s <= 0)):
        raise ValueError("The scoring window needs a finite start and positive finite duration")
    end = float(df.time_s.iloc[-1] + 1 / fps)
    if session.duration_s is not None:
        end = min(end, session.start_s + session.duration_s)
    if end <= session.start_s or session.start_s < 0:
        raise ValueError("The analysis window has no video frames")
    duration = end - session.start_s
    cfg = replace(cfg, sessions={video_id: Session(session.start_s, duration, session.start_zone)})
    cr = clean(df, cfg.px_per_m, session.start_s, duration,
               cfg.max_speed_m_s, cfg.max_gap_s, cfg.smooth_s)
    if len(cr.t) < 2:
        raise ValueError("The analysis window needs at least two video frames")
    rows = compute_metrics(cr, cfg, duration)
    speed = np.hypot(np.diff(cr.x), np.diff(cr.y)) / np.diff(cr.t) / cfg.px_per_m
    speed = speed[np.isfinite(speed)]
    extra = {"max_speed_m_s": speed.max() if len(speed) else np.nan,
             "median_speed_m_s": np.median(speed) if len(speed) else np.nan,
             "p95_speed_m_s": np.quantile(speed, .95) if len(speed) else np.nan,
             "lost_detection_bouts": len(runs(~cr.raw_valid))}
    rows += [dict(metric=k, zone="", value=float(v), kind=EXTRA_METRICS[k][1]) for k, v in extra.items()]
    metrics = [to_long(rows, video_id, "native")]
    warnings, all_events = [], []
    frame_table = None
    automatic = {}
    if predictions is not None:
        if not classes:
            raise ValueError("Prediction class names are required")
        times = predictions.time_s.to_numpy(float)
        if len(times) != len(df) or not np.allclose(times, df.time_s, atol=1e-6, rtol=0):
            raise ValueError("Pose predictions must match the complete trajectory clock and frame count")
        frame_table, ev = prediction_events(predictions, classes, video_id, fps, smooth_s, min_bout_s)
        automatic = dict.fromkeys(classes[1:], "state")
        ev, automatic = _add_rearing(ev, automatic)
        metrics.append(events_to_metrics(ev, cfg, "native", automatic, video_id))
        ev["source"] = "model"
        all_events.append(ev)
        in_window = (times >= session.start_s) & (times < end)
        known = frame_table.label.ne("unknown").to_numpy()[in_window]
        coverage = float(known.mean())
        metrics.append(to_long([dict(metric="behaviour_scored_fraction", zone="", value=coverage, kind="quality")],
                               video_id, "native"))
        if coverage < 1:
            warnings.append(f"Pose is insufficient in {1 - coverage:.1%} of analysis frames; named behaviour totals are partial.")
    if events is not None or declared:
        ev = normalize_events(events if events is not None else pd.DataFrame(columns=EVENT_COLUMNS))
        ev = ev[ev.video_id == video_id].copy()
        declared = dict(declared or {})
        if assay == "fst":
            from .adapters.boris import DEFAULT_FST_MAP

            aliases = {name: f"fst_{state}" for name, state in DEFAULT_FST_MAP.items()}
            ev["behaviour"] = ev.behaviour.replace(aliases)
            declared = {aliases.get(k, k): v for k, v in declared.items()}
        for name, group in ev.groupby("behaviour"):
            kinds = group.event_type.unique()
            if len(kinds) != 1:
                raise ValueError(f"{name}: annotations mix state and point events")
            if name in declared and declared[name] != kinds[0]:
                raise ValueError(f"{name}: annotation type differs from the declared ethogram")
            declared[name] = kinds[0]
        if assay == "fst" and any(declared.get(f"fst_{name}", "state") != "state"
                                   for name in ("immobile", "swim", "climb")):
            raise ValueError("FST immobile/swim/climb annotations must be state events")
        ev, declared = _add_rearing(ev, declared)
        metrics.append(events_to_metrics(ev, cfg, "native_manual", declared, video_id))
        if assay == "fst":
            groups = ev.groupby("observer") if "observer" in ev else [("", ev)]
            if ev.empty:
                groups = [("", ev)]
            for observer, group in groups:
                states = {name: clip(list(zip(group.loc[group.behaviour == f"fst_{name}", "start"],
                                              group.loc[group.behaviour == f"fst_{name}", "stop"])),
                                     session.start_s, end)
                          for name in ("immobile", "swim", "climb") if f"fst_{name}" in declared}
                if states:
                    from .fst import total

                    coded = total([iv for intervals in states.values() for iv in intervals])
                    if sum(total(iv) for iv in states.values()) > coded + 1e-6:
                        raise ValueError("Manual FST immobile/swim/climb states must not overlap")
                    values = fst_rows(states, duration, cfg)
                    values.append(dict(metric="fst_uncoded_s", zone="", value=max(duration - coded, 0)))
                    for row in values:
                        row["kind"] = METRICS[row["metric"]][1]
                    tool = f"native_manual_{observer}" if observer else "native_manual"
                    metrics.append(to_long(values, video_id, tool))
        ev["source"] = "annotation"
        all_events.append(ev)
    if assay == "fst":
        fst_track = df.copy()
        if {"area_px", "motion_px"}.issubset(fst_track):
            valid = (np.isfinite(fst_track[["x_px", "y_px", "area_px", "motion_px"]]).all(axis=1)
                     & (fst_track.area_px > 0) & (fst_track.motion_px >= 0))
            fst_track.loc[~valid, "motion_px"] = np.nan
        fst_values = fst_metrics(fst_track, cfg, video_id, "native")
        fst_values["metric"] = fst_values.metric.replace({"tracked_fraction": "fst_motion_valid_fraction"})
        metrics.append(fst_values)
        warnings.append("Motion-based FST scoring distinguishes immobility from activity; swim/climb require annotations.")
    metrics = pd.concat(metrics, ignore_index=True).drop_duplicates(["video_id", "tool", "metric", "zone"])
    measured = metrics.groupby(["metric", "zone"], dropna=False).tool.agg(lambda s: ",".join(sorted(set(s))))
    states = sorted(set(STATES) | set(automatic) | {k for k, v in (declared or {}).items() if v == "state"})
    points = sorted(set(POINTS) | {k for k, v in (declared or {}).items() if v == "point"})
    catalog = metric_catalog(cfg.zone_names(), states, points)
    coverage_rows = []
    for row in catalog.to_dict("records"):
        tools = measured.get((row["metric"], row["zone"]), "")
        if tools:
            status = "measured"
        elif row["input"].startswith("fst") and assay != "fst":
            status = "other_assay"
        elif row["input"] == "vendor_export":
            status = "vendor_export_only"
        elif row["input"] == "pose_model":
            status = "requires_pose_model"
        elif row["input"] == "state" and any(row["metric"].startswith(b + "_") for b in STATES[:4]):
            status = "requires_pose_model_or_annotations"
        else:
            status = "requires_annotations"
        coverage_rows.append({**row, "status": status, "measured_by": tools})
    return dict(metrics=metrics[LONG_COLUMNS], coverage=pd.DataFrame(coverage_rows), trajectory=df,
                predictions=frame_table,
                events=pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame(columns=[*EVENT_COLUMNS, "source"]),
                metadata=dict(video_id=video_id, version=__version__, assay=assay, fps=fps,
                              start_s=session.start_s, end_s=end, duration_s=duration,
                              smooth_s=smooth_s, min_bout_s=min_bout_s, warnings=warnings,
                              parameters=dict(px_per_m=cfg.px_per_m, max_speed_m_s=cfg.max_speed_m_s,
                                              max_gap_s=cfg.max_gap_s, trajectory_smooth_s=cfg.smooth_s,
                                              immobile_speed_m_s=cfg.immobile_speed_m_s,
                                              immobile_min_s=cfg.immobile_min_s, hysteresis_mode=cfg.hysteresis_mode,
                                              fst=asdict(cfg.fst)),
                              tracking_source="provided trajectory", generated_utc=datetime.now(timezone.utc).isoformat()))


def save_analysis(result: dict, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for key in ("metrics", "coverage", "trajectory", "events", "predictions"):
        if result[key] is not None:
            result[key].to_csv(output / f"{key}.csv", index=False)
        elif (output / f"{key}.csv").exists():
            (output / f"{key}.csv").unlink()
    metadata = result["metadata"]
    metadata.setdefault("software", SOFTWARE_NAME)
    metadata.setdefault("software_full_name", SOFTWARE_FULL_NAME)
    (output / "analysis.json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    write_analysis_report(result["metrics"], result["coverage"], metadata, output)


def write_analysis_report(metrics, coverage, metadata, output):
    """Refresh a report's presentation without rewriting its saved measurements."""
    output = Path(output)
    values = metrics.copy()
    values["tool"] = values.tool.map(display_name)
    values["value"] = values.value.map(lambda x: f"{x:.5g}" if np.isfinite(x) else "undefined")
    warnings = "".join(f"<li>{html.escape(s)}</li>" for s in metadata["warnings"])
    measured = int(coverage.status.eq("measured").sum())
    coverage = coverage[["metric", "zone", "unit", "status", "measured_by"]].copy()
    coverage["measured_by"] = coverage.measured_by.fillna("").map(
        lambda s: ", ".join(display_name(v) for v in s.split(",") if v))
    tracking_source = metadata["tracking_source"].replace("our OpenCV tracker", "RATEL (OpenCV)")
    body = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<title>{html.escape(metadata['video_id'])} — RATEL analysis</title>
<style>body{{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 24px;color:#183047;background:#f5f7fa}}
table{{border-collapse:collapse;background:white;width:100%;font-size:14px}}th,td{{text-align:left;padding:8px;border-bottom:1px solid #dbe3ea}}
th{{background:#e6edf3}}h1,h2{{color:#123e59}}.note{{padding:16px;background:#fff3dc;border-radius:8px}}a{{color:#075e87}}
</style><h1>RATEL · {html.escape(metadata['video_id'])}</h1>
<p>{SOFTWARE_FULL_NAME}</p>
<p>RATEL {html.escape(__version__)} · {html.escape(metadata['assay'].upper())} ·
{metadata['start_s']:.3f}–{metadata['end_s']:.3f} s · {measured} catalog measurements available</p>
<p>Movement source: {html.escape(tracking_source)}.
Model and manual results retain separate tool labels. Undefined latency means no observed event.
Behaviours requiring annotations are listed in coverage and have no numeric result.</p>
<div class="note"><ul>{warnings or '<li>Review tracking and behaviour proposals against the recording.</li>'}</ul></div>
<p><a href="metrics.csv">Metrics CSV</a> · <a href="events.csv">Events CSV</a> ·
<a href="coverage.csv">Coverage CSV</a> · <a href="analysis.json">Provenance</a></p>
<h2>Measurements</h2>{values.to_html(index=False, escape=True, border=0)}
<h2>Coverage</h2>{coverage.to_html(index=False, escape=True, border=0)}
</html>"""
    (output / "report.html").write_text(body, encoding="utf-8")


def analyze(source, cfg: Config, output, input_kind="auto", fps=None, video_id=None, pose_path=None,
            model_path=None, events_path=None, ethogram_path=None, assay="oft", qc=False,
            background_path=None, smooth_s=0.2, min_bout_s=0.2, max_frames=None, config_path=None,
            events_format="intervals"):
    """Analyse one recording; the CLI and benchmark share this implementation."""
    import cv2
    from .adapters.deeplabcut import trajectory
    from .trajectory import load_trajectory

    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    video_id = video_id or source.stem
    if input_kind == "auto":
        if source.suffix.lower() in {".csv", ".tsv"}:
            header = pd.read_csv(source, nrows=0).columns
            input_kind = "trajectory" if {"x_px", "y_px"}.issubset(header) else "pose"
        else:
            input_kind = "video"
    if input_kind == "video":
        from .trackers.opencv_tracker import TrackerParams, track

        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise ValueError(f"Could not open {source}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        cv2.setNumThreads(2)
        bg = cv2.imread(str(background_path), cv2.IMREAD_GRAYSCALE) if background_path else None
        if background_path and bg is None:
            raise ValueError(f"Could not read background {background_path}")
        df = track(source, cfg.zones, TrackerParams.from_dict(cfg.tracker), cfg.mask_margin_px, bg,
                   output / "tracking_qc.mp4" if qc else None)
    elif input_kind == "pose":
        if fps is None or not np.isfinite(fps) or fps <= 0:
            raise ValueError("Pose CSV input requires a positive --fps")
        pose_path = source
        df = trajectory(source, fps)
    elif input_kind == "trajectory":
        df = load_trajectory(source, fps)
    else:
        raise ValueError("Input kind must be video, trajectory or pose")
    if max_frames is not None:
        if max_frames < 2 or max_frames > len(df):
            raise ValueError("--max-frames must be between two and the input frame count")
        df = df.iloc[:max_frames]
    df, fps = complete_track(df, fps)
    predictions, model = None, None
    if model_path:
        from .classifier import load_model, predict_pose

        if pose_path is None:
            raise ValueError("A behaviour model requires --pose or --input-kind pose")
        model = load_model(model_path)
        predictions = predict_pose(pose_path, fps, model, len(df))
    ev = None
    if events_path:
        if events_format == "boris":
            from .adapters.boris import read_events

            ev = normalize_events(read_events(events_path, include_points=True))
        elif events_format == "intervals":
            from .adapters.behaviours import read_intervals

            ev = read_intervals(events_path)
        else:
            raise ValueError("Annotation format must be intervals or boris")
    if ev is not None and len(ev) and video_id not in set(ev.video_id):
        raise ValueError(f"Annotations contain no video_id {video_id!r}")
    declared = None
    if ethogram_path:
        from .ethogram import behaviour_name

        ethogram = pd.read_csv(ethogram_path)
        if not {"behaviour", "event_type"}.issubset(ethogram):
            raise ValueError("Ethogram needs behaviour,event_type columns")
        names = [behaviour_name(b) for b in ethogram.behaviour]
        if len(set(names)) != len(names):
            raise ValueError("Ethogram behaviours must have unique normalized names")
        declared = dict(zip(names, ethogram.event_type))
        if not set(declared.values()).issubset({"state", "point"}):
            raise ValueError("Ethogram event_type must be state or point")
        if ev is None:
            raise ValueError("--ethogram declares actually scored annotations and requires --events")
    result = analyse_track(df, cfg, video_id, fps, predictions, model["classes"] if model else None,
                           ev, declared, assay, smooth_s, min_bout_s)
    inputs = {"source": source, "pose": pose_path, "model": model_path,
              "events": events_path, "ethogram": ethogram_path, "background": background_path,
              "config": config_path}
    result["metadata"].update(
        input_kind=input_kind, tracking_source="RATEL (OpenCV)" if input_kind == "video" else
        "supplied pose centroid" if input_kind == "pose" else "provided trajectory",
        seconds=time.perf_counter() - started,
        inputs={k: dict(path=str(Path(p).resolve()), sha256=file_hash(p))
                for k, p in inputs.items() if p is not None},
        model={k: model[k] for k in ("classes", "trained_on", "training_frames", "feature_signature")} if model else None)
    save_analysis(result, output)
    return result
