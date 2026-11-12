"""RATEL - Rodent Activity Tracking and Ethogram Lab. ``behav`` remains an alias."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import pandas as pd

from . import __version__
from .config import load_config

VIDEO_EXT = {".avi", ".mp4", ".mov", ".mkv", ".wmv", ".mpg", ".mpeg", ".m4v"}


def _expand(paths: list[str], exts: set[str]) -> list[Path]:
    out = []
    for p in map(Path, paths):
        if p.is_dir():
            out += sorted(q for q in p.iterdir() if q.suffix.lower() in exts)
        else:
            out.append(p)
    return out


def cmd_preview(a):
    from .trackers.opencv_tracker import preview

    cfg = load_config(a.config)
    out = a.output or f"{Path(a.video).stem}_zones.png"
    preview(a.video, cfg.zones, out, a.time, cfg.mask_margin_px)
    print(f"Wrote {out}")


def cmd_track(a):
    from .trackers.opencv_tracker import TrackerParams, track

    cfg = load_config(a.config)
    params = TrackerParams.from_dict(cfg.tracker)
    bg = cv2.imread(a.background, cv2.IMREAD_GRAYSCALE) if a.background else None
    out_dir = Path(a.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    for video in _expand(a.videos, VIDEO_EXT):
        qc = out_dir / f"{video.stem}_qc.mp4" if a.qc else None
        df = track(video, cfg.zones, params, cfg.mask_margin_px, bg, qc)
        dst = out_dir / f"{video.stem}.csv"
        df.to_csv(dst, index=False)
        print(f"{video.name}: {df['x_px'].notna().mean():.1%} frames tracked -> {dst}")


def cmd_metrics(a):
    from .metrics import metrics_for_trajectory
    from .trajectory import load_trajectory

    cfg = load_config(a.config)
    cols = dict(c.split("=", 1) for c in a.columns) if a.columns else None
    parts = []
    for f in _expand(a.tracks, {".csv"}):
        df = load_trajectory(f, a.fps, cols)
        parts.append(metrics_for_trajectory(df, cfg, f.stem, a.tool))
    _write(pd.concat(parts, ignore_index=True), a.output)


def cmd_dlc(a):
    from .adapters import deeplabcut as dlc

    ids = {}
    if a.id_map:
        m = pd.read_csv(a.id_map, dtype=str)
        ids = {Path(f).stem: v for v, f in zip(m.iloc[:, 0], m.iloc[:, 1])}
    out_dir = Path(a.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in _expand(a.files, {".csv"}):
        vid = ids.get(f.stem) or dlc.video_id_from_filename(f.stem)
        df = dlc.trajectory(f, a.fps, a.bodypart, a.min_likelihood)
        df.to_csv(out_dir / f"{vid}.csv", index=False)
        print(f"{f.name} -> {vid}.csv ({df['x_px'].notna().mean():.1%} frames above likelihood {a.min_likelihood})")


def cmd_track_oft(a):
    from .trackers.free_oft import run

    run(a.video, load_config(a.config), a.engine, a.output, a.percentile, a.lower, a.upper, a.stride, a.lab_profile)


def cmd_manual_track(a):
    from .manual_tracking import run

    output = a.output or str(Path("output/tracks/manual") / f"{Path(a.video).stem}.csv")
    run(a.video, output, load_config(a.config) if a.config else None, a.speed)


def _pairs(items: list[str] | None, what: str) -> dict[str, str]:
    out = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"{what}: expected NAME=TARGET, got {item!r}")
        k, v = item.rsplit("=", 1)
        out[k.strip()] = v.strip()
    return out


def cmd_boris(a):
    from .adapters.boris import read_boris

    files = _expand(a.exports, {".csv", ".tsv", ".txt", ".xlsx", ".xls"})
    long = read_boris(files, load_config(a.config), _pairs(a.zone_map, "--zone-map"),
                      _pairs(a.fst_map, "--fst-map"), a.video_id)
    _write(long, a.output)


def cmd_fst(a):
    from .fst import fst_metrics
    from .trajectory import load_trajectory

    cfg = load_config(a.config)
    parts = [fst_metrics(load_trajectory(f, a.fps), cfg, f.stem, a.tool) for f in _expand(a.tracks, {".csv"})]
    _write(pd.concat(parts, ignore_index=True), a.output)


def cmd_compare(a):
    from .compare import compare

    long = pd.concat([pd.read_csv(f, dtype={"video_id": str, "zone": str}, keep_default_na=False,
                                  na_values=[""]) for f in a.metrics], ignore_index=True)
    long["zone"] = long["zone"].fillna("")
    res = compare(long, a.reference)
    _write(res, a.output)
    if not res.empty:
        cols = ["metric", "zone", "tool_a", "tool_b", "n", "bias", "bias_pct", "ccc", "icc_a1"]
        with pd.option_context("display.width", 160, "display.max_rows", 200, "display.precision", 3):
            print(res[cols].to_string(index=False))


def cmd_ethogram(a):
    from .adapters.behaviours import read_deepethogram, read_frames, read_intervals
    from .adapters.boris import read_events
    from .ethogram import behaviour_name, events_to_metrics, normalize_events

    cfg = load_config(a.config)
    cols = _pairs(a.columns, "--columns")
    mapping = _pairs(a.behaviour_map, "--behaviour-map")
    declared_mapping = {k.strip().lower(): v for k, v in mapping.items()}
    declared = {behaviour_name(declared_mapping.get(b.strip().lower(), b)): "state" for b in (a.behaviours or [])}
    declared.update({behaviour_name(declared_mapping.get(b.strip().lower(), b)): "point" for b in (a.points or [])})
    parts = []
    for path in _expand(a.exports, {".csv", ".tsv", ".txt", ".xlsx", ".xls", ".h5", ".hdf5"}):
        vid = a.video_id or path.stem
        if a.format == "boris":
            ev = normalize_events(read_events(path, include_points=True), mapping)
            if a.video_id:
                ev["video_id"] = a.video_id
        elif a.format == "intervals":
            ev = read_intervals(path, a.video_id, cols, mapping)
        elif a.format == "deepethogram":
            if not a.fps or not a.behaviours:
                raise SystemExit("DeepEthogram HDF5 needs --fps and --behaviours in stored class order")
            ev = read_deepethogram(path, a.fps, a.behaviours, a.dataset, a.threshold, vid)
            ev = normalize_events(ev, mapping)
        else:
            ev = read_frames(path, a.fps, a.behaviours, a.threshold, vid, cols,
                             "behaviour" if a.format == "labels" else None, mapping)
        parts.append(events_to_metrics(ev, cfg, a.tool, declared, vid))
    if not parts:
        raise SystemExit("No behaviour export files found")
    _write(pd.concat(parts, ignore_index=True), a.output)


def cmd_analyze(a):
    from .native import analyze

    result = analyze(a.input, load_config(a.config), a.output, a.input_kind, a.fps, a.video_id,
                     a.pose, a.model, a.events, a.ethogram, a.assay, a.qc, a.background,
                     a.smooth_s, a.min_bout_s, a.max_frames, a.config, a.events_format)
    meta = result["metadata"]
    print(f"{meta['video_id']}: {len(result['metrics'])} metric rows in {meta['seconds']:.2f}s")
    for warning in meta["warnings"]:
        print(warning)
    print(f"Report: {(Path(a.output) / 'report.html').resolve()}")


def cmd_train(a):
    from .classifier import train_manifest

    model = train_manifest(a.manifest, a.events, a.behaviours, a.output, a.stride)
    print(f"Trained on {model['training_frames']} frames from {len(model['trained_on'])} recordings -> {a.output}")
    print("This is a deployment fit. Benchmark on separate recordings before interpreting its accuracy.")


def cmd_capabilities(a):
    from .native import metric_catalog

    zones = load_config(a.config).zone_names() if a.config else ["center", "periphery"]
    table = metric_catalog(zones)
    if a.output:
        _write(table, a.output)
    else:
        print(table[["metric", "zone", "unit", "input"]].to_string(index=False))


def _write(df: pd.DataFrame, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Wrote {len(df)} rows to {path}")


def cmd_scorers(a):
    from . import scorers

    if a.scorer_action == "list":
        print(scorers.status().to_string(index=False))
    elif a.scorer_action == "launch":
        scorers.launch(a.tool, a.args)
    elif a.scorer_action == "compare":
        scorers.batch(a.manifest, a.output, a.reference)
    elif a.scorer_action == "install":
        import subprocess

        python_tools = [t for t in a.tools if scorers.TOOLS[t][1]]
        apps = [t for t in a.tools if not scorers.TOOLS[t][1]]
        root = scorers.ROOT
        installer_python = root / ".venv/Scripts/python.exe"
        if python_tools:
            subprocess.run([str(installer_python), str(root / "scripts/setup_behaviour_tools.py"),
                            "--tools", *python_tools, "--output", str(root / "output/scoring_tools")], check=True)
        if apps:
            subprocess.run([sys.executable, str(root / "scripts/install_scoring_apps.py"), *apps], check=True)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ratel", description=__doc__)
    ap.add_argument("--version", action="version", version=f"RATEL {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("analyze", help="one recording -> RATEL movement, behaviour, FST and coverage report")
    p.add_argument("input", help="video, canonical trajectory CSV, or DLC pose CSV")
    p.add_argument("-c", "--config", required=True)
    p.add_argument("--input-kind", choices=["auto", "video", "trajectory", "pose"], default="auto")
    p.add_argument("--fps", type=float, help="required for pose CSVs; original video frame rate")
    p.add_argument("--video-id", help="recording ID used in sessions and annotation tables")
    p.add_argument("--pose", help="pose CSV aligned to input video or trajectory for behaviour classification")
    p.add_argument("--model", help="RATEL saved .joblib pose behaviour model")
    p.add_argument("--events", help="canonical annotation CSV; retains manual observer/tool labels")
    p.add_argument("--events-format", choices=["intervals", "boris"], default="intervals")
    p.add_argument("--ethogram", help="behaviour,event_type CSV declaring the behaviours actually manually scored")
    p.add_argument("--assay", choices=["oft", "fst"], default="oft")
    p.add_argument("--smooth-s", type=float, default=0.2, help="offline probability smoothing window")
    p.add_argument("--min-bout-s", type=float, default=0.2, help="shorter named bouts become background")
    p.add_argument("--max-frames", type=int, help="clip surplus pose rows to the known video frame count")
    p.add_argument("--qc", action="store_true", help="save tracking overlay when input is a video")
    p.add_argument("--background", help="empty-arena image, useful for FST")
    p.add_argument("-o", "--output", default="output/native_analysis")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("train", help="fit and save the RATEL pose classifier on explicitly scored recordings")
    p.add_argument("manifest", help="CSV: video_id,pose_path,fps,duration_s[,start_s]")
    p.add_argument("--events", required=True, help="canonical state annotation CSV")
    p.add_argument("--behaviours", nargs="+", required=True, help="all behaviours scored across each training window")
    p.add_argument("--lab-profile", action="store_true", help="Tracktor coat-colour filter or FastTrack median background")
    p.add_argument("--stride", type=int, default=5, help="take every Nth training frame")
    p.add_argument("-o", "--output", default="output/models/ratel_pose.joblib")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("capabilities", help="list supported metric definitions and required inputs")
    p.add_argument("-c", "--config", help="use these zone names instead of center/periphery")
    p.add_argument("-o", "--output", help="optional metric catalogue CSV")
    p.set_defaults(func=cmd_capabilities)

    p = sub.add_parser("preview", help="draw config zones on a video frame (PNG)")
    p.add_argument("video")
    p.add_argument("-c", "--config", required=True)
    p.add_argument("-t", "--time", type=float, default=10.0, help="frame time in s")
    p.add_argument("-o", "--output")
    p.set_defaults(func=cmd_preview)

    p = sub.add_parser("track", help="run the RATEL (OpenCV) tracker -> per-frame CSV per video")
    p.add_argument("videos", nargs="+", help="video files or folders")
    p.add_argument("-c", "--config", required=True)
    p.add_argument("-o", "--output", default="output/tracks/opencv")
    p.add_argument("--qc", action="store_true", help="also write an annotated QC video")
    p.add_argument("--background", help="empty-arena image to use instead of the median background")
    p.set_defaults(func=cmd_track)

    p = sub.add_parser("manual-track", help="mark body-centre positions with a tablet pen or mouse")
    p.add_argument("video", help="source video (read-only)")
    p.add_argument("-c", "--config", help="use this config's per-video scoring window; otherwise the whole video")
    p.add_argument("--speed", type=float, choices=[0.1, 0.25, 0.5, 1.0], default=0.25)
    p.add_argument("-o", "--output", help="canonical CSV; existing manual tracks resume after source/window checks")
    p.set_defaults(func=cmd_manual_track)

    p = sub.add_parser("metrics", help="per-frame trajectories -> shared metrics (long CSV)")
    p.add_argument("tracks", nargs="+", help="trajectory CSVs or folders")
    p.add_argument("-c", "--config", required=True)
    p.add_argument("--tool", required=True, help="tool name to record, e.g. opencv, eztrack")
    p.add_argument("--fps", type=float, help="needed when the CSVs have frame but no time_s")
    p.add_argument("--columns", nargs="*", help="rename columns, e.g. x_px=X y_px=Y frame=Frame")
    p.add_argument("-o", "--output", required=True)
    p.set_defaults(func=cmd_metrics)

    p = sub.add_parser("dlc", help="DeepLabCut CSVs -> per-frame trajectory CSVs (then use `metrics`)")
    p.add_argument("files", nargs="+", help="DLC output CSVs or folders")
    p.add_argument("--fps", type=float, required=True)
    p.add_argument("--bodypart", default="bodycentre")
    p.add_argument("--min-likelihood", type=float, default=0.9)
    p.add_argument("--id-map", help="CSV: video_id,dlc_file (when DLC file names do not match the videos)")
    p.add_argument("-o", "--output", default="output/tracks/dlc")
    p.set_defaults(func=cmd_dlc)

    p = sub.add_parser("track-oft", help="run an open-field tracker on a video -> per-frame CSV")
    p.add_argument("video")
    p.add_argument("--engine", choices=["eztrack", "animapp", "tracktor", "fasttrack", "superanimal"], required=True)
    p.add_argument("-c", "--config", required=True)
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--lab-profile", action="store_true", help="Tracktor coat-colour filter or FastTrack median background")
    p.add_argument("--stride", type=int, help="SuperAnimal only: evaluate every Nth source frame (default 30)")
    p.add_argument("--percentile", type=float, default=99.7, help="ezTrack bright difference percentile")
    p.add_argument("--lower", type=int, nargs=3, default=(3, 7, 218), help="AnimApp lower HSV")
    p.add_argument("--upper", type=int, nargs=3, default=(42, 100, 255), help="AnimApp upper HSV")
    p.set_defaults(func=cmd_track_oft)

    p = sub.add_parser("boris", help="BORIS event exports (manual scoring) -> shared metrics (long CSV)")
    p.add_argument("exports", nargs="+", help="aggregated or tabular event exports (.csv/.tsv/.xlsx) or folders")
    p.add_argument("-c", "--config", required=True)
    p.add_argument("--zone-map", nargs="*", metavar="BEHAVIOUR=ZONE",
                   help='e.g. "Center zone=center"; behaviours named like a config zone map to it anyway')
    p.add_argument("--fst-map", nargs="*", metavar="BEHAVIOUR=STATE",
                   help="STATE is immobile, swim or climb; added to the defaults "
                        "(immobility/immobile/floating, swimming/swim, climbing/climb)")
    p.add_argument("--video-id", choices=["media", "observation"], default="media",
                   help="video id from the media file name (default) or the BORIS observation id")
    p.add_argument("-o", "--output", default="output/metrics/manual.csv")
    p.set_defaults(func=cmd_boris)

    p = sub.add_parser("ethogram", help="named behaviour events/predictions -> shared duration, bout and point metrics")
    p.add_argument("exports", nargs="+")
    p.add_argument("-c", "--config", required=True)
    p.add_argument("--tool", required=True, help="e.g. simba, labgym, deepethogram, dlcanalyzer or manual")
    p.add_argument("--format", choices=["intervals", "frames", "labels", "boris", "deepethogram"], default="intervals")
    p.add_argument("--fps", type=float)
    p.add_argument("--behaviours", nargs="*", help="scored states; frame format: source column names")
    p.add_argument("--points", nargs="*", help="scored point behaviours, e.g. defecation urination")
    p.add_argument("--behaviour-map", nargs="*", metavar="SOURCE=BEHAVIOUR")
    p.add_argument("--columns", nargs="*", metavar="CANONICAL=SOURCE")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--video-id", help="override a single export's recording ID")
    p.add_argument("--dataset", help="DeepEthogram HDF5 probability dataset, e.g. resnet18/P")
    p.add_argument("-o", "--output", required=True)
    p.set_defaults(func=cmd_ethogram)

    p = sub.add_parser("fst", help="forced swim immobility from `track` CSVs (motion_px) -> shared metrics")
    p.add_argument("tracks", nargs="+", help="per-frame CSVs from `behav track`, or folders")
    p.add_argument("-c", "--config", required=True)
    p.add_argument("--tool", default="opencv")
    p.add_argument("--fps", type=float, help="needed when the CSVs have frame but no time_s")
    p.add_argument("-o", "--output", default="output/metrics/opencv_fst.csv")
    p.set_defaults(func=cmd_fst)

    p = sub.add_parser("compare", help="agreement statistics between tools")
    p.add_argument("metrics", nargs="+", help="long metric CSVs from `metrics` / `boris` / `fst`")
    p.add_argument("--reference", help="compare every tool to this one (e.g. manual)")
    p.add_argument("-o", "--output", default="output/agreement.csv")
    p.set_defaults(func=cmd_compare)

    from .scorers import TOOLS
    p = sub.add_parser("scorers", help="install, inspect, launch or compare scoring applications")
    actions = p.add_subparsers(dest="scorer_action", required=True)
    q = actions.add_parser("list", help="inspect local installations and runtime requirements")
    q.set_defaults(func=cmd_scorers)
    q = actions.add_parser("install", help="install selected tools in project-local environments")
    q.add_argument("tools", choices=sorted(TOOLS), nargs="+")
    q.set_defaults(func=cmd_scorers)
    q = actions.add_parser("launch", help="open an installed scoring application")
    q.add_argument("tool", choices=sorted(TOOLS))
    q.add_argument("args", nargs=argparse.REMAINDER)
    q.set_defaults(func=cmd_scorers)
    q = actions.add_parser("compare", help="import exports listed in a CSV manifest and compare shared metrics")
    q.add_argument("manifest")
    q.add_argument("-o", "--output", required=True)
    q.add_argument("--reference", help="run_id to compare against")
    q.set_defaults(func=cmd_scorers)

    a = ap.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    sys.exit(main())
