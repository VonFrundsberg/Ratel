"""Compare full TailTrail batch runs with the pilot's DLC and RATEL tracks.

First run tailtrail_batch.py on the four OFT videos with image calibration into
output/tailtrail_auto_benchmark/image, and with --landmarks-map into its sibling
landmark directory. Then run this script. --partial reports only paired,
completed recordings while a benchmark is still running.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from behaviour.adapters.deeplabcut import landmarks
from behaviour.adapters.tailtrail import read_data_xlsx, read_settings
from behaviour.compare import compare
from behaviour.config import load_config
from behaviour.metrics import metrics_for_trajectory
from behaviour.trajectory import load_trajectory
from behaviour.zones import PolygonZone

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "reports/zenodo_pilot"
DATA = ROOT / "videos/zenodo_3608658"
TOOLS = {"tailtrail_image": "TailTrail / image", "tailtrail_landmark": "TailTrail / landmarks",
         "ratel": "RATEL", "dlc_geometry_only": "DLC / image geometry"}


def sha256(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def completed_cases(folder: Path) -> dict:
    status = pd.read_csv(folder / "batch_status.csv").fillna("")
    cases = {}
    for row in status.to_dict("records"):
        if row["state"] not in ("done", "skipped"):
            continue
        path = Path(row["result_dir"])
        result = json.loads((path / "result.json").read_text(encoding="utf-8"))
        calibration = json.loads((path / "calibration.json").read_text(encoding="utf-8"))
        if not calibration["accepted"] or calibration["job_id"] != result["job_id"]:
            raise ValueError(f"Incomplete calibration: {path}")
        vid = Path(result["video"]).stem
        if vid in cases:
            raise ValueError(f"Duplicate recording: {vid}")
        cases[vid] = dict(folder=path, result=result, calibration=calibration)
    return cases


def config_for(vid: str, settings: dict, result: dict):
    cfg = copy.deepcopy(load_config(PILOT / "oft/configs" / f"{vid}.toml"))
    cfg.sessions = {}
    cfg.start_s = result["start_frame"] / result["fps"]
    cfg.duration_s = result["processed_frames"] / result["fps"]
    cfg.px_per_m = float(settings["scale"])
    cfg.zones = [PolygonZone(roi["name"].lower(), float(roi["hysteresis"]) * cfg.px_per_m,
                             points=roi["contour"]) for roi in settings["rois"]]
    return cfg


def zone_masks(settings: dict, width: int, height: int) -> dict:
    masks, assigned = {}, np.zeros((height, width), bool)
    for roi in settings["rois"]:
        mask = np.zeros((height, width), np.uint8)
        cv2.fillPoly(mask, [np.asarray(roi["contour"], np.int32)], 1)
        masks[roi["name"].lower()] = (mask > 0) & ~assigned
        assigned |= mask > 0
    circle = np.zeros((height, width), np.uint8)
    cv2.circle(circle, (width // 2, height // 2), int(settings["mask_radius"]), 1, -1)
    masks["mask"] = circle > 0
    return masks


def iou(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.count_nonzero(a & b) / np.count_nonzero(a | b))


def markdown_table(columns: list[str], rows: list[list]) -> str:
    return "\n".join(["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
                     + ["| " + " | ".join(str(v) for v in row) + " |" for row in rows])


def number(value, digits=2):
    return f"{value:.{digits}f}" if np.isfinite(value) else "—"


def plot_accuracy(primary: pd.DataFrame, videos: list[str], output: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))
    colors = {"dlc": "#252525", "ratel": "#397cac", "tailtrail_image": "#a05eb5", "tailtrail_landmark": "#acb2bb"}
    names = {"dlc": "DLC reference", **TOOLS}
    for ax, (metric, zone, label) in zip(axes, (("total_distance_m", "", "Distance (m)"),
                                               ("time_s", "center", "Centre time (s)"),
                                               ("entries", "center", "Centre entries"))):
        data = primary[(primary.metric == metric) & (primary.zone == zone)]
        x = np.arange(len(videos))
        for offset, tool in enumerate(colors):
            values = data[data.tool == tool].set_index("video_id").value.reindex(videos)
            ax.bar(x + (offset - 1.5) * .20, values, width=.19, color=colors[tool], label=names[tool])
        ax.set_xticks(x, videos)
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, .90))
    fig.savefig(output / "accuracy.png", dpi=180)
    fig.savefig(output / "accuracy.svg")
    plt.close(fig)


def analyse(batch_root: Path, output: Path, partial: bool):
    cases = {mode: completed_cases(batch_root / mode) for mode in ("image", "landmark")}
    mapping = pd.read_csv(DATA / "oft_dlc_map.csv").set_index("ID").dlc_file.to_dict()
    videos = sorted(set(cases["image"]) & set(cases["landmark"]) & set(mapping))
    if not videos or (not partial and set(videos) != set(mapping)):
        missing = {mode: sorted(set(mapping) - set(data)) for mode, data in cases.items()}
        raise ValueError(f"Benchmark is incomplete: {missing}. Use --partial for an interim report.")
    output.mkdir(parents=True, exist_ok=True)
    base = load_config(PILOT / "oft/configs" / f"{videos[0]}.toml")
    summaries = {mode: read_data_xlsx(batch_root / mode / "data.xlsx", base) for mode in cases}
    primary, matched, runs, geometry, provenance = [], [], [], [], []
    for vid in videos:
        result = cases["image"][vid]["result"]
        keys = ("start_frame", "end_frame_exclusive", "processed_frames", "fps")
        if any(result[k] != cases["landmark"][vid]["result"][k] for k in keys):
            raise ValueError(f"Analysis windows differ between arms for {vid}")
        jobs = {mode: json.loads((batch_root / mode / "jobs" / (cases[mode][vid]["result"]["job_id"] + ".json"))
                                .read_text(encoding="utf-8")) for mode in cases}
        for key in ("exe_sha256", "worker_sha256", "settings_sha256", "threads", "video_size", "video_mtime_ns"):
            if jobs["image"][key] != jobs["landmark"][key]:
                raise ValueError(f"Different engine, template or input between arms for {vid}: {key}")
        for key in ("arena_side_m", "center_scale", "periphery_scale", "hysteresis_m", "mask_margin_px", "code_sha256"):
            if jobs["image"]["calibration"][key] != jobs["landmark"]["calibration"][key]:
                raise ValueError(f"Different calibration protocol between arms for {vid}: {key}")
        if cases["image"][vid]["calibration"]["source"] != "image" or cases["landmark"][vid]["calibration"]["source"] != "dlc":
            raise ValueError(f"Unexpected calibration source for {vid}")
        original = read_settings(PILOT / "tailtrail/settings" / vid)
        fixed_cfg = config_for(vid, original, result)
        dlc_path = PILOT / "oft/tracks/dlc" / f"{vid}.csv"
        ratel_path = PILOT / "oft/tracks/opencv" / f"{vid}.csv"
        dlc, ratel = load_trajectory(dlc_path), load_trajectory(ratel_path)
        reference = metrics_for_trajectory(dlc, fixed_cfg, vid, "dlc")
        primary += [reference, metrics_for_trajectory(ratel, fixed_cfg, vid, "ratel")]
        truth = landmarks(DATA / "dlc" / mapping[vid], ["tl", "tr", "br", "bl"])
        reference_masks = zone_masks(original, result["width"], result["height"])
        for mode in cases:
            case = cases[mode][vid]
            settings = read_settings(case["folder"] / "settings.pickle")
            cfg = config_for(vid, settings, case["result"])
            native = summaries[mode].loc[summaries[mode].video_id == vid].copy()
            native["tool"] = "tailtrail_" + mode
            if native.empty:
                raise ValueError(f"Missing spreadsheet summary for {mode}/{vid}")
            primary.append(native)
            matched_reference = metrics_for_trajectory(dlc, cfg, vid, "dlc")
            agreement = compare(pd.concat([native, matched_reference]), reference="dlc")
            agreement.insert(0, "evaluation", "geometry_matched")
            matched.append(agreement.assign(video_id=vid))
            if mode == "image":
                sensitivity = matched_reference.copy()
                sensitivity["tool"] = "dlc_geometry_only"
                primary.append(sensitivity)
            q = native[native.kind == "quality"].set_index("metric").value
            run = case["result"]
            runs.append(dict(video_id=vid, calibration=mode, duration_s=q["duration_s"],
                             processed_frames=run["processed_frames"], losses=q["losses"],
                             outside_rois_s=q["unassigned_time_s"],
                             outside_rois_pct=100 * q["unassigned_time_s"] / q["duration_s"],
                             elapsed_s=run["elapsed_s"], training_samples=run["training_samples"]))
            masks = zone_masks(settings, result["width"], result["height"])
            errors = np.linalg.norm(np.asarray(case["calibration"]["corners"]) - truth, axis=1)
            geometry.append(dict(video_id=vid, calibration=mode, mean_corner_error_px=errors.mean(),
                                 max_corner_error_px=errors.max(), scale_px_per_m=settings["scale"],
                                 scale_error_pct=100 * (settings["scale"] / original["scale"] - 1),
                                 center_iou=iou(masks["center"], reference_masks["center"]),
                                 periphery_iou=iou(masks["periphery"], reference_masks["periphery"]),
                                 mask_iou=iou(masks["mask"], reference_masks["mask"])))
            provenance.append(dict(video_id=vid, calibration=mode, result=str(case["folder"] / "result.json"),
                                   result_sha256=sha256(case["folder"] / "result.json"),
                                   settings_sha256=sha256(case["folder"] / "settings.pickle"),
                                   calibration_sha256=sha256(case["folder"] / "calibration.json"),
                                   job=str(batch_root / mode / "jobs" / (run["job_id"] + ".json")),
                                   job_sha256=sha256(batch_root / mode / "jobs" / (run["job_id"] + ".json")),
                                   reference_settings_sha256=sha256(PILOT / "tailtrail/settings" / vid),
                                   base_config_sha256=sha256(PILOT / "oft/configs" / f"{vid}.toml"),
                                   dlc_trajectory_sha256=sha256(dlc_path), ratel_trajectory_sha256=sha256(ratel_path),
                                   job_id=run["job_id"]))
    primary = pd.concat(primary, ignore_index=True)
    agreement = compare(primary, reference="dlc")
    agreement["bias_pct_of_reference"] = 100 * agreement.bias / agreement.mean_a
    # Pool geometry-matched references across videos separately for each arm.
    pooled = []
    for mode in cases:
        rows = []
        for vid in videos:
            case = cases[mode][vid]
            cfg = config_for(vid, read_settings(case["folder"] / "settings.pickle"), case["result"])
            reference = metrics_for_trajectory(load_trajectory(PILOT / "oft/tracks/dlc" / f"{vid}.csv"), cfg, vid, "dlc")
            native = primary[(primary.video_id == vid) & (primary.tool == "tailtrail_" + mode)]
            rows += [reference, native]
        pooled.append(compare(pd.concat(rows, ignore_index=True), reference="dlc"))
    pd.concat(pooled, ignore_index=True).to_csv(output / "agreement_geometry_matched.csv", index=False)
    pd.concat(matched, ignore_index=True).to_csv(output / "per_video_geometry_matched.csv", index=False)
    primary.to_csv(output / "metrics.csv", index=False)
    agreement.to_csv(output / "agreement_fixed_reference.csv", index=False)
    geometry, runs = pd.DataFrame(geometry), pd.DataFrame(runs)
    geometry.to_csv(output / "calibration_accuracy.csv", index=False)
    runs.to_csv(output / "runs.csv", index=False)
    ref = primary[primary.tool == "dlc"][["video_id", "metric", "zone", "value"]].rename(columns={"value": "reference"})
    errors = primary[primary.tool != "dlc"].merge(ref, on=["video_id", "metric", "zone"])
    errors["error"] = errors.value - errors.reference
    errors["absolute_error"] = errors.error.abs()
    errors.to_csv(output / "per_video_errors.csv", index=False)
    plot_accuracy(primary, videos, output)
    title = "TailTrail automatic calibration benchmark"
    if len(videos) < len(mapping):
        title += f" — INTERIM ({len(videos)}/{len(mapping)} recordings)"
    distance = agreement[(agreement.metric == "total_distance_m") & (agreement.tool_b == "tailtrail_image")].iloc[0]
    center = agreement[(agreement.metric == "time_s") & (agreement.zone == "center") & (agreement.tool_b == "tailtrail_image")].iloc[0]
    image_geometry = geometry[geometry.calibration == "image"]
    report = [f"# {title}", "", f"Completed paired recordings: {', '.join(videos)}.", "",
              f"Image calibration had a maximum corner discrepancy of {image_geometry.max_corner_error_px.max():.2f} px "
              f"and maximum absolute scale discrepancy of {image_geometry.scale_error_pct.abs().max():.2f}%. "
              f"Native TailTrail mean distance was {distance.mean_b:.2f} m versus DLC {distance.mean_a:.2f} m "
              f"({distance.bias_pct_of_reference:+.1f}%); centre-time MAE was {center.mae:.2f} s.", "",
              "Both TailTrail arms used the same detector template, native engine, background training, "
              "and exact frame windows. The image arm fitted its geometry without DLC input; the landmark "
              "arm used the original pilot method. The square-arena side is assumed to be 0.42 m, as in the pilot.", "",
              "## Geometry accuracy", "",
              "Reference: DLC median arena landmarks and the original saved TailTrail ROI/mask geometry. "
              "IoU is pixel intersection over union; periphery is the exclusive ring after removing the centre.", "",
              markdown_table(["Recording", "Calibration", "Max corner error (px)", "Scale error (%)", "Centre IoU", "Periphery IoU", "Mask IoU"],
                             [[r.video_id, r.calibration, number(r.max_corner_error_px), number(r.scale_error_pct),
                               number(r.center_iou, 3), number(r.periphery_iou, 3), number(r.mask_iou, 3)] for r in geometry.itertuples()]), "",
              "## Agreement with DLC under the original pilot definitions", "",
              "DLC and RATEL positions were re-scored using the original saved ROIs and scale, with each "
              "recording cut to the exact TailTrail frame interval. TailTrail's start-zone entrance count "
              "was corrected using the existing adapter. Positive bias means overestimation. The bias "
              "percentage below uses the DLC mean as denominator; the CSV also retains the project's pair-mean bias.", ""]
    chosen = [("total_distance_m", "", "Distance (m)"), ("time_s", "center", "Centre time (s)"),
              ("time_s", "periphery", "Periphery time (s)"), ("entries", "center", "Centre entries"),
              ("entries", "periphery", "Periphery entries")]
    rows = []
    for metric, zone, label in chosen:
        for tool in ("tailtrail_image", "tailtrail_landmark", "ratel"):
            r = agreement[(agreement.metric == metric) & (agreement.zone == zone) & (agreement.tool_b == tool)].iloc[0]
            rows.append([label, TOOLS[tool], int(r.n), number(r.mean_a), number(r.mean_b), number(r.mae),
                         number(r.bias_pct_of_reference, 1), number(r.ccc, 3)])
    report += [markdown_table(["Metric", "Tool", "n", "DLC mean", "Tool mean", "MAE", "Bias vs DLC (%)", "CCC"], rows), "",
               "![Comparison by recording](accuracy.png)", "", "## Effect of geometry alone", "",
               "The same DLC positions were also scored with image-derived geometry. This isolates the "
               "effect of changed scale and ROIs without using TailTrail detections.", "",
               markdown_table(["Metric", "MAE", "Bias vs DLC (%)", "CCC"],
                              [[label, number(r.mae), number(r.bias_pct_of_reference, 1), number(r.ccc, 3)]
                               for metric, zone, label in chosen
                               for r in agreement[(agreement.metric == metric) & (agreement.zone == zone)
                                                  & (agreement.tool_b == "dlc_geometry_only")].itertuples()]), "",
               "## Run diagnostics", "",
               markdown_table(["Recording", "Calibration", "Frames", "Duration (s)", "Loss events", "Outside ROIs (s)", "Processing (s)"],
                              [[r.video_id, r.calibration, r.processed_frames, number(r.duration_s), int(r.losses),
                                number(r.outside_rois_s), number(r.elapsed_s, 1)] for r in runs.itertuples()]), "",
               "Outside-ROI time includes wall excursions and is not a direct detection-loss rate. "
               "The geometry-matched agreement CSV re-scores DLC using each arm's own ROIs and scale "
               "to distinguish tracker disagreement from geometry effects.", "",
               "## Scope", "",
               "This measures agreement with an automated DLC reference on four public mouse recordings. "
               "It is not independently established accuracy on the rat experiment. TailTrail exports "
               "summaries, so per-frame position accuracy, detection precision/recall and manual behaviour "
               "accuracy are not measured. DLC and RATEL retain the pilot's shared cleaning and hysteresis "
               "rules; native TailTrail totals retain its own tracking rules.", ""]
    historical = pd.read_csv(PILOT / "tailtrail/metrics.csv").fillna({"zone": ""})
    context = historical[historical.metric == "total_distance_m"].groupby("tool").value.mean()
    report += [f"For context, the historical GUI runs reported mean distance {context['tailtrail']:.2f} m "
               f"versus DLC {context['dlc']:.2f} m under their earlier approximate GUI windows. Those runs "
               "are preserved and are not treated as a timing-matched experimental control here.", ""]
    (output / "report.md").write_text("\n".join(report), encoding="utf-8")
    (output / "provenance.json").write_text(json.dumps(dict(
        recordings=videos, complete=len(videos) == len(mapping), analysis_script_sha256=sha256(Path(__file__)),
        python=sys.version, batch_root=str(batch_root), inputs=provenance), indent=2), encoding="utf-8")
    print(agreement[(agreement.metric.isin(["total_distance_m", "time_s", "entries"]))
                    & (agreement.tool_b != "dlc_geometry_only")][
                        ["metric", "zone", "tool_b", "n", "mean_a", "mean_b", "mae", "bias_pct_of_reference", "ccc"]].to_string(index=False))
    print("Wrote", output / "report.md")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch-root", type=Path, default=ROOT / "output/tailtrail_auto_benchmark")
    ap.add_argument("--output", type=Path, help="default: <batch-root>/summary")
    ap.add_argument("--partial", action="store_true", help="allow a clearly labelled interim report")
    args = ap.parse_args()
    analyse(args.batch_root.resolve(), (args.output or args.batch_root / "summary").resolve(), args.partial)


if __name__ == "__main__":
    main()
