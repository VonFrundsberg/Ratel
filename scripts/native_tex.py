"""Generate RATEL's editable TeX tables, numbers and vector figures from saved results.

No model fitting, video processing or source-data changes are performed.
Compile reports/native_analyzer/report.tex after running this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from behaviour.branding import display_name
from open_field_tex import NAMES as PREVIOUS_NAMES, escape, number, table

ROOT = Path(__file__).resolve().parents[1]
BEHAVIOURS = ["grooming", "supported_rearing", "unsupported_rearing", "rearing"]
LABELS = dict(zip(BEHAVIOURS, ["Grooming", "Supported rearing", "Unsupported rearing", "Combined rearing"]))
NAMES = {**PREVIOUS_NAMES, "native_default": display_name("native_default"), "native_raw": display_name("native_raw")}


def generate(out, dest, report_date):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gen = dest / "generated"
    gen.mkdir(parents=True, exist_ok=True)
    board = pd.read_csv(out / "comparison.csv").sort_values("macro_f1", ascending=False)
    agr = pd.read_csv(out / "metric_agreement.csv")
    movement = pd.read_csv(out / "tracker_agreement.csv").fillna({"zone": ""})
    fst = pd.read_csv(out / "fst_scores.csv")
    timing = pd.read_csv(out / "timing.csv")
    positions = pd.read_csv(out / "tracker_positions.csv")
    summaries = pd.read_csv(out / "summary_scores_vs_each_observer.csv")
    frames = pd.read_csv(out / "frame_scores.csv")
    delta = pd.read_csv(out / "paired_differences.csv").set_index("reference")
    catalog = pd.read_csv(out / "metric_catalog.csv")
    plan = json.loads((out / "benchmark_plan.json").read_text())
    verified = json.loads((out / "verification.json").read_text())
    full_video = json.loads((out / "full_video_run.json").read_text())
    card_path = out / "models/ratel_pose.json"
    if not card_path.exists():
        card_path = out / "models/native_pose.json"
    card = json.loads(card_path.read_text())
    native = board.set_index("configuration").loc["native_default"]
    previous = board.set_index("configuration").loc["temporal_hist_gradient"]
    bootstrap = delta.loc["temporal_hist_gradient"]
    inference = timing[timing.stage == "pose_features_and_inference"]
    training = timing[timing.stage == "training"]
    track_distance = movement[movement.metric == "total_distance_m"].iloc[0]

    def score(behaviour, metric, variant="default"):
        return agr[(agr.variant == variant) & (agr.metric == f"{behaviour}_{metric}")].iloc[0]

    macros = {
        "ReportDate": datetime.fromisoformat(report_date).strftime("%d %B %Y").lstrip("0"),
        "BenchmarkDate": datetime.fromisoformat(plan["generated_utc"]).strftime("%d %B %Y").lstrip("0"),
        "NMetricDefinitions": str(len(catalog)),
        "NConfigurations": str(len(board)),
        "NReferenceConfigurations": str(len(board) - 2),
        "NVideos": str(verified["held_out_videos"]),
        "NFolds": str(verified["held_out_folds"]),
        "NFstVideos": str(len(fst)),
        "NFeatures": str(len(card["feature_names"])),
        "NTrainingFrames": f"{card['training_frames']:,}",
        "NCommonFrames": f"{int(frames[(frames.variant == 'default') & (frames.behaviour == 'grooming')].frames.sum()):,}",
        "NativeMacro": number(native.macro_f1),
        "RawMacro": number(previous.macro_f1),
        "MacroLow": number(native.macro_ci_low),
        "MacroHigh": number(native.macro_ci_high),
        "MacroGain": number(bootstrap.macro_f1_difference),
        "MacroGainPoints": number(100 * bootstrap.macro_f1_difference, 2),
        "GainLow": number(bootstrap.paired_ci_low),
        "GainHigh": number(bootstrap.paired_ci_high),
        "GroomFone": number(native.grooming_f1),
        "SupportedFone": number(native.supported_rearing_f1),
        "UnsupportedFone": number(native.unsupported_rearing_f1),
        "RearFone": number(native.rearing_f1),
        "GroomHumanBouts": number(score("grooming", "bouts").mean_a, 2),
        "GroomRawBouts": number(score("grooming", "bouts", "raw").mean_b, 2),
        "GroomDefaultBouts": number(score("grooming", "bouts").mean_b, 2),
        "GroomBoutCcc": number(score("grooming", "bouts").ccc),
        "GroomLatencyMae": number(score("grooming", "latency_s").mae, 2),
        "DistanceCcc": number(track_distance.ccc),
        "DistanceMae": number(track_distance.mae),
        "PositionMedian": number(positions.median_error_mm.median(), 2),
        "FstKappa": number(fst.mean_rater_kappa.mean()),
        "FstMae": number(np.abs(fst.tool_s - fst.raters_mean_s).mean(), 2),
        "FoldTrainingSeconds": number(training.seconds.sum(), 2),
        "InferenceSeconds": number(inference.seconds.sum(), 2),
        "InferenceFrames": f"{int(inference.frames.sum()):,}",
        "InferenceThroughput": f"{inference.frames.sum() / inference.seconds.sum():,.0f}",
        "FullVideoFrames": f"{full_video['frames']:,}",
        "FullVideoDuration": number(full_video["duration_s"], 2),
        "FullVideoSeconds": number(full_video["seconds"], 2),
        "NFocusedTests": str(verified["focused_tests_passed"]),
        "TrainingStride": str(plan["train_stride"]),
        "RandomSeed": str(plan["seed"]),
    }
    (gen / "numbers.tex").write_text(
        "% RATEL display names; numbers from the unchanged saved benchmark evidence.\n" +
        "\n".join(r"\newcommand{\%s}{%s}" % (k, v) for k, v in macros.items()) + "\n", encoding="utf-8")

    rows = []
    for rec in board.to_dict("records"):
        values = [escape(NAMES[rec["configuration"]]),
                  *[number(rec[b + "_f1"]) for b in BEHAVIOURS], number(rec["macro_f1"])]
        if rec["configuration"] == "native_default":
            values = [r"\textbf{" + value + "}" for value in values]
        rows.append(values)
    table(gen / "classification.tex", ["Configuration", "Groom.", "Supp.", "Unsupp.", "All rear", "Macro"],
          rows, "@{}lrrrrr@{}")

    requirements = {
        "trajectory": ("Trajectory and zone summaries", "Position, scale and arena geometry"),
        "state": ("Named state summaries", "Seven states, eight summaries each"),
        "point": ("Point-event summaries", "Three events, three summaries each"),
        "fst": ("FST motion summaries and quality", "Motion, area and test window"),
        "fst_annotations": ("FST coded-state summaries", "Swim/climb annotations and uncoded time"),
        "pose_model": ("Behaviour scoring coverage", "Pose validity during the analysis window"),
        "vendor_export": ("Vendor-specific diagnostic", "TailTrail's exported loss count"),
    }
    counts = catalog.groupby("input").size()
    rows = [[label, str(int(counts[k])), detail] for k, (label, detail) in requirements.items()]
    table(gen / "coverage.tex", ["Measurement family", "Fields", "Required information"], rows, "@{}lrl@{}")

    table(gen / "duration.tex", ["Behaviour", "Human (s)", "RATEL (s)", "MAE (s)", "Bias (s)", "CCC"],
          [[LABELS[b], *[number(score(b, "time_s")[m], 2 if m != "ccc" else 3, m == "bias")
                        for m in ["mean_a", "mean_b", "mae", "bias", "ccc"]]] for b in BEHAVIOURS], "@{}lrrrrr@{}")
    table(gen / "bouts.tex", ["Behaviour", "Human", "Raw", "Filtered", "MAE", "CCC"],
          [[LABELS[b], number(score(b, "bouts").mean_a, 2), number(score(b, "bouts", "raw").mean_b, 2),
            number(score(b, "bouts").mean_b, 2), number(score(b, "bouts").mae, 2), number(score(b, "bouts").ccc)]
           for b in BEHAVIOURS], "@{}lrrrrr@{}")
    table(gen / "latency.tex", ["Behaviour", "$n$", "Human (s)", "RATEL (s)", "MAE (s)", "CCC"],
          [[LABELS[b], str(int(score(b, "latency_s").n)),
            *[number(score(b, "latency_s")[m], 2 if m != "ccc" else 3) for m in ["mean_a", "mean_b", "mae", "ccc"]]]
           for b in BEHAVIOURS], "@{}lrrrrr@{}")
    definitions = [("total_distance_m", "", "Distance (m)"), ("time_s", "center", "Centre time (s)"),
                   ("time_s", "periphery", "Periphery time (s)"), ("entries", "center", "Centre entries"),
                   ("entries", "periphery", "Periphery entries"), ("immobile_time_s", "", "Centroid immobility (s)")]
    rows = []
    for metric, zone, label in definitions:
        r = movement[(movement.metric == metric) & (movement.zone == zone)].iloc[0]
        rows.append([label, number(r.mean_a, 2), number(r.mean_b, 2), number(r.bias, 3, True), number(r.mae), number(r.ccc)])
    table(gen / "tracking.tex", ["Metric", "DLC mean", "RATEL mean", "Bias", "MAE", "CCC"], rows, "@{}lrrrrr@{}")
    fst["order"] = fst.video_id.str.extract(r"(\d+)$").astype(int)
    fst = fst.sort_values("order")
    table(gen / "fst.tex", ["Video", "Threshold", "Smooth (s)", "Min. (s)", r"$\kappa$", "RATEL (s)", "Human (s)"],
          [[escape(r.video_id), number(r.threshold, 4), number(r.smooth_s, 1), number(r.min_bout_s, 1),
            number(r.mean_rater_kappa), number(r.tool_s, 2), number(r.raters_mean_s, 2)]
           for r in fst.itertuples(index=False)], "@{}lrrrrrr@{}")
    stages = {"training": "Five held-out model fits and saves", "pose_features_and_inference": "Pose features and classifier inference",
              "metrics_raw": "Raw metric summaries", "metrics_default": "Default metric summaries",
              "deployment_training": "Separate deployment-model fit and save"}
    rows = []
    for key, label in stages.items():
        g = timing[timing.stage == key]
        rows.append([label, f"{int(g.frames.sum()):,}", number(g.seconds.sum(), 2)])
    table(gen / "runtime.tex", ["Measured stage", "Frames", "Wall time (s)"], rows, "@{}lrr@{}")

    plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                         "pdf.fonttype": 42, "savefig.bbox": "tight", "axes.spines.top": False,
                         "axes.spines.right": False})
    plotted = board.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.3, 3.9))
    colors = ["#087f73" if c.startswith("native_") else "#718aa2" for c in plotted.configuration]
    errors = np.vstack([plotted.macro_f1 - plotted.macro_ci_low, plotted.macro_ci_high - plotted.macro_f1])
    ax.barh([NAMES[c] for c in plotted.configuration], plotted.macro_f1, color=colors,
            xerr=errors, error_kw=dict(ecolor="#243746", capsize=2, linewidth=.7))
    ax.set(xlabel="Macro F1 across three named behaviours", xlim=(0, .8))
    ax.tick_params(axis="y", labelsize=8)
    ax.xaxis.grid(True, alpha=.18)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(gen / "classification.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(7.3, 2.6))
    for ax, b in zip(axes, BEHAVIOURS[:3]):
        g = summaries[(summaries.variant == "default") & (summaries.metric == f"{b}_time_s")]
        g = g.groupby("video_id")[["manual", "predicted"]].mean()
        limit = max(g.to_numpy().max(), 1) * 1.08
        ax.scatter(g.manual, g.predicted, s=20, color="#087f73", edgecolors="white", linewidths=.4)
        ax.plot([0, limit], [0, limit], "--", color="#7c8c97", linewidth=.8)
        ax.set(xlim=(0, limit), ylim=(0, limit), xlabel="Mean observer time (s)", title=LABELS[b])
        ax.set_aspect("equal", adjustable="box")
    axes[0].set_ylabel("RATEL duration (s)")
    fig.tight_layout()
    fig.savefig(gen / "duration.pdf")
    plt.close(fig)

    inputs = [out / p for p in ["comparison.csv", "metric_agreement.csv", "tracker_agreement.csv", "fst_scores.csv", "timing.csv",
              "tracker_positions.csv", "summary_scores_vs_each_observer.csv", "frame_scores.csv", "paired_differences.csv",
              "metric_catalog.csv", "benchmark_plan.json", "verification.json", "full_video_run.json"]]
    inputs.append(card_path)
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    manifest = dict(report_date=report_date, benchmark_date=plan["generated_utc"],
                    input_sha256={str(p.resolve()): digest(p) for p in inputs},
                    generated_sha256={p.name: digest(p) for p in sorted(gen.iterdir()) if p.is_file()},
                    generator_sha256=digest(Path(__file__)),
                    software="RATEL", display_names={k: NAMES[k] for k in ("native_default", "native_raw")},
                    source="Completed RATEL benchmark; no training or video processing during report generation")
    (dest / "tex_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"TeX assets generated in {gen}: {len(board)} configurations, {len(catalog)} definitions, two vector figures")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=ROOT / "output/native_benchmark")
    ap.add_argument("--report-dir", type=Path, default=ROOT / "reports/native_analyzer")
    ap.add_argument("--report-date", default="2026-10-18", help="ISO date printed on this report")
    a = ap.parse_args()
    generate(a.output, a.report_dir, a.report_date)


if __name__ == "__main__":
    main()
