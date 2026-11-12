"""Generate LaTeX tables, numbers and vector figures from the completed benchmark.

This reads saved results only; it never trains models or changes source data.
Run before compiling reports/open_field_expansion/report.tex.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NAMES = {
    "temporal_hist_gradient": "Temporal gradient boosting",
    "temporal_mlp": "Temporal MLP",
    "temporal_rf_seed2": "Temporal RF (second seed)",
    "temporal_rf": "Temporal random forest",
    "temporal_extra_trees": "Temporal extra trees",
    "pose_rf": "Pose random forest",
    "pose_logistic": "Pose logistic regression",
    "simba_rf": "SimBA features + custom RF",
    "deepethogram_rgb_probe": "DeepEthogram RGB probe",
    "simba_extra_trees": "SimBA features + extra trees",
    "labgym_animation_small": "LabGym small animation model",
    "simba_native_binary_rf": "SimBA native binary RF",
    "majority": "Majority-class reference",
}
BEHAVIOURS = ["grooming", "supported_rearing", "unsupported_rearing", "rearing"]
LABELS = dict(zip(BEHAVIOURS, ["Grooming", "Supported rearing", "Unsupported rearing", "Combined rearing"]))


def escape(value):
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
                    "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
                    "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(replacements.get(c, c) for c in str(value))


def number(value, digits=3, signed=False):
    if not np.isfinite(value):
        return "--"
    return format(value, f"+.{digits}f" if signed else f".{digits}f")


def table(path, headers, rows, columns):
    lines = [r"\begin{tabular}{" + columns + "}", r"\toprule",
             " & ".join(headers) + r" \\", r"\midrule"]
    lines += [" & ".join(row) + r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def generate(out, dest):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gen = dest / "generated"
    gen.mkdir(parents=True, exist_ok=True)
    scores = pd.read_csv(out / "leaderboard.csv")
    raw = scores[(scores.variant == "raw") & (scores.cadence == "common_2.5hz")]
    wide = raw.pivot(index="tool", columns="behaviour", values="f1")
    wide["macro"] = wide[BEHAVIOURS[:3]].mean(axis=1)
    wide = wide.sort_values("macro", ascending=False)
    top = wide.index[0]
    agr = pd.read_csv(out / "metric_agreement.csv")
    human = pd.read_csv(out / "human_agreement_summary.csv").set_index("behaviour")
    quality = pd.read_csv(out / "data_quality.csv")
    movement = pd.read_csv(out / "movement_metrics.csv")
    positions = pd.read_csv(out / "tracker_positions.csv")
    verification = json.loads((out / "verification.json").read_text())
    features = json.loads((out / "features.json").read_text())

    def agreement_row(behaviour, metric, variant="raw"):
        return agr[(agr.tool == top) & (agr.behaviour == behaviour) &
                   (agr.metric == metric) & (agr.variant == variant)].iloc[0]

    macro = {
        "RunDate": datetime.fromisoformat(verification["checked_utc"]).strftime("%d %B %Y").lstrip("0"),
        "NConfigurations": str(len(wide)),
        "NLabelledVideos": str(len(quality)),
        "NWindowFrames": f"{int(quality.frames.sum()):,}",
        "NPredictions": str(verification["held_out_prediction_files"]),
        "LabelledHours": f"{quality.scored_s.sum()/3600:.2f}",
        "AmbiguousPct": f"{100*(quality.frames*quality.consensus_ambiguous_fraction).sum()/quality.frames.sum():.2f}",
        "NMovementVideos": str(movement.video_id.nunique()),
        "NMovementRows": f"{len(movement):,}",
        "MovementHours": f"{movement.loc[movement.metric == 'duration_s', 'value'].sum()/3600:.2f}",
        "TrackedMeanPct": f"{100*movement.loc[movement.metric == 'tracked_fraction', 'value'].mean():.2f}",
        "TrackedMinPct": f"{100*movement.loc[movement.metric == 'tracked_fraction', 'value'].min():.2f}",
        "PositionMedianMm": f"{positions.median_error_mm.median():.2f}",
        "PositionMinMm": f"{positions.median_error_mm.min():.2f}",
        "PositionMaxMm": f"{positions.median_error_mm.max():.2f}",
        "PositionPairPct": f"{100*positions.valid_position_pairs.sum()/positions.compared_frames.sum():.3f}",
        "NShapeFeatures": str(len(features["shape"])),
        "NTemporalFeatures": str(len(features["temporal"])),
        "TopMacroFone": number(wide.loc[top, "macro"]),
        "GroomRawBouts": number(agreement_row("grooming", "bouts").mean_b, 2),
        "GroomHumanBouts": number(agreement_row("grooming", "bouts").mean_a, 2),
        "GroomSmoothBouts": number(agreement_row("grooming", "bouts", "smooth_1.0s").mean_b, 2),
        "GroomRawCcc": number(agreement_row("grooming", "time_s").ccc),
        "GroomSmoothCcc": number(agreement_row("grooming", "time_s", "smooth_1.0s").ccc),
    }
    for b, name in zip(BEHAVIOURS, ["Groom", "Supported", "Unsupported", "Rear"]):
        macro[name + "Fone"] = number(wide.loc[top, b])
        macro[name + "HumanFone"] = number(human.loc[b, "f1"])
    (gen / "numbers.tex").write_text(
        "% Generated from saved benchmark results; do not edit.\n" +
        "\n".join(r"\newcommand{\%s}{%s}" % (k, v) for k, v in macro.items()) + "\n",
        encoding="utf-8")

    rows = []
    for name, row in wide.iterrows():
        values = [escape(NAMES.get(name, name)), *[number(row[b]) for b in BEHAVIOURS], number(row["macro"])]
        if name == top:
            values = [r"\textbf{" + s + "}" for s in values]
        rows.append(values)
    table(gen / "ranking.tex",
          ["Configuration", "Groom.", "Supp.", "Unsupp.", "All rear", "Macro"],
          rows, "@{}lrrrrr@{}")

    rows = []
    for b in BEHAVIOURS:
        r = raw[(raw.tool == top) & (raw.behaviour == b)].iloc[0]
        rows.append([LABELS[b], number(r.precision), number(r.recall), number(r.f1),
                     number(r.f1_ci_low) + "--" + number(r.f1_ci_high)])
    table(gen / "leader_uncertainty.tex", ["Behaviour", "Precision", "Recall", "$F_1$", "95\\% interval"],
          rows, "@{}lrrrr@{}")
    table(gen / "human.tex", ["Behaviour", "Precision", "Recall", "$F_1$"],
          [[LABELS[b], *[number(human.loc[b, m]) for m in ["precision", "recall", "f1"]]]
           for b in BEHAVIOURS], "@{}lrrr@{}")
    table(gen / "duration.tex",
          ["Behaviour", "Human (s)", "Model (s)", "MAE (s)", "Bias (s)", "CCC"],
          [[LABELS[b], *[number(agreement_row(b, "time_s")[m], 2 if m != "ccc" else 3, m == "bias")
                         for m in ["mean_a", "mean_b", "mae", "bias", "ccc"]]]
           for b in BEHAVIOURS], "@{}lrrrrr@{}")
    rows = []
    for b in BEHAVIOURS:
        r, s, t = [agreement_row(b, "bouts", v) for v in ["raw", "smooth_0.2s", "smooth_1.0s"]]
        rows.append([LABELS[b], *[number(v, 2) for v in [r.mean_a, r.mean_b, s.mean_b, t.mean_b, t.mae]],
                     number(t.ccc)])
    table(gen / "bouts.tex",
          ["Behaviour", "Human", "Raw", "0.2 s", "1.0 s", "MAE, 1 s", "CCC, 1 s"],
          rows, "@{}lrrrrrr@{}")
    rows = []
    for variant, label in [("raw", "Raw"), ("smooth_0.2s", "0.2 s smoothing"),
                           ("smooth_1.0s", "1.0 s smoothing"), ("label_shift_-6", "Labels shifted $-6$ frames"),
                           ("label_shift_6", "Labels shifted $+6$ frames")]:
        g = scores[(scores.tool == top) & (scores.variant == variant) &
                   (scores.cadence == "common_2.5hz")].set_index("behaviour")
        rows.append([label, *[number(g.loc[b, "f1"]) for b in BEHAVIOURS]])
    table(gen / "sensitivity.tex", ["Setting", "Groom.", "Supp.", "Unsupp.", "All rear"],
          rows, "@{}lrrrr@{}")
    comparison = pd.read_csv(out / "tracker_agreement.csv").fillna({"zone": ""})
    definitions = [("total_distance_m", "", "Distance (m)"), ("time_s", "center", "Centre time (s)"),
                   ("time_s", "periphery", "Periphery time (s)"), ("entries", "center", "Centre entries"),
                   ("entries", "periphery", "Periphery entries"), ("immobile_time_s", "", "Immobile time (s)")]
    rows = []
    for metric, zone, label in definitions:
        r = comparison[(comparison.metric == metric) & (comparison.zone == zone)].iloc[0]
        rows.append([label, number(r.mean_a, 2), number(r.mean_b, 2), number(r.bias, 3, True),
                     number(r.mae), number(r.ccc)])
    table(gen / "tracking.tex", ["Metric", "DLC mean", "RATEL mean", "Bias", "MAE", "CCC"],
          rows, "@{}lrrrrr@{}")

    plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.titlesize": 10,
                         "axes.labelsize": 9, "pdf.fonttype": 42, "savefig.bbox": "tight",
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(7.5, 4.55))
    matrix = wide[BEHAVIOURS].to_numpy()
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(4), ["Grooming", "Supported\nrearing", "Unsupported\nrearing", "Combined\nrearing"])
    ax.set_yticks(range(len(wide)), [NAMES[n] for n in wide.index])
    ax.tick_params(length=0)
    for i in range(matrix.shape[0]):
        for j in range(4):
            ax.text(j, i, number(matrix[i, j], 2), ha="center", va="center",
                    color="white" if matrix[i, j] > .6 else "#18252d")
    fig.colorbar(im, ax=ax, fraction=.03, pad=.03, label="$F_1$")
    fig.tight_layout()
    fig.savefig(gen / "classification.pdf")
    plt.close(fig)
    summary = pd.read_csv(out / "summary_scores_vs_each_observer.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.65))
    for ax, b in zip(axes, BEHAVIOURS[:3]):
        g = summary[(summary.tool == top) & (summary.variant == "raw") &
                    (summary.behaviour == b) & (summary.metric == "time_s")]
        g = g.groupby("video_id")[["manual", "predicted"]].mean()
        limit = max(g.to_numpy().max(), 1)*1.06
        ax.scatter(g.manual, g.predicted, color="#176b87", s=18, edgecolors="white", linewidths=.3)
        ax.plot([0, limit], [0, limit], "--", color="#8c969c", linewidth=1)
        ax.set(xlim=(0, limit), ylim=(0, limit), xlabel="Mean observer duration (s)", title=LABELS[b])
        ax.set_aspect("equal", adjustable="box")
    axes[0].set_ylabel("Predicted duration (s)")
    fig.tight_layout()
    fig.savefig(gen / "duration.pdf")
    plt.close(fig)
    positions = positions.sort_values("median_error_mm")
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.barh(positions.video_id.str.replace("_", " "), positions.median_error_mm, color="#176b87")
    ax.invert_yaxis()
    ax.set(xlabel="Median position disagreement (mm; assumed arena scale)", ylabel="Recording")
    ax.tick_params(axis="y", labelsize=8)
    ax.xaxis.grid(True, alpha=.2)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(gen / "tracking.pdf")
    plt.close(fig)
    print(f"TeX assets: {gen}; {len(wide)} configurations; leading model {top}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=ROOT / "output/open_field_benchmark")
    ap.add_argument("--report-dir", type=Path, default=ROOT / "reports/open_field_expansion")
    a = ap.parse_args()
    generate(a.output, a.report_dir)


if __name__ == "__main__":
    main()
