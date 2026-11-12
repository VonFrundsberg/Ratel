"""Evaluate held-out predictions, human agreement and dataset coverage."""

from __future__ import annotations

import json
import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from behaviour.compare import agreement
from behaviour.fst import runs
from behaviour.ethogram import behaviour_rows, normalize_events

NAMES = ["none", "grooming", "supported_rearing", "unsupported_rearing"]
BEHAVIOURS = [*NAMES[1:], "rearing"]
SEED = 20261017
SCOPES = {
    "majority": "Training-fold majority-class reference; no behaviour detector.",
    "pose_logistic": "Custom balanced logistic regression on pose geometry.",
    "pose_rf": "Custom random forest on pose geometry.",
    "temporal_rf": "Custom random forest on geometry, movement and temporal features.",
    "temporal_rf_seed2": "Same temporal forest with a second fixed random seed.",
    "temporal_extra_trees": "Custom extra-trees classifier on temporal pose features.",
    "temporal_hist_gradient": "Custom histogram gradient boosting on temporal pose features.",
    "temporal_mlp": "Custom 64/32-unit MLP on temporal pose features.",
    "simba_rf": "Native SimBA 8-point features; custom multiclass random forest.",
    "simba_extra_trees": "Native SimBA 8-point features; custom multiclass extra trees.",
    "simba_native_binary_rf": "Native SimBA features and RF factory; three independent binary classifiers at threshold 0.5.",
    "deepethogram_rgb_probe": "Native frozen ImageNet ResNet18 RGB encoder; custom balanced logistic head; 64-pixel pose-guided crops.",
    "labgym_animation_small": "Native LabGym small animation model; 15 grayscale 16-pixel frames, eight epochs, CPU, pose-guided crops.",
}


def counts(truth, predicted):
    return dict(tp=int((truth & predicted).sum()), fp=int((~truth & predicted).sum()),
                fn=int((truth & ~predicted).sum()), tn=int((~truth & ~predicted).sum()))


def rates(row):
    tp, fp, fn, tn = (row[k] for k in ["tp", "fp", "fn", "tn"])
    divide = lambda a, b: a / b if b else np.nan
    return dict(precision=divide(tp, tp+fp), recall=divide(tp, tp+fn),
                specificity=divide(tn, tn+fp), f1=divide(2*tp, 2*tp+fp+fn))


def mask_for(y, behaviour):
    return np.isin(y, [2, 3]) if behaviour == "rearing" else y == NAMES.index(behaviour)


def state_metrics(mask, times, dt, end, start=None):
    start = times[0] if start is None else start
    stop = np.minimum(np.r_[times[1:], times[-1]+dt], end)
    durations = np.maximum(stop-times, 0)
    iv = runs(mask)
    lengths = np.array([durations[a:b].sum() for a, b in iv])
    time = float(durations[mask].sum())
    return dict(time_s=time, time_pct=100*time/(end-start), bouts=len(iv),
                mean_bout_s=float(lengths.mean()) if len(iv) else np.nan,
                median_bout_s=float(np.median(lengths)) if len(iv) else np.nan,
                max_bout_s=float(lengths.max()) if len(iv) else np.nan,
                frequency_per_min=60*len(iv)/(end-start),
                latency_s=float(times[iv[0][0]]-start) if iv else np.nan)


def evaluate(out):
    from scipy.ndimage import uniform_filter1d
    from sklearn.metrics import confusion_matrix

    metadata = {r["video_id"]: r for r in json.loads((out / "recordings.json").read_text())}
    folds = json.loads((out / "splits.json").read_text())["folds"]
    expected = {v: f for f in folds for v in f["test"]}
    folders = [p for p in sorted((out / "predictions").iterdir())
               if all((p / f"{v}.npz").exists() for v in metadata)]
    label_path = Path(next(iter(metadata.values()))["pose_path"]).parent.parent / "Labels/AllLabDataOFT_final.csv"
    original = pd.read_csv(label_path, sep=";")
    manual_events = original[original.type.isin(["Grooming", "Supported", "Unsupported"])].rename(
        columns={"ID": "video_id", "Experimenter": "observer", "type": "behaviour", "from": "start", "to": "stop"})
    manual_events["event_type"] = "state"
    manual_events = normalize_events(manual_events[["video_id", "observer", "behaviour", "start", "stop", "event_type"]])
    manual_events.to_csv(out / "manual_events.csv", index=False)
    manual_metric_rows = []
    frame_rows, summary_rows, human_rows, confusion_rows, metric_rows = [], [], [], [], []
    for vid, rec in metadata.items():
        with np.load(out / "cache" / f"{vid}.npz") as z:
            truth, raters, times, observers = z["y"], z["raters"], z["t"], z["observers"]
        continuous_manual = {}
        for observer in observers:
            ev = manual_events[(manual_events.video_id == vid) & (manual_events.observer == observer)].copy()
            rears = ev[ev.behaviour.isin(["supported_rearing", "unsupported_rearing"])].copy()
            rears["behaviour"] = "rearing"
            ev = pd.concat([ev, rears], ignore_index=True)
            values = behaviour_rows(ev, rec["start_s"], rec["end_s"], dict.fromkeys(BEHAVIOURS, "state"))
            continuous_manual[observer] = {r["metric"]: r["value"] for r in values}
            manual_metric_rows += [dict(video_id=vid, tool=f"manual_{observer}", **r) for r in values]
        for j, observer in enumerate(observers):
            other = np.delete(raters, j, axis=1)
            ok = (other[:, 0] == other[:, 1]) & (other[:, 0] >= 0) & (raters[:, j] >= 0)
            for b in BEHAVIOURS:
                c = counts(mask_for(other[ok, 0], b), mask_for(raters[ok, j], b))
                human_rows.append(dict(video_id=vid, observer=observer, behaviour=b,
                                       compared_frames=int(ok.sum()), **c, **rates(c)))
        for folder in folders:
            path = folder / f"{vid}.npz"
            if not path.exists():
                continue
            with np.load(path) as z:
                p, t = z["p"], z["t"]
                multilabel = bool(z["multilabel"]) if "multilabel" in z else False
                trained_on = set(z["trained_on"].tolist())
                if trained_on != set(expected[vid]["train"]) or int(z["fold"]) != expected[vid]["fold"]:
                    raise ValueError(f"{path}: training/evaluation leakage")
                if not np.isfinite(p).all() or p.shape != (len(t), 4):
                    raise ValueError(f"{path}: malformed predictions")
                if (p < 0).any() or (p > 1).any() or (np.diff(t) <= 0).any():
                    raise ValueError(f"{path}: invalid probabilities or timestamps")
                if not multilabel and not np.allclose(p.sum(axis=1), 1, atol=1e-5):
                    raise ValueError(f"{path}: multiclass probabilities do not sum to one")
            ix = np.searchsorted(times, t)
            if np.any(ix >= len(times)) or not np.allclose(times[ix], t, atol=1e-5):
                raise ValueError(f"{path}: timestamps do not match the original recording")
            if not (np.array_equal(ix, np.arange(len(times))) or
                    np.array_equal(ix, np.arange(0, len(times), 10))):
                raise ValueError(f"{path}: incomplete prediction cadence")
            dt = float(np.median(np.diff(t)))
            target, rater = truth[ix], raters[ix]
            common = (ix % 10 == 0)  # shared 2.5 Hz evaluation cadence
            for smooth_s in (0.0, 0.2, 1.0):
                w = max(1, int(round(smooth_s / dt))) | 1
                proba = uniform_filter1d(p, size=w, axis=0, mode="nearest") if w > 1 else p
                prediction = proba.argmax(axis=1)
                predicted_masks = {b: mask_for(prediction, b) for b in BEHAVIOURS}
                if multilabel:
                    predicted_masks = {b: proba[:, NAMES.index(b)] >= 0.5 for b in NAMES[1:]}
                    predicted_masks["rearing"] = predicted_masks["supported_rearing"] | predicted_masks["unsupported_rearing"]
                variant = "raw" if smooth_s == 0 else f"smooth_{smooth_s}s"
                for cadence, cadence_mask in [("native", np.ones(len(t), bool)), ("common_2.5hz", common)]:
                    ok = (target >= 0) & cadence_mask
                    if not ok.any():
                        continue
                    for b in BEHAVIOURS:
                        c = counts(mask_for(target[ok], b), predicted_masks[b][ok])
                        frame_rows.append(dict(video_id=vid, tool=folder.name, variant=variant, cadence=cadence,
                                               behaviour=b, frames=int(ok.sum()), **c, **rates(c)))
                if smooth_s == 0:
                    if not multilabel:
                        cm = confusion_matrix(target[target >= 0], prediction[target >= 0], labels=np.arange(4))
                        for a in range(4):
                            for b in range(4):
                                confusion_rows.append(dict(video_id=vid, tool=folder.name, truth=NAMES[a],
                                                            predicted=NAMES[b], frames=int(cm[a, b])))
                    for shift in (-6, 6):
                        shift_ix = ix + shift
                        ok = (shift_ix >= 0) & (shift_ix < len(truth)) & common
                        ok[ok] &= truth[shift_ix[ok]] >= 0
                        for b in BEHAVIOURS:
                            c = counts(mask_for(truth[shift_ix[ok]], b), predicted_masks[b][ok])
                            frame_rows.append(dict(video_id=vid, tool=folder.name, variant=f"label_shift_{shift}",
                                                   cadence="common_2.5hz", behaviour=b, frames=int(ok.sum()), **c, **rates(c)))
                for b in BEHAVIOURS:
                    predicted = state_metrics(predicted_masks[b], t, dt, rec["end_s"], rec["start_s"])
                    for j, observer in enumerate(observers):
                        manual = {m: continuous_manual[observer][f"{b}_{m}"] for m in predicted}
                        for metric in predicted:
                            summary_rows.append(dict(video_id=vid, tool=folder.name, variant=variant,
                                                     observer=observer, behaviour=b, metric=metric,
                                                     manual=manual[metric], predicted=predicted[metric]))
                    if smooth_s == 0:
                        for metric, value in predicted.items():
                            metric_rows.append(dict(video_id=vid, tool=folder.name, metric=f"{b}_{metric}",
                                                    zone="", value=value, kind="behaviour"))
    frames, summaries, human = map(pd.DataFrame, [frame_rows, summary_rows, human_rows])
    frames.to_csv(out / "frame_scores.csv", index=False)
    summaries.to_csv(out / "summary_scores_vs_each_observer.csv", index=False)
    human.to_csv(out / "human_agreement.csv", index=False)
    pd.DataFrame(confusion_rows).to_csv(out / "confusion_matrices.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(out / "behaviour_metrics.csv", index=False)
    pd.DataFrame(manual_metric_rows).to_csv(out / "manual_metrics.csv", index=False)
    rng = np.random.default_rng(SEED)
    pooled = []
    for key, g in frames.groupby(["tool", "variant", "cadence", "behaviour"]):
        c = g[["tp", "fp", "fn", "tn"]].sum().to_dict()
        x = g[["tp", "fp", "fn", "tn"]].to_numpy(float)
        boot = x[rng.integers(0, len(x), size=(2000, len(x)))].sum(axis=1)
        den = 2*boot[:, 0]+boot[:, 1]+boot[:, 2]
        f1 = np.divide(2*boot[:, 0], den, out=np.zeros(len(boot)), where=den > 0)
        pooled.append(dict(zip(["tool", "variant", "cadence", "behaviour"], key),
                           videos=g.video_id.nunique(), **c, **rates(c),
                           f1_ci_low=float(np.quantile(f1, .025)), f1_ci_high=float(np.quantile(f1, .975))))
    leaderboard = pd.DataFrame(pooled)
    leaderboard.to_csv(out / "leaderboard.csv", index=False)
    mean_rater = summaries.groupby(["video_id", "tool", "variant", "behaviour", "metric"], as_index=False)[
        ["manual", "predicted"]].mean()
    agr = []
    for key, g in mean_rater.groupby(["tool", "variant", "behaviour", "metric"]):
        agr.append(dict(zip(["tool", "variant", "behaviour", "metric"], key),
                        **agreement(g.manual.to_numpy(), g.predicted.to_numpy())))
    pd.DataFrame(agr).to_csv(out / "metric_agreement.csv", index=False)
    human_summary = []
    for b, g in human.groupby("behaviour"):
        c = g[["tp", "fp", "fn", "tn"]].sum().to_dict()
        human_summary.append(dict(behaviour=b, **c, **rates(c)))
    pd.DataFrame(human_summary).to_csv(out / "human_agreement_summary.csv", index=False)
    return leaderboard, pd.DataFrame(agr), pd.DataFrame(human_summary), frames


def markdown_table(df):
    rows = [list(df.columns), ["---"]*len(df.columns), *df.astype(str).values.tolist()]
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


def save_provenance(out, dest, tools):
    root = Path(__file__).resolve().parents[1]
    paths = [root / "pyproject.toml", root / "configs/open_field_ethogram.csv",
             *sorted((root / "scripts").glob("*benchmark*.py")),
             root / "scripts/tool_worker.py", root / "scripts/open_field_report.py",
             root / "scripts/setup_behaviour_tools.py",
             root / "src/behaviour/ethogram.py", root / "src/behaviour/pose_features.py",
             root / "src/behaviour/adapters/behaviours.py",
             root / "src/behaviour/adapters/boris.py", root / "src/behaviour/cli.py",
             *sorted((out / "logs").glob("packages_*.txt"))]
    paths += [out / name for name in ["recordings.json", "splits.json", "data_quality.csv",
                                      "manual_events.csv", "manual_metrics.csv", "leaderboard.csv",
                                      "metric_agreement.csv", "movement_status.csv", "movement_metrics.csv",
                                      "tracker_positions.csv", "tracker_agreement.csv",
                                      "verification.json", "features.json", "pixel_cache/specification.json",
                                      "simba_project/project_config.ini"]]
    paths += [dest / "workflow.md", root / "src/behaviour/metrics.py",
              root / "src/behaviour/trajectory.py", root / "src/behaviour/compare.py"]
    records = json.loads((out / "recordings.json").read_text())
    paths.append(Path(records[0]["pose_path"]).parent.parent / "Labels/AllLabDataOFT_final.csv")
    hashes = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}
    provenance = dict(generated_utc=datetime.now(timezone.utc).isoformat(), seed=SEED,
                      completed_configurations=list(tools), recordings=records, sha256=hashes,
                      evaluation=dict(folds=5, train_videos_per_fold=16, test_videos_per_fold=4,
                                      main_cadence_hz=2.5, video_bootstrap_draws=2000,
                                      variants=["raw", "smooth_0.2s", "smooth_1.0s", "label_shift_-6", "label_shift_6"]))
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    (dest / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")


def render_html(lines, dest):
    """Render this report's small Markdown subset into a standalone HTML file."""
    import base64
    import html

    def inline(s):
        s = html.escape(s)
        return re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r'<a href="\2">\1</a>', s)

    parts, i = [], 0
    while i < len(lines):
        line = lines[i]
        if not line:
            i += 1
            continue
        if line.startswith("~~~"):
            i += 1
            code = []
            while i < len(lines) and not lines[i].startswith("~~~"):
                code.append(lines[i])
                i += 1
            parts.append("<pre><code>" + html.escape("\n".join(code)) + "</code></pre>")
        elif line.startswith("|"):
            block = []
            while i < len(lines) and lines[i].startswith("|"):
                block.append([s.strip() for s in lines[i].strip("|").split("|")])
                i += 1
            header = "<tr>" + "".join("<th>" + inline(s) + "</th>" for s in block[0]) + "</tr>"
            rows = "".join("<tr>" + "".join("<td>" + inline(s) + "</td>" for s in row) + "</tr>"
                           for row in block[2:])
            parts.append('<div class="table"><table><thead>' + header + "</thead><tbody>" + rows + "</tbody></table></div>")
            continue
        elif line.startswith("- "):
            items = []
            while i < len(lines) and lines[i].startswith("- "):
                items.append("<li>" + inline(lines[i][2:]) + "</li>")
                i += 1
            parts.append("<ul>" + "".join(items) + "</ul>")
            continue
        elif line.startswith("#"):
            level = len(line)-len(line.lstrip("#"))
            parts.append(f"<h{level}>" + inline(line[level:].strip()) + f"</h{level}>")
        elif line.startswith("!["):
            match = re.fullmatch(r"!\[([^]]*)\]\(([^)]+)\)", line)
            picture = dest / match[2]
            data = base64.b64encode(picture.read_bytes()).decode()
            parts.append(f'<figure><img alt="{html.escape(match[1])}" src="data:image/png;base64,{data}"></figure>')
        else:
            parts.append("<p>" + inline(line) + "</p>")
        i += 1
    css = (
        "body{font:16px/1.55 system-ui;max-width:1140px;margin:40px auto;padding:0 24px;color:#17252a}"
        "h1{font-size:2.1rem}h2{margin-top:2em;border-bottom:1px solid #ddd;padding-bottom:.35em}"
        ".table{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px}"
        "th,td{padding:8px;border-bottom:1px solid #ddd;text-align:left}th{background:#eef5f7}"
        "tbody tr:nth-child(even){background:#fafcfc}img{max-width:100%}figure{margin:24px 0}"
        "pre{overflow-x:auto;background:#f2f5f6;padding:18px;font-size:13px}li{margin:8px 0}"
        "a{color:#096b88}@media print{body{margin:0}h2{break-after:avoid}figure{break-inside:avoid}}"
    )
    return "<!doctype html><meta charset=utf-8><title>RATEL: open-field benchmark</title><style>" + css + "</style>" + "".join(parts)


def report(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    leaderboard, agr, human, frames = evaluate(out)
    dest = Path(__file__).resolve().parents[1] / "reports" / "open_field_expansion"
    dest.mkdir(parents=True, exist_ok=True)
    raw = leaderboard[(leaderboard.variant == "raw") & (leaderboard.cadence == "common_2.5hz")]
    wide = raw.pivot(index="tool", columns="behaviour", values="f1")
    wide["macro_behaviour_f1"] = wide[NAMES[1:]].mean(axis=1)
    wide = wide.sort_values("macro_behaviour_f1", ascending=False)
    fig, ax = plt.subplots(figsize=(10, max(4, len(wide)*.42)))
    matrix = wide[BEHAVIOURS].to_numpy()
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(4), ["Grooming", "Supported rear", "Unsupported rear", "All rearing"])
    ax.set_yticks(range(len(wide)), wide.index)
    for i in range(matrix.shape[0]):
        for j in range(4):
            ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center",
                    color="white" if matrix[i,j] > .6 else "#111111")
    ax.set_title("Held-out video behaviour F1; common 2.5 Hz scoring grid")
    fig.colorbar(im, ax=ax, label="F1")
    fig.tight_layout()
    fig.savefig(dest / "behaviour_f1.png", dpi=170)
    plt.close(fig)
    summary = pd.read_csv(out / "summary_scores_vs_each_observer.csv")
    top = wide.index[0]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, b in zip(axes, NAMES[1:]):
        g = summary[(summary.tool == top) & (summary.variant == "raw") &
                    (summary.behaviour == b) & (summary.metric == "time_s")]
        g = g.groupby("video_id")[["manual", "predicted"]].mean()
        ax.scatter(g.manual, g.predicted, color="#176b87", s=28)
        lim = max(g.to_numpy().max(), 1)*1.06
        ax.plot([0, lim], [0, lim], "--", color="#999999")
        ax.set(xlabel="Mean human duration (s)", ylabel="Predicted duration (s)", title=b.replace("_", " "))
    fig.suptitle(f"Duration agreement: {top}")
    fig.tight_layout()
    fig.savefig(dest / "duration_agreement.png", dpi=170)
    plt.close(fig)
    quality = pd.read_csv(out / "data_quality.csv")
    status = pd.read_csv(out / "movement_status.csv") if (out / "movement_status.csv").exists() else pd.DataFrame()
    total_frames = int(quality.frames.sum())
    incomplete = quality.incomplete_marker_observers.fillna("").ne("").sum()
    top_agr = agr[(agr.tool == top) & (agr.variant == "raw") & (agr.metric == "time_s")]
    findings = [f"- {r.behaviour.replace('_', ' ')}: duration MAE {r.mae:.1f} s/video; "
                f"bias {r.bias:+.1f} s; CCC {r.ccc:.3f}." for r in top_agr.itertuples()]
    software = []
    for path in sorted((out / "tools").glob("*.json")):
        d = json.loads(path.read_text())
        software.append(f"- {d['tool']} {d.get('version', '')}: {d['status']}.")
    movement_text = status.status.value_counts().to_dict() if not status.empty else "not run"
    scopes = pd.DataFrame([dict(configuration=name, tested_component=SCOPES.get(name, "See worker source."))
                           for name in wide.index])
    top_summary = agr[(agr.tool == top) & agr.metric.isin(["time_s", "bouts"])][
        ["variant", "behaviour", "metric", "mean_a", "mean_b", "mae", "bias", "ccc"]].copy()
    top_summary = top_summary.rename(columns={"mean_a": "human_mean", "mean_b": "predicted_mean"})
    top_summary.to_csv(dest / "top_model_summary.csv", index=False)
    tracking = pd.DataFrame()
    tracking_text = "Tracking comparison has not been run."
    if (out / "tracker_positions.csv").exists():
        positions = pd.read_csv(out / "tracker_positions.csv").sort_values("median_error_mm")
        tracking_text = (
            f"RATEL (OpenCV) and DLC were compared on {len(positions)} recordings. Median position disagreement "
            f"across videos was {positions.median_error_mm.median():.2f} mm; individual-video medians "
            f"ranged from {positions.median_error_mm.min():.2f} to {positions.median_error_mm.max():.2f} mm. "
            f"Both positions were available for {positions.valid_position_pairs.sum()/positions.compared_frames.sum():.3%} "
            "of the shared-window frames. "
            "DLC is an automated reference, so this is agreement between trackers, not manually established accuracy. "
            "Four pilot RATEL tracks were reused and 16 were newly computed."
        )
        comparison = pd.read_csv(out / "tracker_agreement.csv")
        tracking = comparison[comparison.metric.isin(
            ["total_distance_m", "immobile_time_s", "time_s", "entries"])][
            ["metric", "zone", "n", "bias", "mae", "ccc"]].fillna("")
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(positions.video_id, positions.median_error_mm, color="#176b87")
        ax.set(xlabel="Median position disagreement (mm; assumed arena scale)", ylabel="Recording",
               title="RATEL versus DLC: all 20 labelled recordings")
        ax.invert_yaxis()
        fig.tight_layout()
        fig.savefig(dest / "tracking_agreement.png", dpi=170)
        plt.close(fig)
    movement_details = ""
    if (out / "movement_metrics.csv").exists():
        movement_metrics = pd.read_csv(out / "movement_metrics.csv")
        fraction = movement_metrics.loc[movement_metrics.metric == "tracked_fraction", "value"]
        movement_details = (
            f"{movement_metrics.video_id.nunique()} recordings produced {len(movement_metrics):,} metric rows "
            "(13 behavioural quantities and four quality quantities per recording with these two zones). "
            f"Tracked fraction averaged {fraction.mean():.2%}; the minimum was {fraction.min():.2%}. "
            "This is a coverage diagnostic after cleaning, not verification of position accuracy. "
            "No treatment comparison was performed."
        )
    sensitivity = leaderboard[(leaderboard.tool == top) &
                             (leaderboard.cadence == "common_2.5hz")][
        ["variant", "behaviour", "f1", "f1_ci_low", "f1_ci_high"]].copy()
    intervals = raw[raw.tool == top][["behaviour", "precision", "recall", "f1", "f1_ci_low", "f1_ci_high"]]
    lines = [
        "# RATEL: expanded open-field benchmark", "",
        f"Evaluated {len(wide)} classifier configurations on 20 held-out mouse videos "
        f"({total_frames:,} frames in the shared windows at 25 fps; three human observers). "
        "Five fixed video folds use 16 videos for fitting and four for evaluation. "
        "Each video receives predictions only from a model that did not train on it.", "",
        "## Results", "", markdown_table(wide.round(3).reset_index()), "",
        f"The highest observed mean F1 for the three named behaviours was {top}. "
        "This ranking is exploratory; repeated tuning on this benchmark would require a fresh final test set.", "",
        "The macro score averages grooming, supported rearing and unsupported rearing. "
        "Combined rearing is shown separately because it is the union of the two rearing categories. "
        "F1 ranges from zero to one. It measures overlap of positive frame labels and does not establish "
        "agreement in the duration, number or timing of bouts.", "",
        "![Behaviour F1](behaviour_f1.png)", "",
        "### Leading configuration: uncertainty and measurement errors", "",
        markdown_table(intervals.round(3)), "", *findings, "",
        "![Duration agreement](duration_agreement.png)", "",
        "The table below compares each video with the mean of its three observers, using their "
        "original continuous annotation intervals. Positive bias means overestimation. CCC measures "
        "agreement on the identity line; a high correlation alone would not establish interchangeability.", "",
        markdown_table(top_summary.round(3)), "",
        "Raw frame predictions fragment behaviour into many short bouts. Smoothing is a fixed sensitivity "
        "check, not a validated final choice: it reduces fragmentation but can erase short real behaviours. "
        "Check the human/predicted bout means and absolute errors before using any output as an experimental endpoint.", "",
        "### Fixed scoring sensitivities", "", markdown_table(sensitivity.round(3)), "",
        "## Human reference", "",
        "Each observer is compared with the other two on frames where those two agree. "
        "These descriptive scores measure annotation disagreement, not an upper bound on model performance.", "",
        markdown_table(human[["behaviour", "precision", "recall", "f1"]].round(3)), "",
        "## What was actually tested", "", markdown_table(scopes), "",
        "The LabGym trial is a deliberately small native animation component, not its full detector/categorizer workflow. "
        "DeepEthogram uses its native frozen RGB encoder with a custom supervised head; optical flow and sequence "
        "training were not run. Both pixel trials use DLC-centred crops, include background and score at 2.5 Hz. "
        "They cannot establish the performance of the full software packages or of published pretrained models.", "",
        "## Methods and limits", "",
        "- ETH and BehaviorFlow contain 20 byte-identical recordings. Deduplication uses SHA-256 "
        "and original tracking keys before movement analysis; labelled folds contain unique originals.",
        "- Predictions are compared with majority-vote human labels. Ambiguous frames are excluded "
        "from frame scores. Summary scores use the average observer within each video, computed from original "
        "continuous annotations; the combined rearing bouts are merged across subtypes.",
        "- Main comparison: common 2.5 Hz scoring grid. Native-cadence scores are also saved. "
        "Time smoothing (0, 0.2 and 1.0 s) and fixed label shifts of -6/+6 original frames are reported separately.",
        "- Confidence intervals resample whole videos (2,000 bootstrap draws); frames are not independent "
        "biological replicates. Exact animal identities beyond original video IDs are not supplied here.",
        f"- {incomplete} videos have an observer with a missing end marker. Their shared scoring window "
        "ends at that observer's last annotated bout, so later absence is not inferred.",
        "- Pose rows exceed video header frame counts by six in this source. Frame-zero timestamps "
        "are preserved and trailing excess rows are clipped. Existing four-video position comparisons found "
        "zero best lag; six-frame shifts remain a sensitivity analysis.",
        "- Custom pose models use training frames sampled at 5 Hz and predict at 25 Hz. SimBA uses 221 "
        "engineered features from its native 8-body-part extractor. The two multiclass SimBA-feature baselines "
        "are custom classifiers; the separate native binary RF trial uses SimBA's classifier factory.",
        "- Imputation and standardization are fitted within each training fold. Whole-video pose cleaning and "
        "crop geometry use unlabelled recording information. Experimental treatment metadata is never a feature "
        "or a behaviour label.",
        "- Installed packages and completed classifier experiments are reported separately. A native import "
        "check does not establish behaviour detection accuracy.",
        "- Logistic regression/MLP iteration warnings in run logs indicate that those fixed-budget baselines "
        "may not have converged.",
        "- The 0.42 m arena side is an explicit distance-scaling assumption. Features use relative arena scale. "
        "Movement quality diagnostics do not establish manual position accuracy.",
        "- These public mouse recordings do not establish performance on the eventual rat recordings.",
        "- Defecation, urination, sniffing, freezing and stretch-attend labels are not supplied. Import and summary "
        "metrics are available, but detection accuracy cannot be measured without annotations.", "",
        "## Installed tool checks", "", *(software or ["No tool probes saved yet."]), "",
        "## Tracking agreement", "", tracking_text, "",
        markdown_table(tracking.round(3)) if not tracking.empty else "", "",
        "![Tracking agreement](tracking_agreement.png)" if not tracking.empty else "", "",
        "## Larger movement run", "", f"Recording statuses: {movement_text}.", "", movement_details, "",
        "The 432 catalogued open-field files reduce to 412 distinct recordings after known duplicates are removed. "
        "411 have usable DLC CSVs; one has a MAT reference that this adapter does not support. "
        "This larger run measures processing coverage and trajectory summaries; these videos do not supply "
        "independent human behaviour labels.", "",
        "## Practical next step", "",
        "Use the shared importer for manual grooming, rearing and point-event scoring now. "
        "The custom temporal models are candidates for assisted annotation, with human review. "
        "The present results do not justify replacing manual grooming or bout counts. "
        "Collect representative rat recordings, annotate a blinded subset and reserve an untouched final test set "
        "before choosing settings for the actual experiment.", "",
        "## Reproduce", "", "~~~powershell",
        ".venv/Scripts/python scripts/open_field_benchmark.py prepare",
        "output/tool_envs/simba/Scripts/python scripts/tool_worker.py simba",
        ".venv/Scripts/python scripts/open_field_benchmark.py classify",
        "output/tool_envs/simba/Scripts/python scripts/tool_worker.py simba-binary",
        ".venv/Scripts/python scripts/pixel_benchmark.py",
        "output/tool_envs/deepethogram/Scripts/python scripts/tool_worker.py deepethogram-rgb",
        "output/tool_envs/labgym/Scripts/python scripts/tool_worker.py labgym-animation --epochs 8",
        ".venv/Scripts/python scripts/open_field_benchmark.py movement",
        ".venv/Scripts/python scripts/benchmark_trackers.py --workers 2",
        ".venv/Scripts/python scripts/open_field_benchmark.py report", "~~~", "",
        "Setup, import formats and operational definitions are documented in workflow.md. "
        "Use a fresh output directory when changing experiment settings; completed predictions are cached. "
        "Exact package lists are in logs/packages_*.txt. provenance.json records input and script hashes.", "",
        f"Detailed CSVs, predictions, quality checks and fixed splits: {out.resolve()}", "",
        "## Primary sources", "",
        "- [ETH public recordings](https://zenodo.org/records/3608658) and "
        "[DLCAnalyzer annotations and source](https://github.com/ETHZ-INS/DLCAnalyzer).",
        "- [SimBA source and documentation](https://github.com/sgoldenlab/simba).",
        "- [LabGym source and documentation](https://github.com/umyelab/LabGym).",
        "- [DeepEthogram source and documentation](https://github.com/jbohnslav/deepethogram).",
        "- [BORIS event and ethogram documentation](https://www.boris.unito.it/user_guide/create_project/).", "",
    ]
    text = "\n".join(lines)
    (dest / "report.md").write_text(text, encoding="utf-8")
    (dest / "report.html").write_text(render_html(text.splitlines(), dest), encoding="utf-8")
    wide.to_csv(dest / "behaviour_comparison.csv")
    save_provenance(out, dest, wide.index)
    print(f"Report: {dest / 'report.html'}", flush=True)
    print(wide.round(3).to_string(), flush=True)
