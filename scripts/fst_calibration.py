"""Forced swim immobility: rater agreement and calibration of `behaviour.fst` scoring.

Data: Sturman et al. (2020) FST labels ("Floating" bouts, 4 raters, 10 videos) and
OpenCV tracks of the top view (motion_px / area_px): FST_1 from scripts/zenodo_report.py
(reports/zenodo_pilot/fst/tracks, read-only here), the others from
`behav track ... -c reports/zenodo_pilot/fst/fst.toml -o output/fst_calibration/tracks`.

1. Inter-rater agreement on all labelled videos: immobile seconds per rater
   (ICC(A,1) over the 4 raters, pairwise Lin's CCC), and frame-level Cohen's kappa
   between raters on every video we also have tracks for.
2. Parameter sweep (motion_threshold x smooth_s x min_immobile_s) scored as the mean
   frame-level kappa against each rater.  With only one labelled tracked video
   this is in-sample (descriptive); with more, each video is scored with
   parameters chosen on the others (leave-one-video-out).

Usage:  .venv/Scripts/python scripts/fst_calibration.py
"""

from __future__ import annotations

from itertools import combinations, product
from pathlib import Path

import numpy as np
import pandas as pd

from behaviour.compare import lins_ccc
from behaviour.config import load_config
from behaviour.fst import clip, score_immobility, total, union
from behaviour.trajectory import load_trajectory

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "videos" / "zenodo_3608658" / "labels" / "AllFSTLables.csv"
FST_CFG = ROOT / "reports" / "zenodo_pilot" / "fst" / "fst.toml"
OUT = ROOT / "output" / "fst_calibration"
TRACK_DIRS = [OUT / "tracks", ROOT / "reports" / "zenodo_pilot" / "fst" / "tracks"]

THRESHOLDS = np.round(np.geomspace(0.005, 1.0, 47), 4)
SMOOTH_S = [0.2, 0.5, 1.0]
MIN_IMMOBILE_S = [0.5, 1.0, 2.0]


def icc_a1_k(y: np.ndarray) -> float:
    """ICC(A,1) for an n subjects x k raters matrix (McGraw & Wong 1996)."""
    n, k = y.shape
    grand = y.mean()
    ss_r = k * ((y.mean(1) - grand) ** 2).sum()
    ss_c = n * ((y.mean(0) - grand) ** 2).sum()
    ss_e = ((y - grand) ** 2).sum() - ss_r - ss_c
    msr, msc, mse = ss_r / (n - 1), ss_c / (k - 1), ss_e / ((n - 1) * (k - 1))
    return (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)


def kappa(a: np.ndarray, b: np.ndarray) -> float:
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


def read_labels():
    lab = pd.read_csv(LABELS, sep=";")
    windows = {}
    for vid, g in lab[lab.type.str.match(r"Start.?End")].groupby("ID"):
        per = g.groupby("Experimenter")["from"].agg(["min", "max", "size"])
        start = per["min"].median()
        ends = per.loc[per["size"] >= 2, "max"]  # raters who marked both start and end
        windows[vid] = (float(start), float(ends.median() - start))
    bouts = {(vid, r): union(list(zip(g["from"], g["to"])))
             for (vid, r), g in lab[lab.type == "Floating"].groupby(["ID", "Experimenter"])}
    raters = sorted(lab.Experimenter.unique())
    return windows, bouts, raters


def mask_from_bouts(t_abs: np.ndarray, iv) -> np.ndarray:
    m = np.zeros(len(t_abs), bool)
    for a, b in iv:
        m |= (t_abs >= a) & (t_abs < b)
    return m


def find_track(vid: str) -> Path | None:
    return next((d / f"{vid}.csv" for d in TRACK_DIRS if (d / f"{vid}.csv").exists()), None)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    windows, bouts, raters = read_labels()
    videos = sorted(windows, key=lambda v: int(v.split("_")[1]))

    # ---- 1a. immobile seconds per rater, all labelled videos
    secs = pd.DataFrame({r: [total(clip(bouts.get((v, r), []), windows[v][0], sum(windows[v])))
                             for v in videos] for r in raters}, index=videos)
    secs["window_s"] = [windows[v][1] for v in videos]
    y = secs[raters].to_numpy()
    pair_ccc = {f"{a}-{b}": lins_ccc(secs[a].to_numpy(), secs[b].to_numpy()) for a, b in combinations(raters, 2)}
    secs.round(1).to_csv(OUT / "rater_immobile_seconds.csv")

    print("Immobile seconds per rater (whole labelled window):")
    print(secs.round(1).to_string())
    print(f"\nICC(A,1) over {len(raters)} raters, {len(videos)} videos: {icc_a1_k(y):.3f}")
    print("Pairwise CCC: " + ", ".join(f"{k} {v:.2f}" for k, v in pair_ccc.items()))
    for drop in raters:
        keep = [r for r in raters if r != drop]
        print(f"  ICC(A,1) without {drop}: {icc_a1_k(secs[keep].to_numpy()):.3f}")
    spread = y.max(1) - y.min(1)
    print(f"Max - min between raters per video: median {np.median(spread):.0f} s, range {spread.min():.0f}-{spread.max():.0f} s")

    # ---- 1b/2. frame level, on videos with both labels and tracks
    cfg = load_config(FST_CFG)
    tracked = [v for v in videos if find_track(v)]
    print(f"\nLabelled videos with tracks: {tracked}")
    rows, rater_kappas = [], []
    for vid in tracked:
        start, dur = windows[vid]
        df = load_trajectory(find_track(vid))
        w = df[(df.time_s >= start) & (df.time_s < start + dur)]
        t_abs = w.time_s.to_numpy()
        t = t_abs - start
        rm = {r: mask_from_bouts(t_abs, bouts.get((vid, r), [])) for r in raters}
        for a, b in combinations(raters, 2):
            rater_kappas.append(dict(video_id=vid, pair=f"{a}-{b}", kappa=kappa(rm[a], rm[b])))
        print(f"{vid}: window {start:.1f}+{dur:.1f} s, tracked in window {w.x_px.notna().mean():.1%}")
        for thr, sm, mi in product(THRESHOLDS, SMOOTH_S, MIN_IMMOBILE_S):
            cfg.fst.motion_threshold, cfg.fst.smooth_s, cfg.fst.min_immobile_s = float(thr), sm, mi
            dur_f, _, imm = score_immobility(t, w.motion_px.to_numpy(), w.area_px.to_numpy(), cfg, dur)
            ks = [kappa(imm, rm[r]) for r in raters]
            rows.append(dict(video_id=vid, motion_threshold=thr, smooth_s=sm, min_immobile_s=mi,
                             mean_kappa=np.mean(ks), min_kappa=np.min(ks),
                             immobile_s=float(dur_f[imm].sum()),
                             raters_mean_s=float(np.mean([dur_f[rm[r]].sum() for r in raters])),
                             raters_min_s=float(np.min([dur_f[rm[r]].sum() for r in raters])),
                             raters_max_s=float(np.max([dur_f[rm[r]].sum() for r in raters]))))
    sweep = pd.DataFrame(rows)
    sweep.to_csv(OUT / "sweep.csv", index=False)
    rk = pd.DataFrame(rater_kappas)
    rk.to_csv(OUT / "rater_kappa.csv", index=False)

    print("\nFrame-level kappa between raters (6 pairs per video):")
    print(rk.groupby("video_id", sort=False)["kappa"].agg(["min", "median", "max"]).round(3).to_string())

    print("\nBest parameter sets per video (mean kappa vs each rater):")
    for vid, g in sweep.groupby("video_id", sort=False):
        print(g.sort_values("mean_kappa", ascending=False).head(3).round(3).to_string(index=False))
        d = g[(g.smooth_s == 0.2) & (g.min_immobile_s == 1.0)].iloc[(g[(g.smooth_s == 0.2) & (g.min_immobile_s == 1.0)].motion_threshold - 0.05).abs().argmin()]
        best = g.mean_kappa.max()
        ok = g[(g.smooth_s == 0.5) & (g.min_immobile_s == 1.0) & (g.mean_kappa >= best - 0.05)].motion_threshold
        print(f"thresholds within 0.05 kappa of best (smooth 0.5, min 1.0): {ok.min():.3f}-{ok.max():.3f}")
        print(f"current default (0.05, smooth 0.2, min 1.0): mean kappa {d.mean_kappa:.3f}, "
              f"immobile {d.immobile_s:.0f} s vs raters' mean {d.raters_mean_s:.0f} s")

    key = ["motion_threshold", "smooth_s", "min_immobile_s"]
    if len(tracked) >= 2:
        lovo = []
        for vid in tracked:
            train = sweep[sweep.video_id != vid].groupby(key)["mean_kappa"].mean().idxmax()
            test = sweep[(sweep.video_id == vid) & (sweep[key].apply(tuple, axis=1) == train)].iloc[0]
            rkv = rk[rk.video_id == vid].kappa
            lovo.append(dict(video_id=vid, threshold=float(train[0]), smooth_s=float(train[1]),
                             min_bout_s=float(train[2]), tool_kappa=test.mean_kappa,
                             best_possible=sweep[sweep.video_id == vid].mean_kappa.max(),
                             rater_kappa_min=rkv.min(), rater_kappa_median=rkv.median(),
                             tool_s=test.immobile_s, raters_mean_s=test.raters_mean_s,
                             raters_min_s=test.raters_min_s, raters_max_s=test.raters_max_s))
        lovo = pd.DataFrame(lovo)
        lovo.to_csv(OUT / "leave_one_video_out.csv", index=False)
        print("\nLeave-one-video-out (parameters chosen on the other videos):")
        print(lovo.round(3).to_string(index=False))
        inside = ((lovo.tool_s >= lovo.raters_min_s) & (lovo.tool_s <= lovo.raters_max_s)).sum()
        print(f"held-out kappa: mean {lovo.tool_kappa.mean():.3f} vs raters' median pairwise "
              f"{lovo.rater_kappa_median.mean():.3f}")
        print(f"held-out immobile seconds inside the raters' range: {inside}/{len(lovo)}; CCC tool vs "
              f"raters' mean: {lins_ccc(lovo.raters_mean_s.to_numpy(), lovo.tool_s.to_numpy()):.3f}")
        pooled = sweep.groupby(key)["mean_kappa"].agg(["mean", "min"]).sort_values("mean", ascending=False)
        print("\nBest settings pooled over all tracked videos (mean / worst-video kappa):")
        print(pooled.head(5).round(3).to_string())

    # kappa vs threshold curve at each smoothing (min bout 1 s), for plotting
    curve = sweep[sweep.min_immobile_s == 1.0].pivot_table(index="motion_threshold", columns=["video_id", "smooth_s"],
                                                            values="mean_kappa")
    curve.to_csv(OUT / "kappa_vs_threshold.csv")


if __name__ == "__main__":
    main()
