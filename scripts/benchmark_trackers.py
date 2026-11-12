"""Extend RATEL vs DLC trajectory agreement to all 20 labelled OFT videos."""

from __future__ import annotations

import argparse
import contextlib
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from behaviour.adapters.deeplabcut import trajectory
from behaviour.compare import compare
from behaviour.config import load_config
from behaviour.metrics import metrics_for_trajectory
from behaviour.trackers.opencv_tracker import TrackerParams, track

ROOT = Path(__file__).resolve().parents[1]


def run_one(args):
    rec, folder = args
    out = Path(folder)
    cv2.setNumThreads(2)
    vid = rec["video_id"]
    cfg = load_config(out / "configs" / f"{vid}.toml")
    cache = out / "tracks/opencv" / f"{vid}.csv"
    legacy = ROOT / "output/zenodo_pilot/tracks/opencv" / f"{vid}.csv"
    started = time.perf_counter()
    origin = "benchmark_cache"
    if cache.exists():
        oc = pd.read_csv(cache)
    elif legacy.exists():
        oc = pd.read_csv(legacy)
        oc.to_csv(cache, index=False)
        origin = "existing_four_video_pilot"
    else:
        origin = "new_native_opencv_run"
        with (out / "logs" / f"opencv_{vid}.log").open("w", encoding="utf-8") as log:
            with contextlib.redirect_stdout(log):
                oc = track(rec["video_path"], cfg.zones, TrackerParams.from_dict(cfg.tracker),
                           cfg.mask_margin_px)
        oc.to_csv(cache, index=False)
    dlc = trajectory(rec["pose_path"], rec["fps"])
    metrics = pd.concat([metrics_for_trajectory(oc, cfg, vid, "opencv"),
                         metrics_for_trajectory(dlc, cfg, vid, "dlc")], ignore_index=True)
    n = min(len(oc), len(dlc))
    keep = (oc.time_s.to_numpy()[:n] >= rec["start_s"]) & (oc.time_s.to_numpy()[:n] < rec["end_s"])
    error = np.hypot(oc.x_px.to_numpy()[:n]-dlc.x_px.to_numpy()[:n],
                     oc.y_px.to_numpy()[:n]-dlc.y_px.to_numpy()[:n])[keep]
    row = dict(video_id=vid, source=origin, seconds=time.perf_counter()-started,
               valid_position_pairs=int(np.isfinite(error).sum()), compared_frames=len(error),
               median_error_px=float(np.nanmedian(error)), p95_error_px=float(np.nanpercentile(error, 95)),
               median_error_mm=float(np.nanmedian(error)/rec["px_per_mm"]))
    return row, metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=ROOT / "output/open_field_benchmark")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    for d in ["tracks/opencv", "logs"]:
        (a.output / d).mkdir(parents=True, exist_ok=True)
    records = json.loads((a.output / "recordings.json").read_text())
    positions, metrics, failures = [], [], []
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures = {pool.submit(run_one, (rec, str(a.output))): rec for rec in records}
        for future in as_completed(futures):
            rec = futures[future]
            try:
                row, long = future.result()
                positions.append(row)
                metrics.append(long)
                print(f"{rec['video_id']}: {row['source']}; median error {row['median_error_px']:.2f}px; "
                      f"{row['seconds']:.1f}s", flush=True)
            except Exception as e:
                failures.append(dict(video_id=rec["video_id"], reason=f"{type(e).__name__}: {e}"))
                print(f"FAILED {rec['video_id']}: {e}", flush=True)
            pd.DataFrame(positions).to_csv(a.output / "tracker_positions.csv", index=False)
            if metrics:
                pd.concat(metrics, ignore_index=True).to_csv(a.output / "tracker_metrics.csv", index=False)
    if metrics:
        compare(pd.concat(metrics, ignore_index=True), reference="dlc").to_csv(
            a.output / "tracker_agreement.csv", index=False)
    (a.output / "tracker_failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    if failures:
        raise SystemExit(f"{len(failures)} videos failed; see tracker_failures.json")


if __name__ == "__main__":
    main()
