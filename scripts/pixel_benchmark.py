"""Prepare pose-guided video crops for native LabGym/DeepEthogram component trials.

Both trials use identical animal-centred crops and whole-video held-out splits.
These bounded component experiments are not claims about a fully tuned native
pipeline. In particular, DeepEthogram optical flow and sequence training are
not part of its RGB feature probe.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from behaviour.adapters.deeplabcut import read_dlc

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "output/open_field_benchmark"


def prepare(out):
    cv2.setNumThreads(2)
    records = json.loads((out / "recordings.json").read_text())
    dest = out / "pixel_cache"
    dest.mkdir(exist_ok=True)
    specification = dict(sample_stride_frames=10, sequence_frames=15, sequence_size_px=16,
                         rgb_size_px=64, centering="DLC bodycentre", background="included",
                         sequence_alignment="15 frames ending at target frame", fps=25)
    (dest / "specification.json").write_text(json.dumps(specification, indent=2), encoding="utf-8")
    for rec in records:
        vid = rec["video_id"]
        path = dest / f"{vid}.npz"
        if path.exists():
            continue
        started = time.perf_counter()
        pose = read_dlc(rec["pose_path"])
        centers = pd.DataFrame(pose.loc[:, [("bodycentre", "x"), ("bodycentre", "y")]].values).ffill().bfill().values
        lengths = np.linalg.norm(pose.loc[:, [("nose", "x"), ("nose", "y")]].values -
                                 pose.loc[:, [("tailbase", "x"), ("tailbase", "y")]].values, axis=1)
        size = int(np.clip(np.nanmedian(lengths)*2.6, 80, 200))
        with np.load(out / "cache" / f"{vid}.npz") as z:
            wanted = z["frame"][::10]
            times = z["t"][::10]
            y = z["y"][::10]
        wanted_set = set(wanted.tolist())
        cap = cv2.VideoCapture(rec["video_path"])
        sequence, animations, rgb, actual = deque(maxlen=15), [], [], []
        frame = 0
        while frame <= wanted[-1]:
            ok, image = cap.read()
            if not ok:
                break
            cx, cy = centers[min(frame, len(centers)-1)]
            crop = cv2.getRectSubPix(image, (size, size), (float(cx), float(cy)))
            gray = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (16, 16), interpolation=cv2.INTER_AREA)
            if not sequence:
                sequence.extend([gray]*15)
            sequence.append(gray)
            if frame in wanted_set:
                animations.append(np.stack(sequence))
                rgb.append(cv2.cvtColor(cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
                actual.append(frame)
            frame += 1
        cap.release()
        if actual != wanted.tolist():
            raise ValueError(f"{vid}: decoded {len(actual)} of {len(wanted)} requested crop frames")
        np.savez_compressed(path, animation=np.stack(animations), rgb=np.stack(rgb), t=times, y=y,
                            frame=np.array(actual), crop_size_px=size)
        print(f"Pixel crops {vid}: {len(actual)} sequences in {time.perf_counter()-started:.1f}s", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    a = ap.parse_args()
    prepare(a.output)


if __name__ == "__main__":
    main()
