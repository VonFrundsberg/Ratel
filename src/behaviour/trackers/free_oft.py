"""Headless open-field trackers exporting source-frame pixel coordinates."""

from __future__ import annotations

from contextlib import chdir
from pathlib import Path
import shutil
import sys
import tempfile

import cv2
import numpy as np
import pandas as pd

from ..zones import arena_mask


def eztrack(video, mask, frames, percentile):
    # The verified installer enables the Bokeh backend for headless use.
    import LocationTracking_Functions as ez

    height, width = mask.shape
    size = (width // 2, height // 2)
    small = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    info = dict(file=video.name, fpath=str(video), dpath=str(video.parent),
                start=0, end=frames, dsmpl=0.5, region_names=None,
                mask={"mask": small == 0}, stretch={"width": 1, "height": 1})
    reference_frames = np.linspace(0, max(0, frames - 2), min(frames, 51), dtype=int)
    info["reference"], _ = ez.Reference(info, frames=reference_frames)
    params = dict(method="light", loc_thresh=percentile, use_window=False,
                  window_weight=0, window_size=0, rmv_wire=False)
    raw = ez.TrackLocation(info, params)
    if len(raw) != frames:
        raise ValueError(f"ezTrack decoded {len(raw)} of {frames} expected frames")
    return raw.X.to_numpy() * width / size[0], raw.Y.to_numpy() * height / size[1]


def animapp(video, mask, frames, lower, upper, scratch):
    from animapp import animapp as engine

    left, top, width, height = cv2.boundingRect(mask)
    settings = dict(leftcorner=(left, top), rightcorner=(left + width, top + height),
                    angle=0, ratio=1, startframe=0, hue=(lower[0], upper[0]),
                    sat=(lower[1], upper[1]), val=(lower[2], upper[2]))
    x = np.full(frames, np.nan)
    y = np.full(frames, np.nan)
    # The upstream worker writes next to its input and reads settings from its cwd.
    with tempfile.TemporaryDirectory(dir=scratch) as folder, chdir(folder):
        local = Path("input" + video.suffix)
        shutil.copyfile(video, local)
        Path("settings.yaml").write_text("".join(f"{k}: {v}\n" for k, v in settings.items()))
        engine.filenames = [str(local)]
        engine.stop = False
        engine.ThresholdThread().run()
        output = Path(str(local) + ".csv")
        if output.stat().st_size:
            raw = pd.read_csv(output, header=None, usecols=[0, 1, 2], names=["x", "y", "i"])
            raw = raw[(raw.x != 0) | (raw.y != 0)]  # upstream zero-moment sentinel
            # Frame zero initializes the worker. Its first output (count=0) is source frame 1.
            x[raw.i + 1] = raw.x + left
            y[raw.i + 1] = raw.y + top
    return x, y


def run(video, config, engine, out, percentile, lower, upper, stride=None, lab_profile=False):
    video = Path(video).absolute()
    out = Path(out).absolute()
    if out.suffix.lower() != ".csv":
        raise ValueError("Output must be a trajectory .csv")
    cap = cv2.VideoCapture(str(video))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        # Some MP4 headers overstate the number of decodable frames.
        ok, first = cap.read()
        frames = int(ok)
        while cap.grab():
            frames += 1
        if not ok or fps <= 0 or frames < 2:
            raise ValueError(f"Cannot read video metadata and first frame: {video}")
    finally:
        cap.release()
    mask = arena_mask(config.zones, first.shape[:2], config.mask_margin_px)
    if not mask.any():
        raise ValueError("The configured arena misses the entire video frame")
    out.parent.mkdir(parents=True, exist_ok=True)
    if stride is not None and (engine != "superanimal" or stride < 1):
        raise ValueError("--stride must be positive and is only available with SuperAnimal")
    if lab_profile and engine not in ("fasttrack", "tracktor"):
        raise ValueError("--lab-profile applies to Tracktor and FastTrack only")
    step = 1
    if engine in ("eztrack", "animapp"):
        if engine == "eztrack":
            if not 0 < percentile < 100:
                raise ValueError("ezTrack percentile must be between 0 and 100")
            x, y = eztrack(video, mask, frames, percentile)
        else:
            if not all(0 <= a <= b <= limit for a, b, limit in zip(lower, upper, (179, 255, 255))):
                raise ValueError("Invalid AnimApp HSV interval")
            x, y = animapp(video, mask, frames, lower, upper, out.parent)
        df = pd.DataFrame(dict(frame=np.arange(frames), x_px=x, y_px=y))
    else:
        from .additional_oft import fasttrack_frames, superanimal_frames, tracktor_frames

        assets = Path(sys.prefix) / "share" / "ratel-oft"
        if not assets.is_dir():
            raise FileNotFoundError("Install the OFT extras and run scripts/install_additional_oft.py")
        if engine == "fasttrack":
            df = fasttrack_frames(video, mask, assets, out.parent, frames, lab_profile)
        elif engine == "tracktor":
            df = tracktor_frames(video, mask, assets, lab_profile)
        elif engine == "superanimal":
            step = stride or 30
            df = superanimal_frames(video, mask, assets, step)
        else:
            raise ValueError(f"Unknown engine: {engine}")
    if not np.array_equal(df.frame, np.arange(0, frames, step)):
        raise ValueError(f"{engine} returned unexpected source frame indices")
    df.insert(1, "time_s", df.frame / fps)
    with tempfile.NamedTemporaryFile(dir=out.parent, delete=False, suffix=".tmp", mode="w", newline="") as f:
        partial = Path(f.name)
        df.to_csv(f, index=False)
    partial.replace(out)
    print(f"{video.name}: {df.x_px.notna().mean():.1%} evaluated frames with coordinates -> {out}")
