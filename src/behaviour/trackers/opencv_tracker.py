"""RATEL's centroid tracker (OpenCV, no learning).

Pipeline per frame:
  grey + Gaussian blur -> |frame - background| -> threshold -> arena mask
  -> morphological open/close -> largest blob >= min_area_px -> centroid

A frame counts as lost (NaN) when the blob is larger than ``max_area_px`` or
when more than ``max_foreground_frac`` of the arena changed at once: lights
switching, the lid closing or the experimenter's hand.

The background is the per-pixel median of frames sampled across the video,
which removes the animal as long as it does not sit in one spot for most of
the recording (pass an empty-arena image otherwise).  Using |difference|
works for light-on-dark and dark-on-light animals alike.

Besides position it records ``motion_px``: pixels that changed since the
previous frame inside the (dilated) animal blob.  That is the raw signal for
forced-swim immobility scoring.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from ..zones import Zone, arena_mask

TOOL = "opencv"


@dataclass
class TrackerParams:
    blur: int = 5                  # Gaussian kernel (odd, px); 0/1 disables
    diff_threshold: float = 30.0   # grey levels of |frame - background|
    min_area_px: int = 50
    max_area_px: int | None = None      # larger blobs are not the animal
    max_foreground_frac: float = 0.2    # more of the arena changed -> scene change, frame lost
    open_kernel: int = 3           # removes speckle
    close_kernel: int = 7          # fills holes in the body
    bg_frames: int = 60
    bg_start_s: float | None = None
    bg_end_s: float | None = None
    motion_threshold: float = 15.0
    motion_dilate_px: int = 5

    @classmethod
    def from_dict(cls, d: dict) -> "TrackerParams":
        names = {f.name for f in fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise ValueError(f"Unknown [tracker] options {sorted(unknown)}; valid: {sorted(names)}")
        return cls(**d)


def _prep(frame: np.ndarray, blur: int) -> np.ndarray:
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    if blur and blur > 1:
        k = blur | 1
        g = cv2.GaussianBlur(g, (k, k), 0)
    return g


def _video_info(cap: cv2.VideoCapture) -> tuple[float, int, int, int]:
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0:
        raise ValueError("Video reports no frame rate")
    return fps, n, w, h


def estimate_background(video: str | Path, p: TrackerParams) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(video)
    fps, n, _, _ = _video_info(cap)
    a = int((p.bg_start_s or 0.0) * fps)
    b = int(p.bg_end_s * fps) if p.bg_end_s else n - 1
    idx = np.unique(np.linspace(a, max(a, b), p.bg_frames).astype(int))
    stack = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if ok:
            stack.append(_prep(frame, p.blur))
    cap.release()
    if not stack:
        raise ValueError(f"Could not read frames from {video}")
    return np.median(np.stack(stack), axis=0).astype(np.uint8)


def track(video: str | Path, zones: list[Zone], params: TrackerParams | None = None,
          mask_margin_px: float = 20.0, background: np.ndarray | None = None,
          qc_video: str | Path | None = None, progress: bool = True) -> pd.DataFrame:
    """Track one animal; returns the canonical per-frame trajectory table."""
    p = params or TrackerParams()
    bg = background if background is not None else estimate_background(video, p)
    cap = cv2.VideoCapture(str(video))
    fps, n_frames, w, h = _video_info(cap)
    mask = arena_mask(zones, (h, w), mask_margin_px) if zones else np.full((h, w), 255, np.uint8)
    max_fg = p.max_foreground_frac * np.count_nonzero(mask)

    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p.open_kernel, p.open_kernel)) if p.open_kernel > 1 else None
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p.close_kernel, p.close_kernel)) if p.close_kernel > 1 else None
    k_motion = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * p.motion_dilate_px + 1,) * 2)

    writer = None
    if qc_video:
        writer = cv2.VideoWriter(str(qc_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        trail: list[tuple[int, int]] = []

    rows = []
    prev = None
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        g = _prep(frame, p.blur)
        fg = (cv2.absdiff(g, bg) > p.diff_threshold).astype(np.uint8) * 255
        fg &= mask
        if k_open is not None:
            fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k_open)
        if k_close is not None:
            fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k_close)
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        x = y = area = motion = np.nan
        blob = max(contours, key=cv2.contourArea) if contours else None
        if blob is not None and (np.count_nonzero(fg) > max_fg
                                 or (p.max_area_px and cv2.contourArea(blob) > p.max_area_px)):
            blob = None
        if blob is not None and cv2.contourArea(blob) >= p.min_area_px:
            m = cv2.moments(blob)
            if m["m00"] > 0:
                x, y, area = m["m10"] / m["m00"], m["m01"] / m["m00"], m["m00"]
                if prev is not None:
                    body = np.zeros_like(fg)
                    cv2.drawContours(body, [blob], -1, 255, -1)
                    body = cv2.dilate(body, k_motion)
                    changed = cv2.absdiff(g, prev) > p.motion_threshold
                    motion = float(np.count_nonzero(changed & (body > 0)))
        rows.append((i, i / fps, x, y, area, motion))
        prev = g

        if writer is not None:
            _draw_qc(frame, zones, x, y, trail, int(2 * fps))
            writer.write(frame)
        i += 1
        if progress and n_frames and i % max(n_frames // 10, 1) == 0:
            print(f"  {Path(video).name}: {100 * i // n_frames}%", flush=True)

    cap.release()
    if writer is not None:
        writer.release()
    return pd.DataFrame(rows, columns=["frame", "time_s", "x_px", "y_px", "area_px", "motion_px"])


def _draw_qc(frame, zones, x, y, trail, trail_len):
    palette = [(0, 0, 255), (0, 200, 0), (255, 0, 0), (0, 200, 200), (200, 0, 200)]
    for k, z in enumerate(zones):
        z.draw(frame, palette[k % len(palette)], 1)
    if np.isfinite(x):
        trail.append((int(round(x)), int(round(y))))
        del trail[:-trail_len]
        if len(trail) > 1:
            cv2.polylines(frame, [np.array(trail, np.int32).reshape(-1, 1, 2)], False, (255, 255, 255), 1)
        cv2.circle(frame, trail[-1], 5, (0, 0, 0), 2)
        cv2.circle(frame, trail[-1], 3, (255, 255, 255), -1)
    else:
        cv2.putText(frame, "LOST", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)


def preview(video: str | Path, zones: list[Zone], out_png: str | Path, time_s: float = 10.0,
            mask_margin_px: float = 20.0) -> None:
    """Save one frame with zones and the tracking mask outline drawn on it,
    to check that config coordinates line up with this camera."""
    cap = cv2.VideoCapture(str(video))
    fps, n, w, h = _video_info(cap)
    frame_idx = int(time_s * fps)
    if n:
        frame_idx = min(frame_idx, n - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"Could not read frame at {time_s}s from {video}")
    palette = [(0, 0, 255), (0, 200, 0), (255, 0, 0), (0, 200, 200), (200, 0, 200)]
    for k, z in enumerate(zones):
        z.draw(frame, palette[k % len(palette)], 2)
        if isinstance(getattr(z, "cx", None), float):
            cv2.putText(frame, z.name, (int(z.cx) + 4, int(z.cy - z.r) + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, palette[k % len(palette)], 1)
    mask = arena_mask(zones, (h, w), mask_margin_px)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_png), frame)
