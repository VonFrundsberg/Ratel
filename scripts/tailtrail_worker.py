"""One recording in an isolated process, using TailTrail's own bundled engine.

Called by tailtrail_batch.py with the prepared Python 3.13 interpreter. Only the
GUI value containers and export destination are replaced; tracking, background
training, ROI rules, distance calculations and summary export use vendor code.
"""

from __future__ import annotations

import io
import json
import math
import pickle
import sys
import time
from pathlib import Path


class Value:
    """The get/set interface used by the engine's tkinter variables, without Tk."""

    def __init__(self, value=0):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class RowCollector:
    def add_row(self, data):
        self.row = data


def load_settings(path: Path):
    from src.model.region_of_interest import RegionOfInterest

    class SettingsUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if (module, name) == ("src.model.region_of_interest", "RegionOfInterest"):
                return RegionOfInterest
            if (module, name) in {
                ("numpy._core.multiarray", "_reconstruct"),
                ("numpy.core.multiarray", "_reconstruct"),
                ("numpy", "ndarray"), ("numpy", "dtype"),
            }:
                return super().find_class(module, name)
            raise pickle.UnpicklingError(f"Refusing settings class {module}.{name}")

    raw = SettingsUnpickler(io.BytesIO(path.read_bytes())).load()
    if not isinstance(raw, dict) or not raw.get("rois"):
        raise ValueError("Save a TailTrail settings file with at least one ROI first")
    scale = float(raw.get("scale", 0))
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("TailTrail scale must be positive pixels per metre")
    names = [roi.name.casefold() for roi in raw["rois"]]
    if len(set(names)) != len(names):
        raise ValueError("TailTrail ROI names must be unique")
    for roi in raw["rois"]:
        if not roi.contour or not roi.hysteresis_contour:
            raise ValueError(f"ROI {roi.name} has no saved contours; save its settings in TailTrail again")
    return raw


def save_png(path: Path, frame):
    import cv2

    success, encoded = cv2.imencode(".png", frame)
    if not success:
        raise RuntimeError(f"Could not encode {path.name}")
    encoded.tofile(str(path))


def run(job: dict):
    import cv2
    import src.tracker as tracking
    import src.video_processor as processing
    from src import data_export
    from src.params_storage import ParamsStorage

    # The engine never creates widgets. Its parameters only need get/set.
    for module in (tracking, processing):
        for name in ("StringVar", "IntVar", "DoubleVar", "BooleanVar"):
            setattr(module, name, Value)
    cv2.setNumThreads(job["threads"])
    video, output = Path(job["video"]), Path(job["result_dir"])
    output.mkdir(parents=True, exist_ok=True)
    raw = load_settings(Path(job["settings_snapshot"]))
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open recording: {video}")
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps, total = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if width <= 0 or height <= 0 or not math.isfinite(fps) or fps <= 0 or total <= 0:
            raise ValueError("Recording has invalid dimensions, frame count or frame rate")
        start = math.ceil(job["start_s"] * fps - 1e-8)
        requested_end = job["end_s"]
        if requested_end is not None and requested_end > total / fps + 1 / fps:
            raise ValueError(f"Requested end {requested_end}s exceeds recording length {total / fps:.3f}s")
        end = total if requested_end is None else min(total, math.ceil(requested_end * fps - 1e-8))
        if not 0 <= start < end <= total:
            raise ValueError(f"Empty or invalid frame range [{start}, {end}) of {total}")
        print(f"Video: {width}x{height}, {fps:g} fps; frames [{start}, {end})", flush=True)
        storage, collector = ParamsStorage(), RowCollector()
        width_var, height_var, fps_var = Value(width), Value(height), Value(fps)
        rois, scale, destination = Value(raw["rois"]), Value(raw["scale"]), Value(str(output))
        tracker = tracking.Tracker(storage, destination, collector, rois, width_var, height_var, fps_var, scale)
        processor = processing.VideoProcessor(None, width_var, height_var, fps_var, storage, rois, tracker, destination)
        for key, variable in storage.params.items():
            if key in raw:
                variable.set(raw[key])
        # Saved paths must never redirect this run into the user's GUI exports.
        destination.set(str(output))
        for name in ("export_tracking", "export_tracking_with_background", "export_heatmap",
                     "export_heatmap_with_background", "export_tracking_video", "export_heatmap_video"):
            getattr(processor, name).set(False)
        processor.silent = True
        processor.update_translation()
        processor.update_mask()
        processor.create_background_subtraction()
        processor.create_heatmap()
        tracker.reset()

        class TrainingCapture:
            reads = 0

            def set(self, prop, value):
                if not cap.set(prop, value):
                    raise RuntimeError(f"Video seeking failed during training at frame {value}")
                return True

            def read(self):
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("Video decoding failed during background training")
                self.reads += 1
                if self.reads % 100 == 0:
                    print(f"Background: {self.reads} samples", flush=True)
                return ok, frame

        training = TrainingCapture()
        started = time.monotonic()
        # This is exactly the GUI's Train algorithm call: samples from the
        # middle 80% of the FULL recording, independent of the analysis window.
        processor.train_background_subtraction(training, total)
        print(f"Background trained in {time.monotonic() - started:.1f}s", flush=True)
        if not cap.set(cv2.CAP_PROP_POS_FRAMES, start) or abs(cap.get(cv2.CAP_PROP_POS_FRAMES) - start) > 0.01:
            raise RuntimeError(f"Could not seek to the requested start frame {start}")
        tracker.reset()
        tracker.active = True
        last_update = time.monotonic()
        for frame_number in range(start, end):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Video decoding failed at frame {frame_number}; result not marked complete")
            processor.pipeline(frame)
            if frame_number == start and job["images"]:
                preview = cv2.cvtColor(processor.masked_frame, cv2.COLOR_BGR2RGB)
                save_png(output / "roi_preview.png", cv2.cvtColor(processor.draw_rois(preview), cv2.COLOR_RGB2BGR))
            if time.monotonic() - last_update >= 10:
                percent = 100 * (frame_number - start + 1) / (end - start)
                print(f"Tracking: {percent:.1f}%", flush=True)
                last_update = time.monotonic()
        tracker.active = False
        if tracker.frames_count != end - start:
            raise RuntimeError(f"Frame accounting mismatch: {tracker.frames_count} versus {end - start}")
        tracker.export(video.stem)
        row = collector.row
        # TailTrail omits ROIs that were never visited. Explicit zeros keep
        # those recordings usable in the existing spreadsheet importer.
        for roi in rois.get():
            for column in (data_export.COLUMN_ROI_TIME, data_export.COLUMN_ROI_DISTANCE,
                           data_export.COLUMN_ROI_ENTRANCES):
                row.setdefault(f"{roi.name} {column}", 0)
        if job["images"]:
            track = tracker.draw_track_bgr(processor.masked_frame.copy())
            track = processor.draw_rois(cv2.cvtColor(track, cv2.COLOR_BGR2RGB))
            save_png(output / "tracking.png", cv2.cvtColor(track, cv2.COLOR_RGB2BGR))
            heatmap = processor.heatmap.create_heatmap_overlay()
            save_png(output / "heatmap.png", cv2.addWeighted(heatmap, 0.7, processor.masked_frame_black, 0.3, 0))
        result = dict(job_id=job["job_id"], video=str(video), row=row, width=width, height=height,
                      fps=fps, total_frames=total, start_frame=start, end_frame_exclusive=end,
                      processed_frames=tracker.frames_count, training_samples=training.reads,
                      elapsed_s=round(time.monotonic() - started, 3))
        # A completion marker is written only after every requested artifact.
        temporary = output / "result.json.tmp"
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(output / "result.json")
        print(f"Complete: {tracker.frames_count} frames; {row}", flush=True)
    finally:
        cap.release()


if __name__ == "__main__":
    run(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")))
