"""Run the installed TailTrail engine on recordings without GUI interaction.

Example (run from the project directory):
  .venv/Scripts/python scripts/tailtrail_batch.py videos/zenodo_3608658 \
    --pattern "OFT_*.mp4" \
    --settings reports/zenodo_pilot/tailtrail/settings \
    --windows reports/zenodo_pilot/tailtrail/setup.csv -o output/tailtrail_batch

Re-run the same command to resume. See reports/tailtrail_batch.md for details.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import io
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path

from tailtrail_runtime import file_hash, folder_lock, prepare_runtime

ROOT = Path(__file__).resolve().parents[1]
WORKER = Path(__file__).with_name("tailtrail_worker.py")
CALIBRATOR = ROOT / "src/behaviour/tailtrail_calibration.py"
EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv", ".mpg", ".mpeg", ".m4v", ".wmv"}


class CalibrationReview(RuntimeError):
    pass


def write_atomic(path: Path, content: str):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict], columns: list[str]):
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    write_atomic(path, text.getvalue())


def recordings(paths: list[Path], recursive: bool, pattern: str) -> list[Path]:
    found = set()
    for path in paths:
        if path.is_dir():
            files = path.rglob(pattern) if recursive else path.glob(pattern)
            found.update(p.resolve() for p in files if p.is_file() and p.suffix.lower() in EXTENSIONS)
        elif path.is_file() and path.suffix.lower() in EXTENSIONS:
            found.add(path.resolve())
        else:
            raise ValueError(f"Not a recording or directory: {path}")
    videos = sorted(found, key=lambda p: str(p).casefold())
    names = [p.stem.casefold() for p in videos]
    if len(names) != len(set(names)):
        raise ValueError("Recording names must have unique stems; use separate output folders for duplicate names")
    if not videos:
        raise ValueError("No recordings found")
    return videos


def windows_csv(path: Path | None) -> dict:
    if path is None:
        return {}
    windows = {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not {"video", "start_s", "end_s"} <= set(reader.fieldnames or []):
            raise ValueError("Window CSV needs video,start_s,end_s columns (seconds; end is exclusive)")
        for row in reader:
            key = row["video"].strip().casefold()
            if not key or key in windows:
                raise ValueError(f"Empty or duplicate video in {path}: {key!r}")
            start, end = float(row["start_s"]), float(row["end_s"])
            if not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end:
                raise ValueError(f"Invalid analysis window for {key}")
            windows[key] = (start, end)
    return windows


def settings_for(video: Path, settings: Path) -> Path:
    if settings.is_file():
        return settings.resolve()
    for name in (video.stem, video.name, video.stem + ".pkl", video.stem + ".pickle"):
        candidate = settings / name
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"No saved settings for {video.name} in {settings}")


def landmark_files(path: Path | None) -> dict:
    if path is None:
        return {}
    files = {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        key_column = "video" if "video" in fields else "ID"
        if key_column not in fields or "dlc_file" not in fields:
            raise ValueError("Landmark map needs video,dlc_file or ID,dlc_file columns")
        for row in reader:
            key = Path(row[key_column].strip()).stem.casefold()
            file = Path(row["dlc_file"].strip())
            if not file.is_absolute():
                direct, pilot = path.parent / file, path.parent / "dlc" / file
                if direct.is_file() and pilot.is_file() and direct.resolve() != pilot.resolve():
                    raise ValueError(f"Ambiguous DLC file {file}; use an absolute path in {path}")
                file = direct if direct.is_file() else pilot
            if not key or key in files or not file.is_file():
                raise ValueError(f"Empty/duplicate video or missing DLC file in {path}: {row}")
            files[key] = file.resolve()
    return files


def plan(args, exe_hash: str) -> list[dict]:
    windows = windows_csv(args.windows)
    landmarks = landmark_files(args.landmarks_map)
    jobs = []
    worker_hash = file_hash(WORKER)
    for video in recordings(args.videos, args.recursive, args.pattern):
        settings = settings_for(video, args.settings)
        start, end = args.start_s, None if args.duration_s is None else args.start_s + args.duration_s
        if args.windows:
            key = video.name.casefold()
            if key not in windows:
                raise ValueError(f"Window CSV has no row for {video.name}; include its extension")
            start, end = windows[key]
        stat = video.stat()
        identity = dict(video=str(video), video_size=stat.st_size, video_mtime_ns=stat.st_mtime_ns,
                        settings_sha256=file_hash(settings), exe_sha256=exe_hash,
                        worker_sha256=worker_hash, start_s=start, end_s=end,
                        threads=args.threads, images=not args.no_images)
        if args.auto_calibrate:
            landmark = landmarks.get(video.stem.casefold())
            if args.landmarks_map and landmark is None:
                raise ValueError(f"Landmark map has no row for {video.name}")
            identity["calibration"] = dict(
                arena_shape=args.arena_shape, arena_diameter_m=args.arena_diameter_m,
                arena_side_m=args.arena_side_m, center_scale=args.center_scale,
                periphery_scale=args.periphery_scale, hysteresis_m=args.hysteresis_m,
                mask_margin_px=args.mask_margin_px, code_sha256=file_hash(CALIBRATOR),
                landmarks_file=str(landmark) if landmark else None,
                landmarks_sha256=file_hash(landmark) if landmark else None)
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        label = re.sub(r"[^\w.-]+", "_", video.stem)[:64]
        job_id = f"{label}-{digest[:16]}"
        jobs.append(dict(identity, job_id=job_id, settings=str(settings),
                         settings_snapshot=str(args.output / "results" / job_id / "settings.pickle"),
                         result_dir=str(args.output / "results" / job_id)))
    return jobs


def calibrated(job: dict) -> dict | None:
    folder = Path(job["result_dir"])
    try:
        report = json.loads((folder / "calibration.json").read_text(encoding="utf-8"))
        if (report["job_id"] == job["job_id"] and report["accepted"]
                and report["settings_sha256"] == file_hash(Path(job["settings_snapshot"]))
                and (folder / "calibration_preview.png").is_file()):
            return report
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def completed(job: dict) -> dict | None:
    if job.get("calibration") and calibrated(job) is None:
        return None
    path = Path(job["result_dir"]) / "result.json"
    if path.exists():
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
            if result["job_id"] == job["job_id"] and result["processed_frames"] > 0 and result["row"]:
                if not job["images"] or all((path.parent / name).is_file() for name in
                                           ("roi_preview.png", "tracking.png", "heatmap.png")):
                    return result
        except (ValueError, KeyError, TypeError):
            pass
    return None


def export_summary(output: Path, results: list[dict]):
    from openpyxl import Workbook

    columns = list(dict.fromkeys(key for result in results for key in result["row"]))
    if not columns:
        columns = ["#", "total time (s)", "total distance (m)", "losses"]
    rows = [result["row"] for result in results]
    write_csv(output / "data.csv", rows, columns)
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Sheet1"
    sheet.append(columns)
    for row in rows:
        sheet.append([row.get(key) for key in columns])
    temporary = output / "data.tmp.xlsx"
    workbook.save(temporary)
    temporary.replace(output / "data.xlsx")


def stop_worker(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def execute(args, jobs: list[dict], python: Path | None) -> int:
    output = args.output
    (output / "logs").mkdir(exist_ok=True)
    (output / "jobs").mkdir(exist_ok=True)
    status = []
    results = {job["job_id"]: result for job in jobs
               if not args.calibrate_only and not args.force and (result := completed(job)) is not None}

    def update_summary():
        if args.calibrate_only:
            return False
        # Include every cached success for this selection, even if a recording
        # earlier in the queue has not finished yet. Empty batches clear stale
        # summaries from a previous selection in the same output directory.
        selected = [results[job["job_id"]] for job in jobs if job["job_id"] in results]
        try:
            export_summary(output, selected)
            return False
        except OSError as exc:
            print(f"  Summary export failed: {exc}. Individual results are saved; close Excel and re-run.", flush=True)
            return True

    columns = ["video", "settings", "state", "elapsed_s", "error", "result_dir", "log",
               "calibration_source", "calibration"]
    failures, summary_error = 0, update_summary()
    for index, job in enumerate(jobs, 1):
        video = Path(job["video"])
        log_path = output / "logs" / (job["job_id"] + ".log")
        row = dict(video=str(video), settings=job["settings"], state="running", elapsed_s=0,
                   error="", result_dir=job["result_dir"], log=str(log_path),
                   calibration_source=("dlc" if job["calibration"]["landmarks_file"] else "image") if job.get("calibration") else "",
                   calibration=str(Path(job["result_dir"]) / "calibration.json") if job.get("calibration") else "")
        status.append(row)
        result = completed(job)
        if args.calibrate_only and calibrated(job) is not None and not args.force:
            result = None
            row["state"] = "calibrated"
            print(f"[{index}/{len(jobs)}] Already calibrated: {video.name}", flush=True)
        elif result and not args.force:
            row.update(state="skipped", elapsed_s=result["elapsed_s"])
            print(f"[{index}/{len(jobs)}] Already complete: {video.name}", flush=True)
        else:
            result = None
            print(f"[{index}/{len(jobs)}] Running: {video.name}", flush=True)
            write_csv(output / "batch_status.csv", status, columns)
            started, process = time.monotonic(), None
            try:
                # Inputs are checked again so changing a recording/settings while
                # a long batch is in progress cannot invalidate resume identities.
                stat = video.stat()
                if (stat.st_size, stat.st_mtime_ns) != (job["video_size"], job["video_mtime_ns"]):
                    raise RuntimeError("Recording changed after the batch was planned; run the command again")
                settings_bytes = Path(job["settings"]).read_bytes()
                if hashlib.sha256(settings_bytes).hexdigest() != job["settings_sha256"]:
                    raise RuntimeError("Settings changed after the batch was planned; run the command again")
                folder = Path(job["result_dir"])
                folder.mkdir(parents=True, exist_ok=True)
                (folder / "result.json").unlink(missing_ok=True)
                job_path = output / "jobs" / (job["job_id"] + ".json")
                write_atomic(job_path, json.dumps(job, indent=2))
                if job.get("calibration"):
                    import cv2
                    from behaviour.tailtrail_calibration import calibrate

                    cv2.setNumThreads(job["threads"])
                    params = job["calibration"]
                    landmark = params["landmarks_file"]
                    if landmark and file_hash(Path(landmark)) != params["landmarks_sha256"]:
                        raise RuntimeError("DLC landmarks changed after planning; run the command again")
                    template = folder / "source_settings.pickle"
                    template.write_bytes(settings_bytes)
                    report = None if args.force else calibrated(job)
                    if report is None:
                        report = calibrate(video, template, folder, params, job["job_id"])
                    if not report["accepted"]:
                        raise CalibrationReview(report["reason"])
                    geometry = report["geometry"]
                    print(f"  Calibrated ({report['source']}): {geometry['scale_px_per_m']:g} px/m; mask radius {geometry['mask_radius']}", flush=True)
                else:
                    Path(job["settings_snapshot"]).write_bytes(settings_bytes)
                if args.calibrate_only:
                    row.update(state="calibrated", elapsed_s=round(time.monotonic() - started, 2))
                else:
                    with log_path.open("w", encoding="utf-8") as log:
                        process = subprocess.Popen([str(python), "-X", "utf8", "-u", str(WORKER), str(job_path)],
                                                   cwd=folder, stdout=log, stderr=subprocess.STDOUT,
                                                   creationflags=subprocess.CREATE_NO_WINDOW)
                        timeout = None if args.timeout_min == 0 else args.timeout_min * 60
                        try:
                            code = process.wait(timeout=timeout)
                        except subprocess.TimeoutExpired:
                            stop_worker(process)
                            raise RuntimeError(f"Exceeded {args.timeout_min:g} minute timeout") from None
                    if code:
                        raise RuntimeError(f"TailTrail worker exited with code {code}; see {log_path}")
                    result = completed(job)
                    if result is None:
                        raise RuntimeError("Worker did not produce a complete result")
                    row.update(state="done", elapsed_s=round(time.monotonic() - started, 2))
                    print(f"  Saved {result['processed_frames']} frames in {row['elapsed_s']}s", flush=True)
            except KeyboardInterrupt:
                if process is not None:
                    stop_worker(process)
                row.update(state="interrupted", error="Stopped by user")
                write_csv(output / "batch_status.csv", status, columns)
                print("Batch stopped. Re-run the same command to resume.", flush=True)
                return 130
            except Exception as exc:
                if process is not None:
                    stop_worker(process)
                row.update(state="needs_review" if isinstance(exc, CalibrationReview) else "failed",
                           error=str(exc), elapsed_s=round(time.monotonic() - started, 2))
                failures += 1
                print(f"  {row['state'].upper()}: {exc}", flush=True)
        if result is not None:
            results[job["job_id"]] = result
        write_csv(output / "batch_status.csv", status, columns)
        if row["state"] == "done" or summary_error:
            summary_error = update_summary()
    if args.calibrate_only:
        ready = sum(row["state"] == "calibrated" for row in status)
        print(f"Finished calibration: {ready} ready, {failures} failed or need review. Results: {output}", flush=True)
    else:
        print(f"Finished: {len(results)} complete, {failures} failed or need review. Results: {output}", flush=True)
    return 1 if failures or summary_error else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", type=Path, nargs="*", help="recordings or folders")
    ap.add_argument("--settings", type=Path, help="one saved TailTrail file, or a folder with one file per video stem")
    ap.add_argument("--windows", type=Path, help="CSV with video,start_s,end_s; requires a row for every recording")
    ap.add_argument("--auto-calibrate", action="store_true", help="fit each square arena and generate its scale, ROIs and mask")
    ap.add_argument("--arena-shape", choices=("square", "circle"), default="square")
    ap.add_argument("--arena-diameter-m", type=float, help="known floor diameter for --arena-shape circle")
    ap.add_argument("--arena-side-m", type=float, help="known square-arena side length in metres; required for calibration")
    ap.add_argument("--landmarks-map", type=Path, help="optional video/ID,dlc_file CSV with tl,tr,br,bl landmarks; otherwise fit images")
    ap.add_argument("--calibrate-only", action="store_true", help="generate settings and previews without tracking; implies --auto-calibrate")
    ap.add_argument("--center-scale", type=float, default=.5, help="centre side / arena side (default: 0.5)")
    ap.add_argument("--periphery-scale", type=float, default=1.1, help="outer ROI side / arena side (default: 1.1)")
    ap.add_argument("--hysteresis-m", type=float, default=.01, help="calibrated ROI hysteresis in metres (default: 0.01)")
    ap.add_argument("--mask-margin-px", type=float, default=10, help="circle margin beyond calibrated outer ROI (default: 10)")
    ap.add_argument("--start-s", type=float, default=0.0)
    ap.add_argument("--duration-s", type=float, help="omit to process through the end of each recording")
    ap.add_argument("--exe", type=Path, default=ROOT / "tail-trail/tail-trail.exe")
    ap.add_argument("--runtime-cache", type=Path, default=ROOT / "output/tailtrail_runtime")
    ap.add_argument("-o", "--output", type=Path, default=ROOT / "output/tailtrail_batch")
    ap.add_argument("--recursive", action="store_true", help="include subfolders")
    ap.add_argument("--pattern", default="*", help="filename glob for folders, e.g. OFT_*.mp4 (default: all videos)")
    ap.add_argument("--threads", type=int, default=2, help="OpenCV threads per recording (default: 2)")
    ap.add_argument("--timeout-min", type=float, default=120, help="per recording; 0 disables (default: 120)")
    ap.add_argument("--no-images", action="store_true", help="skip ROI preview, track and heatmap PNGs")
    ap.add_argument("--force", action="store_true", help="replace results for jobs already completed with these inputs")
    ap.add_argument("--allow-sleep", action="store_true", help="do not keep Windows awake during the batch")
    ap.add_argument("--dry-run", action="store_true", help="show planned recordings, settings and windows without running")
    ap.add_argument("--prepare-only", action="store_true", help="prepare the local runtime, without processing recordings")
    args = ap.parse_args()
    if args.calibrate_only:
        args.auto_calibrate = True
    if sys.platform != "win32":
        ap.error("This runner uses the Windows TailTrail executable")
    if args.threads < 1 or not math.isfinite(args.timeout_min) or args.timeout_min < 0:
        ap.error("--threads must be positive and --timeout-min must be finite and nonnegative")
    if not math.isfinite(args.start_s) or args.start_s < 0:
        ap.error("--start-s must be finite and nonnegative")
    if args.duration_s is not None and (not math.isfinite(args.duration_s) or args.duration_s <= 0):
        ap.error("--duration-s must be finite and positive")
    if args.windows and (args.start_s != 0 or args.duration_s is not None):
        ap.error("Use either --windows or --start-s/--duration-s")
    if args.prepare_only and args.calibrate_only:
        ap.error("Use either --prepare-only or --calibrate-only")
    if args.auto_calibrate:
        size = args.arena_diameter_m if args.arena_shape == "circle" else args.arena_side_m
        if size is None or not math.isfinite(size) or size <= 0:
            ap.error("Calibration requires --arena-diameter-m for circles or --arena-side-m for squares")
        if args.arena_shape == "circle" and (args.landmarks_map or args.arena_side_m is not None):
            ap.error("Circle calibration uses image boundaries and --arena-diameter-m, without square landmarks/side")
        if args.arena_shape == "square" and args.arena_diameter_m is not None:
            ap.error("--arena-diameter-m requires --arena-shape circle")
        values = (args.center_scale, args.periphery_scale, args.hysteresis_m, args.mask_margin_px)
        if (not all(math.isfinite(v) for v in values) or not 0 < args.center_scale < 1 <= args.periphery_scale
                or args.hysteresis_m < 0 or args.mask_margin_px < 0):
            ap.error("Calibration requires 0 < center-scale < 1 <= periphery-scale and nonnegative hysteresis/mask margin")
    elif args.landmarks_map or args.arena_side_m is not None or args.arena_diameter_m is not None or args.arena_shape != "square":
        ap.error("--landmarks-map and --arena-side-m require --auto-calibrate or --calibrate-only")
    if not args.prepare_only and (not args.videos or args.settings is None):
        ap.error("Provide recording paths and --settings, or use --prepare-only")
    args.exe, args.output, args.runtime_cache = args.exe.resolve(), args.output.resolve(), args.runtime_cache.resolve()
    digest = file_hash(args.exe)
    jobs = [] if args.prepare_only else plan(args, digest)
    if args.dry_run:
        for job in jobs:
            print(f"{job['video']}\n  settings: {job['settings']}\n  seconds: {job['start_s']} -> {job['end_s'] or 'end'}")
            if job.get("calibration"):
                params = job["calibration"]
                print(f"  calibration: {params['landmarks_file'] or 'image floor detection'}; arena {params['arena_shape']}, size {params['arena_diameter_m'] or params['arena_side_m']} m")
        print(f"{len(jobs)} recording(s); output: {args.output}")
        return 0
    if not args.prepare_only:
        # Check the summary writer before starting hours of processing.
        import openpyxl  # noqa: F401
    python = None if args.calibrate_only else prepare_runtime(args.exe, args.runtime_cache, digest)
    if args.prepare_only:
        print(f"Ready: {python}")
        return 0
    with folder_lock(args.output):
        awake = False
        try:
            if not args.allow_sleep:
                awake = bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
                if not awake:
                    print("Could not inhibit sleep; disable automatic sleep in Windows before leaving.", flush=True)
            return execute(args, jobs, python)
        finally:
            if awake:
                ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
