"""Scoring application discovery and a manifest-based shared comparison."""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

from .config import load_config
from .compare import compare
from .ethogram import events_to_metrics, normalize_events
from .adapters.behaviours import read_frames, read_intervals, read_deepethogram
from .adapters.boris import read_events

ROOT = Path(__file__).absolute().parents[2]
TOOLS = {
    "boris": ("manual annotations", None),
    "deeplabcut": ("pose estimation", "deeplabcut"),
    "simba": ("pose-based classification", "simba"),
    "labgym": ("video-based classification", "LabGym"),
    "deepethogram": ("video-based classification", "deepethogram"),
    "behaviordepot": ("MATLAB pose-based scoring", None),
    "dbscorer": ("forced-swim mobility", None),
}


def executable(tool):
    if tool in ("boris", "dbscorer"):
        pattern = "python.exe" if tool == "boris" else "DBscorerV2.exe"
        return next((ROOT / "output/scoring_tools" / tool).rglob(pattern), None)
    if tool == "behaviordepot":
        path = shutil.which("matlab")
        return Path(path) if path else None
    path = ROOT / "output/tool_envs" / tool / "Scripts/python.exe"
    return path if path.exists() else None


def status():
    rows = []
    for tool, (role, module) in TOOLS.items():
        path = executable(tool)
        note = ""
        state = "missing"
        if path and module:
            probe = subprocess.run([str(path), "-c",
                f"import importlib.util; print(bool(importlib.util.find_spec({module!r})))"],
                capture_output=True, text=True, timeout=30)
            state = "module located" if probe.returncode == 0 and probe.stdout.strip() == "True" else "environment needs repair"
        elif path:
            state = "executable located"
        if tool == "dbscorer":
            runtime = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "MATLAB/MATLAB Runtime/v99"
            state = "runtime present" if path and runtime.exists() else "needs MATLAB Runtime 9.9"
            note = "https://github.com/swanandlab/DBscorer#installation"
        if tool == "behaviordepot":
            state = "MATLAB located; check toolboxes" if path else "needs licensed MATLAB and toolboxes"
            note = "downloaded .mltbx requires installation in MATLAB"
        rows.append(dict(tool=tool, role=role, status=state, executable=str(path or ""), note=note))
    return pd.DataFrame(rows)


def launch(tool, args):
    path = executable(tool)
    if not path:
        raise ValueError(f"{tool}: install the application first; see ratel scorers list")
    if tool == "behaviordepot":
        raise ValueError("Install output/scoring_tools/behaviordepot/*.mltbx in MATLAB and launch BehaviorDEPOT from its Apps tab")
    if tool == "boris":
        command = [str(path), "-m", "boris", *args]
    elif tool == "dbscorer":
        command = [str(path), *args]
    else:
        command = [str(path), "-m", TOOLS[tool][1], *args]
        if tool == "simba":
            command = [str(path), "-c", "from simba.SimBA import main; main()", *args]
    # An explicit launch command requests an interactive application.
    subprocess.Popen(command, cwd=path.parent if tool == "boris" else ROOT)


def import_row(row, base):
    source = (base / row["export"]).absolute()
    fmt = row["format"]
    vid = row["video_id"]
    names = row.get("behaviours", "").split("|") if row.get("behaviours") else []
    points = row.get("points", "").split("|") if row.get("points") else []
    mapping = json.loads(row.get("behaviour_map") or "{}")
    columns = json.loads(row.get("columns") or "{}")
    fps = float(row["fps"]) if row.get("fps") else None
    threshold = float(row.get("threshold") or 0.5)
    if fmt == "intervals":
        events = read_intervals(source, vid, columns, mapping)
    elif fmt == "boris":
        events = normalize_events(read_events(source, include_points=True), mapping)
        if events.video_id.nunique() > 1:
            raise ValueError("Export each observation separately")
        events["video_id"] = vid
    elif fmt == "deepethogram":
        if not fps or not names:
            raise ValueError("DeepEthogram needs fps and behaviours in stored order")
        events = normalize_events(read_deepethogram(source, fps, names, row.get("dataset") or None, threshold, vid), mapping)
    elif fmt == "behaviordepot":
        from scipy.io import loadmat
        if not fps or not names:
            raise ValueError("BehaviorDEPOT needs fps and exact Behavior field names")
        behavior = loadmat(source, simplify_cells=True)["Behavior"]
        vectors = {name: np.atleast_1d(behavior[name]["Vector"]).reshape(-1) for name in names}
        from .adapters.behaviours import frame_events
        events = frame_events(pd.DataFrame(vectors), vid, fps, names, threshold, mapping=mapping)
    elif fmt in ("frames", "labels"):
        events = read_frames(source, fps, names, threshold, vid, columns,
                             "behaviour" if fmt == "labels" else None, mapping)
    else:
        raise ValueError(f"Unsupported format {fmt!r}")
    declared = {k: "state" for k in names} if names else {}
    declared.update({k: "point" for k in points})
    declared = {mapping.get(k, k): kind for k, kind in declared.items()}
    cfg = load_config(base / row["config"])
    result = events_to_metrics(events, cfg, row["run_id"], declared, vid)
    return events, result, cfg.session(vid), source


def batch(manifest, output, reference):
    manifest = Path(manifest).absolute()
    output = Path(output).absolute()
    if output.exists():
        raise ValueError("Choose a new output directory for a comparison")
    table = pd.read_csv(manifest, dtype=str, keep_default_na=False)
    required = {"run_id", "tool", "video_id", "config", "export", "format"}
    if required - set(table):
        raise ValueError(f"Missing manifest columns: {sorted(required-set(table))}")
    if table[list(required)].eq("").any().any():
        raise ValueError("Manifest identifiers, paths and formats cannot be empty")
    if table.duplicated(["run_id", "video_id"]).any():
        raise ValueError("Each run/video pair must appear once")
    metrics, records, windows, provenance = [], [], {}, []
    for row in table.to_dict("records"):
        if row["tool"] not in TOOLS:
            raise ValueError(f"Unknown scoring tool {row['tool']}")
        events, result, session, source = import_row(row, manifest.parent)
        duration = result[result.metric == "duration_s"].value.unique()
        if len(duration) != 1:
            raise ValueError("Each export needs one observed duration")
        window = (session.start_s, float(duration[0]))
        if row["video_id"] in windows and windows[row["video_id"]] != window:
            raise ValueError(f"Mismatched observation windows: {row['video_id']}")
        windows[row["video_id"]] = window
        records.append(events.assign(run_id=row["run_id"], source_tool=row["tool"]))
        metrics.append(result)
        from hashlib import sha256
        config = manifest.parent / row["config"]
        provenance.append(dict(**row, source_sha256=sha256(source.read_bytes()).hexdigest(),
                               config_sha256=sha256(config.read_bytes()).hexdigest(),
                               effective_start_s=window[0], effective_duration_s=window[1]))
    combined = pd.concat(metrics, ignore_index=True)
    stats = compare(combined, reference)
    # Parse and validate the entire manifest before publishing derived files.
    import tempfile
    from hashlib import sha256
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as folder:
        staged = Path(folder) / "result"
        staged.mkdir()
        combined.to_csv(staged / "metrics.csv", index=False)
        pd.concat(records, ignore_index=True).to_csv(staged / "events.csv", index=False)
        stats.to_csv(staged / "agreement.csv", index=False)
        (staged / "sources.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        (staged / "run.json").write_text(json.dumps(dict(
            manifest_sha256=sha256(manifest.read_bytes()).hexdigest(),
            importer_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
            python=sys.version, reference=reference), indent=2), encoding="utf-8")
        staged.rename(output)
    print(f"{len(table)} exports -> {output}")
