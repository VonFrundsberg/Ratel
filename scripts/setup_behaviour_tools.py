"""Install optional behaviour tools into isolated, project-local environments.

Run with the project Python after installing uv. Source videos are read-only.
No global Python, PATH, registry, commercial licence or account is changed.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = {
    "deeplabcut": dict(python="3.10", packages=["deeplabcut[gui]==3.0.2"]),
    "simba": dict(python="current", packages=["simba-uw-tf-dev==5.6.1", "setuptools<81"]),
    "labgym": dict(python="3.10", packages=["LabGym==3.1.2"]),
    "deepethogram": dict(python="3.10", packages=["deepethogram==0.5.0", "numpy<2",
                                                "torchmetrics<0.12", "setuptools<81"]),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tools", choices=sorted(TOOLS), nargs="+", default=list(TOOLS))
    ap.add_argument("--output", type=Path, default=ROOT / "output/open_field_benchmark")
    a = ap.parse_args()
    env = os.environ.copy()
    tmp = ROOT / "output/tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    env.update(TEMP=str(tmp), TMP=str(tmp), UV_CACHE_DIR=str(ROOT / "output/package_cache/uv"),
               UV_PYTHON_INSTALL_DIR=str(ROOT / "output/tool_python"), UV_LINK_MODE="copy")
    logs = a.output / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    uv = [sys.executable, "-m", "uv"]
    runtime = list((ROOT / "output/tool_python").glob("cpython-3.10.*-windows-x86_64-none/python.exe"))
    if any(TOOLS[t]["python"] == "3.10" for t in a.tools) and not runtime:
        result = subprocess.run([*uv, "python", "install", "3.10", "--no-bin"], env=env, cwd=ROOT)
        runtime = list((ROOT / "output/tool_python").glob("cpython-3.10.*-windows-x86_64-none/python.exe"))
        if not runtime:
            raise SystemExit(f"Python 3.10 download failed (exit {result.returncode})")
    statuses = []
    for name in a.tools:
        spec = TOOLS[name]
        folder = ROOT / "output/tool_envs" / name
        interpreter = sys.executable if spec["python"] == "current" else str(runtime[-1])
        python = folder / "Scripts/python.exe"
        with (logs / f"install_{name}.log").open("w", encoding="utf-8") as log:
            if not python.exists():
                subprocess.run([*uv, "venv", str(folder), "--python", interpreter],
                               env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            command = [*uv, "pip", "install", "--python", str(python)]
            if name == "deepethogram":
                command += ["--torch-backend", "cu124"]
            command += spec["packages"]
            result = subprocess.run(command, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        row = dict(tool=name, python=str(python), install_exit_code=result.returncode)
        if result.returncode == 0:
            try:
                checked = subprocess.run([str(python), str(ROOT / "scripts/tool_worker.py"), "probe",
                                          "--tool", name, "--output", str(a.output)], env=env, cwd=ROOT,
                                         timeout=120)
                row["probe_exit_code"] = checked.returncode
            except subprocess.TimeoutExpired:
                row.update(probe_exit_code=124, probe_error="Native import exceeded 120 seconds")
            with (logs / f"packages_{name}.txt").open("w", encoding="utf-8") as f:
                subprocess.run([*uv, "pip", "freeze", "--python", str(python)], env=env, cwd=ROOT, stdout=f)
        statuses.append(row)
        print(json.dumps(row), flush=True)
    (a.output / "tool_installation.json").write_text(json.dumps(statuses, indent=2), encoding="utf-8")
    if any(r["install_exit_code"] or r.get("probe_exit_code", 1) for r in statuses):
        raise SystemExit("Some optional tools failed; see individual install logs")


if __name__ == "__main__":
    main()
