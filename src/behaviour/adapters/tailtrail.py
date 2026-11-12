"""TailTrail (lab tracking program, closed-source exe) adapter.

TailTrail exports only one summary row per video to ``data.xlsx``::

    #, total time (s), total distance (m),
    <ROI> time (s), <ROI> distance (m), <ROI> entrances, ..., losses

Differences from our metric definitions, handled here:

* ``entrances`` include the ROI the animal starts in.  We subtract 1 from the
  start zone (``protocol.start_zone`` / sessions CSV).  If the start zone is
  unknown the raw counts are kept under ``entries_incl_start`` (quality), so
  they are never silently compared with corrected counts.
* Re-exporting the same video appends another row; the last row wins.
* The ``#`` cell is the video file name prefixed with "Object".

Its saved settings file is a Python pickle.  ``read_settings`` loads it with a
restricted unpickler that only allows numpy arrays and a stand-in ROI class, so
nothing in the file can execute code.
"""

from __future__ import annotations

import io
import pickle
import re
import sys
import types
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
from ..metrics import LONG_COLUMNS, METRICS

TOOL = "tailtrail"
_COL = re.compile(r"^(?P<roi>.+?)\s+(?P<what>time \(s\)|distance \(m\)|entrances)$", re.IGNORECASE)


# --------------------------------------------------------------------------- settings


class _ROI:
    """Stand-in for TailTrail's src.model.region_of_interest.RegionOfInterest."""


class _SafeUnpickler(pickle.Unpickler):
    _ALLOWED = {
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
    }

    def find_class(self, module, name):
        if module.endswith("region_of_interest") and name == "RegionOfInterest":
            return _ROI
        if (module, name) in self._ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"TailTrail settings: refusing to load {module}.{name}")


def load_raw_settings(path: str | Path) -> dict:
    """The settings pickle as stored (ROIs as stand-in objects), for editing with ``dump_settings``."""
    return _SafeUnpickler(io.BytesIO(Path(path).read_bytes())).load()


def dump_settings(raw: dict, path: str | Path) -> None:
    """Write settings from ``load_raw_settings`` back in TailTrail's own format (File > Load Settings).

    Pickles the stand-in ROIs under TailTrail's class name, so TailTrail rebuilds its own
    objects from them; a file that was loaded and dumped unchanged is byte-identical.
    """
    module, name = "src.model.region_of_interest", "RegionOfInterest"
    fake = types.ModuleType(module)
    fake.RegionOfInterest = _ROI
    names = ("src", "src.model", module)  # pickle imports the parent packages too
    saved = {k: sys.modules.get(k) for k in names}
    _ROI.__module__, _ROI.__qualname__ = module, name
    for k in names[:-1]:
        sys.modules[k] = types.ModuleType(k)
    sys.modules[module] = fake
    try:
        Path(path).write_bytes(pickle.dumps(raw, protocol=4))
    finally:
        _ROI.__module__, _ROI.__qualname__ = __name__, "_ROI"
        for k, m in saved.items():
            if m is None:
                del sys.modules[k]
            else:
                sys.modules[k] = m


def rectangle_contours(x: int, y: int, width: int, height: int, hysteresis_px: int) -> tuple[tuple, tuple]:
    """``contour`` / ``hysteresis_contour`` of a Rectangle ROI as TailTrail stores them:
    a 1-tuple of an int32 (4, 1, 2) array, corners counter-clockwise from the top-left."""
    def rect(x0, y0, x1, y1):
        return (np.array([[[x0, y0]], [[x0, y1]], [[x1, y1]], [[x1, y0]]], dtype=np.int32),)
    h = hysteresis_px
    return rect(x, y, x + width, y + height), rect(x - h, y - h, x + width + h, y + height + h)


def read_settings(path: str | Path) -> dict:
    """Load a TailTrail settings file (File > Save Settings, or ``default_settings``)."""
    data = load_raw_settings(path)
    rois = []
    for r in data.get("rois", []):
        d = dict(vars(r))
        for k in ("contour", "hysteresis_contour"):
            c = d.get(k)
            if isinstance(c, (tuple, list)) and c:  # cv2.findContours output: keep the largest
                c = max(c, key=len)
            d[k] = c.reshape(-1, 2) if isinstance(c, np.ndarray) else None
        rois.append(d)
    data["rois"] = rois
    return data


def settings_to_toml(settings: dict, source: str = "") -> str:
    """Write an experiment config whose zones are exactly TailTrail's ROIs."""
    lines = [
        f"# Generated from TailTrail settings {source}".rstrip(),
        "# Zone order = TailTrail ROI order = priority (first match wins).",
        "",
    ]
    tx, ty, rot = (settings.get(k, 0) for k in ("translate_frame_x", "translate_frame_y", "rotate_frame"))
    if tx or ty or rot:
        lines += [
            f"# WARNING: TailTrail transforms the frame (translate {tx},{ty}, rotate {rot} deg) before",
            "# applying ROIs. These coordinates are in the transformed frame and will NOT line up",
            "# with other tools until that transform is reproduced. Check with `behav preview`.",
            "",
        ]
    lines += [
        "[video]",
        f"px_per_m = {float(settings.get('scale', 0.0))}",
        "mask_margin_px = 20.0",
        "",
        "[window]",
        "start_s = 0.0",
        "duration_s = 300.0",
        '# sessions_csv = "sessions.csv"   # per-video start_s / duration_s / start_zone',
        "",
        "[protocol]",
        '# Zone the rat is placed in; needed to correct TailTrail entry counts.',
        '# start_zone = "center"',
        "",
        "[zone_rules]",
        '# Where the hysteresis band sits: "outer" (enter at border, leave h outside),',
        '# "inner" or "symmetric". "outer" is our reading of the TailTrail manual - verify.',
        'hysteresis_mode = "outer"',
        "",
        "[cleaning]",
        "max_speed_m_s = 2.0",
        "max_gap_s = 0.5",
        "smooth_s = 0.2",
        "",
        "[immobility]",
        "speed_m_s = 0.02",
        "min_duration_s = 1.0",
        "",
        "[tracker]",
        f"min_area_px = {int(settings.get('min_area', 50))}",
        "",
    ]
    for r in settings["rois"]:
        name = str(r["name"]).strip().lower()
        kind = str(r.get("type", "")).lower()
        lines += ["[[zones]]", f'name = "{name}"', f"hysteresis_m = {float(r.get('hysteresis', 0.0))}"]
        if kind == "circle":
            lines += ['shape = "circle"', f"cx = {r['x']}", f"cy = {r['y']}", f"r = {r['radius']}"]
        else:
            # Rectangle / Quadrant: use TailTrail's own contour so no assumptions about
            # its x/y/angle conventions are needed.
            contour = r.get("contour")
            if contour is None or len(contour) < 3:
                raise ValueError(f"TailTrail ROI {name!r} ({kind}) has no contour to import")
            pts = ", ".join(f"[{int(x)}, {int(y)}]" for x, y in contour)
            lines += [f'# TailTrail type: {r.get("type")}', 'shape = "polygon"', f"points = [{pts}]"]
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- data.xlsx


def video_id_from_label(label: str) -> str:
    s = re.sub(r"^\s*object\s*", "", str(label), flags=re.IGNORECASE).strip()
    return Path(s).stem if Path(s).suffix else s


def read_data_xlsx(path: str | Path, cfg: Config) -> pd.DataFrame:
    """Convert TailTrail's data.xlsx into the long metric table."""
    raw = pd.read_excel(path, sheet_name=0)
    raw.columns = [str(c).strip() for c in raw.columns]
    label_col = raw.columns[0]
    raw = raw.dropna(subset=[label_col])
    raw["video_id"] = raw[label_col].map(video_id_from_label)

    dup = raw["video_id"].duplicated(keep="last")
    if dup.any():
        warnings.warn(f"{path}: {sorted(set(raw.loc[dup, 'video_id']))} exported more than once; "
                      "using the last row for each", stacklevel=2)
        raw = raw.loc[~dup]

    zone_names = set(cfg.zone_names())
    roi_cols: dict[str, dict[str, str]] = {}
    for c in raw.columns:
        m = _COL.match(c)
        if m and m["roi"].lower() not in ("total",):
            roi_cols.setdefault(m["roi"].strip().lower(), {})[m["what"].lower()] = c
    unknown = set(roi_cols) - zone_names
    if unknown:
        warnings.warn(f"{path}: TailTrail ROIs {sorted(unknown)} are not zones in the config "
                      f"{sorted(zone_names)}; they will not match other tools", stacklevel=2)

    total_t = _find(raw, "total time (s)")
    total_d = _find(raw, "total distance (m)")
    losses = _find(raw, "losses", required=False)

    rows = []
    for rec in raw.to_dict("records"):
        vid = rec["video_id"]
        start_zone = cfg.session(vid).start_zone
        T = float(rec[total_t])
        D = float(rec[total_d])
        out = [
            ("duration_s", "", T),
            ("total_distance_m", "", D),
            ("mean_speed_m_s", "", D / T if T else np.nan),
        ]
        if losses:
            out.append(("losses", "", float(rec[losses])))
        zone_time_sum = 0.0
        for roi, cols in roi_cols.items():
            if "time (s)" in cols:
                t = float(rec[cols["time (s)"]])
                zone_time_sum += t
                out += [("time_s", roi, t), ("time_pct", roi, 100.0 * t / T if T else np.nan)]
            if "distance (m)" in cols:
                out.append(("distance_m", roi, float(rec[cols["distance (m)"]])))
            if "entrances" in cols:
                n = float(rec[cols["entrances"]])
                if start_zone is None:
                    out.append(("entries_incl_start", roi, n))
                else:
                    out.append(("entries", roi, max(n - (roi == start_zone), 0.0)))
        out.append(("unassigned_time_s", "", max(T - zone_time_sum, 0.0)))
        for metric, zone, value in out:
            kind = METRICS[metric][1] if metric in METRICS else "quality"
            rows.append(dict(video_id=vid, tool=TOOL, metric=metric, zone=zone, value=value, kind=kind))

    if any(r["metric"] == "entries_incl_start" for r in rows):
        warnings.warn("protocol.start_zone is not set: TailTrail entrance counts kept raw as "
                      "'entries_incl_start' and excluded from comparison", stacklevel=2)
    return pd.DataFrame(rows, columns=LONG_COLUMNS)


def _find(df: pd.DataFrame, name: str, required: bool = True) -> str | None:
    for c in df.columns:
        if c.lower() == name:
            return c
    if required:
        raise ValueError(f"TailTrail data.xlsx has no '{name}' column; columns are {list(df.columns)}")
    return None
