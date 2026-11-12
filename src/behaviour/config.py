"""Experiment configuration: arena geometry, analysis window and metric parameters.

One TOML file per arena / camera setup.  All tools are analysed with the same
file so that every metric means the same thing across tools.  See
``configs/example_open_field.toml`` for an annotated example.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path

import pandas as pd

from .zones import HYSTERESIS_MODES, CircleZone, PolygonZone, RectangleZone, Zone


@dataclass
class Session:
    """Per-video overrides (from the optional sessions CSV)."""

    start_s: float
    duration_s: float | None
    start_zone: str | None


@dataclass
class FstParams:
    """Forced swim test scoring (``[fst]``), see ``behaviour/fst.py``."""

    # Calibrated on Sturman et al. (2020) mouse videos (top view, 25 fps, 5 rater-labelled
    # videos; scripts/fst_calibration.py). The ratio depends on species, camera and frame
    # rate, so RECALIBRATE against manual scores on our rat setup before using it.
    motion_threshold: float = 0.40  # changed px / median animal area below which a frame is still
    smooth_s: float = 0.2           # centred rolling mean of that ratio before thresholding
    min_immobile_s: float = 1.0     # still frames count only in runs at least this long
    bin_s: float = 5.0              # time-sampling bins (Detke et al. 1995); 0 disables

    @classmethod
    def from_dict(cls, d: dict) -> "FstParams":
        names = {f.name for f in fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise ValueError(f"Unknown [fst] options {sorted(unknown)}; valid: {sorted(names)}")
        return cls(**{k: float(v) for k, v in d.items()})


@dataclass
class Config:
    px_per_m: float
    zones: list[Zone]
    hysteresis_mode: str = "outer"
    start_s: float = 0.0
    duration_s: float | None = 300.0
    start_zone: str | None = None
    max_speed_m_s: float = 2.0
    max_gap_s: float = 0.5
    smooth_s: float = 0.2
    immobile_speed_m_s: float = 0.02
    immobile_min_s: float = 1.0
    mask_margin_px: float = 20.0
    tracker: dict = field(default_factory=dict)
    fst: FstParams = field(default_factory=FstParams)
    sessions: dict[str, Session] = field(default_factory=dict)

    def zone_names(self) -> list[str]:
        return [z.name for z in self.zones]

    def session(self, video_id: str) -> Session:
        if video_id in self.sessions:
            return self.sessions[video_id]
        return Session(self.start_s, self.duration_s, self.start_zone)


def _zone_from_dict(d: dict, px_per_m: float, default_hysteresis_m: float) -> Zone:
    name = str(d["name"]).strip().lower()
    h_px = float(d.get("hysteresis_m", default_hysteresis_m)) * px_per_m
    shape = d.get("shape", "circle").lower()
    if shape == "circle":
        return CircleZone(name, h_px, cx=float(d["cx"]), cy=float(d["cy"]), r=float(d["r"]))
    if shape == "rectangle":
        return RectangleZone(name, h_px, x=float(d["x"]), y=float(d["y"]),
                             width=float(d["width"]), height=float(d["height"]))
    if shape == "polygon":
        return PolygonZone(name, h_px, points=d["points"])
    raise ValueError(f"Unknown zone shape {shape!r} for zone {name!r}")


def load_sessions(path: Path, defaults: Config) -> dict[str, Session]:
    """CSV with columns video_id[,start_s][,duration_s][,start_zone]."""
    df = pd.read_csv(path, dtype={"video_id": str})
    out = {}
    for row in df.to_dict("records"):
        def get(key, default):
            v = row.get(key)
            return default if v is None or pd.isna(v) or v == "" else v

        start_zone = get("start_zone", defaults.start_zone)
        out[str(row["video_id"])] = Session(
            start_s=float(get("start_s", defaults.start_s)),
            duration_s=None if get("duration_s", defaults.duration_s) is None
            else float(get("duration_s", defaults.duration_s)),
            start_zone=None if start_zone is None else str(start_zone).strip().lower(),
        )
    return out


def load_config(path: str | Path) -> Config:
    path = Path(path)
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    video = raw.get("video", {})
    window = raw.get("window", {})
    protocol = raw.get("protocol", {})
    cleaning = raw.get("cleaning", {})
    immobility = raw.get("immobility", {})

    px_per_m = float(video["px_per_m"])
    rules = raw.get("zone_rules", {})
    default_h = float(rules.get("hysteresis_m", 0.0))
    mode = rules.get("hysteresis_mode", "outer")
    if mode not in HYSTERESIS_MODES:
        raise ValueError(f"zone_rules.hysteresis_mode must be one of {list(HYSTERESIS_MODES)}")
    zones = [_zone_from_dict(z, px_per_m, default_h) for z in raw.get("zones", [])]
    if len({z.name for z in zones}) != len(zones):
        raise ValueError("Zone names must be unique (case-insensitive)")

    duration = window.get("duration_s", 300.0)
    start_zone = protocol.get("start_zone")
    cfg = Config(
        px_per_m=px_per_m,
        zones=zones,
        hysteresis_mode=mode,
        start_s=float(window.get("start_s", 0.0)),
        duration_s=None if duration in (None, 0) else float(duration),
        start_zone=None if not start_zone else str(start_zone).strip().lower(),
        max_speed_m_s=float(cleaning.get("max_speed_m_s", 2.0)),
        max_gap_s=float(cleaning.get("max_gap_s", 0.5)),
        smooth_s=float(cleaning.get("smooth_s", 0.2)),
        immobile_speed_m_s=float(immobility.get("speed_m_s", 0.02)),
        immobile_min_s=float(immobility.get("min_duration_s", 1.0)),
        mask_margin_px=float(video.get("mask_margin_px", 20.0)),
        tracker=dict(raw.get("tracker", {})),
        fst=FstParams.from_dict(raw.get("fst", {})),
    )
    if cfg.start_zone and cfg.start_zone not in cfg.zone_names():
        raise ValueError(f"protocol.start_zone {cfg.start_zone!r} is not one of the zones {cfg.zone_names()}")

    sessions_file = window.get("sessions_csv")
    if sessions_file:
        sessions_path = (path.parent / sessions_file).resolve()
        cfg.sessions = load_sessions(sessions_path, cfg)
    return cfg
