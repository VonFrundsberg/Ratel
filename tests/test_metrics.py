import numpy as np
import pandas as pd
import pytest

from behaviour.metrics import metrics_for_trajectory
from behaviour.trajectory import clean

FPS = 10.0


def traj(x, y, fps=FPS):
    n = len(x)
    return pd.DataFrame({"frame": np.arange(n), "time_s": np.arange(n) / fps, "x_px": x, "y_px": y})


def get(long, metric, zone=""):
    v = long[(long.metric == metric) & (long.zone == zone)]["value"]
    assert len(v) == 1, (metric, zone)
    return float(v.iloc[0])


def test_window_and_straight_line(cfg):
    cfg.smooth_s = 0
    cfg.duration_s = 10.0
    # 20 s of data; analysed window is the first 10 s, moving 32 px/s = 0.1 m/s along x
    n = 200
    x = 320 + 32 * np.arange(n) / FPS
    long = metrics_for_trajectory(traj(x, np.full(n, 240.0)), cfg, "v", "test")
    assert get(long, "duration_s") == pytest.approx(10.0)
    assert get(long, "total_distance_m") == pytest.approx(0.99, abs=1e-6)  # 99 steps of 0.01 m
    assert get(long, "mean_speed_m_s") == pytest.approx(0.099, abs=1e-6)


def test_zone_time_entries_latency(cfg):
    cfg.smooth_s = 0
    cfg.duration_s = None
    cfg.max_speed_m_s = 100  # the test animal teleports between zones
    for z in cfg.zones:
        z.hysteresis_px = 0
    # 3 s center, 2 s inner, 2 s outer, 3 s center (10 frames/s)
    x = np.r_[np.full(30, 320.0), np.full(20, 400.0), np.full(20, 480.0), np.full(30, 320.0)]
    long = metrics_for_trajectory(traj(x, np.full(len(x), 240.0)), cfg, "v", "test")
    assert get(long, "time_s", "center") == pytest.approx(6.0)
    assert get(long, "time_s", "inner") == pytest.approx(2.0)
    assert get(long, "time_s", "outer") == pytest.approx(2.0)
    assert get(long, "time_pct", "center") == pytest.approx(60.0)
    # started in center -> that start does not count as an entry
    assert get(long, "entries", "center") == 1
    assert get(long, "entries", "inner") == 1
    assert get(long, "entries", "outer") == 1
    assert get(long, "latency_s", "center") == 0
    assert get(long, "latency_s", "outer") == pytest.approx(5.0)
    assert get(long, "unassigned_time_s") == 0
    # all movement is the three 80/80/160 px jumps; jump to outer and back to center end there
    assert get(long, "distance_m", "inner") == pytest.approx(80 / 320)
    assert get(long, "distance_m", "outer") == pytest.approx(80 / 320)
    assert get(long, "distance_m", "center") == pytest.approx(160 / 320)


def test_immobility(cfg):
    cfg.smooth_s = 0
    cfg.duration_s = None
    # 2 s still, 1 s moving at 0.1 m/s, 0.5 s still (too short to count), 1 s moving
    x = np.r_[np.full(20, 320.0), 320 + 32 * np.arange(1, 11), np.full(5, 640.0), 640 + 32 * np.arange(1, 11)]
    long = metrics_for_trajectory(traj(x, np.full(len(x), 240.0)), cfg, "v", "test")
    assert get(long, "immobile_time_s") == pytest.approx(1.9)  # 19 zero-speed steps


def test_jump_rejection_and_gap_bridging():
    t = np.arange(10) / FPS
    x = np.array([0, 1, 2, 300, 4, np.nan, np.nan, 7, 8, 9], float)
    df = pd.DataFrame({"time_s": t, "x_px": x, "y_px": np.zeros(10)})
    cr = clean(df, px_per_m=100, max_speed_m_s=2.0, max_gap_s=0.5, smooth_s=0)
    assert cr.n_jumps == 1
    np.testing.assert_allclose(cr.x, np.arange(10))  # jump and gap both interpolated
    assert cr.raw_valid.sum() == 8


def test_long_gap_breaks_distance():
    t = np.arange(20) / FPS
    x = np.r_[np.arange(5.0), np.full(10, np.nan), 100 + np.arange(5.0)]
    df = pd.DataFrame({"time_s": t, "x_px": x, "y_px": np.zeros(20)})
    cr = clean(df, px_per_m=100, max_gap_s=0.5, smooth_s=0)
    assert np.isnan(cr.x[5:15]).all()
