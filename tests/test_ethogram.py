import numpy as np
import pandas as pd
import pytest

from behaviour.adapters.behaviours import frame_events
from behaviour.adapters.boris import read_events
from behaviour.ethogram import behaviour_rows, events_to_metrics, normalize_events

from .test_boris import tabular_v7


def events(rows):
    return normalize_events(pd.DataFrame(rows, columns=["video_id", "behaviour", "start", "stop", "event_type"]))


def test_bouts_merge_clip_and_point_window_boundaries():
    ev = events([("a", "grooming", 0, 4, "state"), ("a", "grooming", 3, 7, "state"),
                 ("a", "grooming", 9, 12, "state"), ("a", "defecation", 2, 2, "point"),
                 ("a", "defecation", 10, 10, "point")])
    rows = {r["metric"]: r["value"] for r in behaviour_rows(ev, 2, 10)}
    assert rows["grooming_time_s"] == 6
    assert rows["grooming_bouts"] == 2
    assert rows["grooming_mean_bout_s"] == 3
    assert rows["grooming_latency_s"] == 0
    assert rows["defecation_count"] == 1
    assert rows["defecation_latency_s"] == 0
    assert rows["defecation_frequency_per_min"] == 7.5


def test_absent_declared_behaviour_has_zero_count_unknown_latency(cfg):
    ev = frame_events(pd.DataFrame({"grooming": [0, 0, 0]}), "a", 10, ["grooming"])
    out = events_to_metrics(ev, cfg, "simba", {"grooming": "state"})
    values = out.set_index("metric").value
    assert values["grooming_time_s"] == 0 and values["grooming_bouts"] == 0
    assert np.isnan(values["grooming_latency_s"])
    assert values["duration_s"] == pytest.approx(0.3)
    assert not any(m.startswith("rearing") for m in values.index)


def test_frame_gaps_are_not_filled_with_behaviour():
    ev = frame_events(pd.DataFrame({"frame": [0, 1, 9, 10], "grooming": [1, 1, 1, 1]}),
                      "a", 10, ["grooming"])
    assert len(ev) == 2
    assert (ev.stop - ev.start).sum() == pytest.approx(0.4)
    with pytest.raises(ValueError, match="strictly increasing"):
        frame_events(pd.DataFrame({"frame": [1, 0], "grooming": [1, 1]}), "a", 10, ["grooming"])


def test_human_observers_stay_separate(cfg):
    ev = events([("a", "grooming", 0, 2, "state"), ("a", "grooming", 0, 5, "state")])
    ev["observer"] = ["Jin", "Oliver"]
    out = events_to_metrics(ev, cfg, "manual")
    assert set(out.tool) == {"manual_Jin", "manual_Oliver"}
    assert sorted(out.loc[out.metric == "grooming_time_s", "value"]) == [2, 5]


def test_boris_point_events_can_be_retained(tmp_path, cfg):
    p = tabular_v7(tmp_path / "events.csv", [(1, "Grooming", "START"), (4, "Grooming", "STOP"),
                                            (5, "Defecation", "POINT")], total_length=10)
    assert len(read_events(p)) == 1
    ev = read_events(p, include_points=True)
    out = events_to_metrics(ev, cfg, "manual")
    assert out.loc[out.metric == "grooming_time_s", "value"].iloc[0] == 3
    assert out.loc[out.metric == "defecation_count", "value"].iloc[0] == 1


def test_invalid_intervals_are_rejected():
    with pytest.raises(ValueError, match="positive duration"):
        events([("a", "grooming", 1, 1, "state")])
    with pytest.raises(ValueError, match="cannot combine"):
        behaviour_rows(events([("a", "grooming", 0, 2, "state"), ("a", "grooming", 3, 3, "point")]), 0, 5)
