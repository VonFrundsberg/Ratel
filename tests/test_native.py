import numpy as np
import pandas as pd
import pytest

from behaviour.classifier import labels_from_events, prediction_labels
from behaviour.config import Config
from behaviour.ethogram import normalize_events
from behaviour.native import analyse_track, complete_track


def test_missing_frame_rows_restore_unknown_time_without_filling_long_gaps():
    track = pd.DataFrame({"time_s": [0., .1, .8, .9], "x_px": [0., 0., 10., 10.], "y_px": 0.})
    full, fps = complete_track(track, 10)
    assert len(full) == 10 and full.x_px.isna().sum() == 6
    result = analyse_track(track, Config(100, [], duration_s=1, max_gap_s=0, smooth_s=0), "a", fps)
    values = result["metrics"].set_index("metric").value
    assert values.total_distance_m == 0
    assert values.tracked_fraction == pytest.approx(.4)
    assert values.lost_detection_bouts == 1
    assert "defecation_count" not in values
    assert result["coverage"].set_index("metric").loc["defecation_count", "status"] == "requires_annotations"


def test_unknown_pose_splits_bouts_and_short_bouts_are_removed():
    table = pd.DataFrame({"time_s": np.arange(8) / 10, "pose_valid": [1, 1, 0, 1, 1, 0, 1, 1],
                          "p_none": [.1, .1, .1, .1, .1, .1, .1, .9],
                          "p_grooming": [.9, .9, .9, .9, .9, .9, .9, .1]})
    y = prediction_labels(table, ["none", "grooming"], 0, .2)
    assert y.tolist() == [1, 1, -1, 1, 1, -1, 0, 0]


def test_observer_conflicts_do_not_become_background_training_labels():
    events = normalize_events(pd.DataFrame({"video_id": ["a"] * 3, "observer": ["one", "two", "three"],
                                            "behaviour": ["grooming", "rearing", "grooming"],
                                            "start": [0., 0., 0.], "stop": [1., 1., .5]}))
    y = labels_from_events(events, "a", np.array([.1, .7, 1.2]), ["none", "grooming", "rearing"])
    assert y.tolist() == [1, -1, 0]


def test_annotations_retain_observers_and_explicit_absence():
    track = pd.DataFrame({"time_s": np.arange(10) / 10, "x_px": 0., "y_px": 0.})
    events = normalize_events(pd.DataFrame({"video_id": ["a"], "observer": ["rater"],
                                            "behaviour": ["grooming"], "start": [.1], "stop": [.4]}))
    result = analyse_track(track, Config(100, [], duration_s=1), "a", 10,
                           events=events, declared={"grooming": "state", "defecation": "point"})
    manual = result["metrics"].query('tool == "native_manual_rater"').set_index("metric").value
    assert manual.grooming_time_s == pytest.approx(.3)
    assert manual.defecation_count == 0 and np.isnan(manual.defecation_latency_s)


def test_fst_manual_aliases_and_motion_quality_remain_separate():
    track = pd.DataFrame({"time_s": np.arange(20) / 10, "x_px": 0., "y_px": 0.,
                          "area_px": 100., "motion_px": 0.})
    track.loc[4:5, "motion_px"] = np.nan
    events = normalize_events(pd.DataFrame({"video_id": ["a", "a"],
                                            "behaviour": ["floating", "swimming"],
                                            "start": [0., 1.], "stop": [1., 2.]}))
    result = analyse_track(track, Config(100, [], duration_s=2), "a", 10, events=events, assay="fst")
    auto = result["metrics"].query('tool == "native"').set_index("metric").value
    manual = result["metrics"].query('tool == "native_manual"').set_index("metric").value
    assert auto.tracked_fraction == 1 and auto.fst_motion_valid_fraction == .9
    assert manual.fst_immobile_s == 1 and manual.fst_swim_s == 1 and manual.fst_uncoded_s == 0
