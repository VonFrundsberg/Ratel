import numpy as np
import pandas as pd
import pytest

from behaviour.compare import agreement, compare, icc_a1, lins_ccc

A = np.array([1, 2, 3, 4, 5.0])


def test_identical_tools_agree_perfectly():
    r = agreement(A, A)
    assert r["bias"] == 0 and r["ccc"] == pytest.approx(1) and r["icc_a1"] == pytest.approx(1)


def test_constant_offset():
    # hand-computed: ICC(A,1) = 5/6, CCC = 4/5, r = 1
    assert icc_a1(A, A + 1) == pytest.approx(5 / 6)
    assert lins_ccc(A, A + 1) == pytest.approx(0.8)
    r = agreement(A, A + 1)
    assert r["bias"] == pytest.approx(1) and r["pearson_r"] == pytest.approx(1)
    assert r["loa_low"] == pytest.approx(1) and r["loa_high"] == pytest.approx(1)


def test_compare_against_reference():
    rows = []
    for i, v in enumerate(A):
        rows += [
            dict(video_id=str(i), tool="manual", metric="entries", zone="center", value=v, kind="behaviour"),
            dict(video_id=str(i), tool="opencv", metric="entries", zone="center", value=v, kind="behaviour"),
            dict(video_id=str(i), tool="tailtrail", metric="entries", zone="center", value=v + 1, kind="behaviour"),
            dict(video_id=str(i), tool="tailtrail", metric="losses", zone="", value=3, kind="quality"),
        ]
    # one video missing from opencv: pairs use only shared videos
    df = pd.DataFrame(rows)
    df = df[~((df.video_id == "4") & (df.tool == "opencv"))]
    res = compare(df, reference="manual")
    assert set(res.tool_b) == {"opencv", "tailtrail"}
    assert "losses" not in set(res.metric)
    oc = res[res.tool_b == "opencv"].iloc[0]
    tt = res[res.tool_b == "tailtrail"].iloc[0]
    assert oc.n == 4 and oc.ccc == pytest.approx(1)
    assert tt.n == 5 and tt.bias == pytest.approx(1)
