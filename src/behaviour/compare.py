"""Agreement between tools on the same videos.

For every (metric, zone) and every pair of tools (or every tool against a
reference such as manual scoring) over the videos both tools have:

* bias, sd of differences, 95% limits of agreement   (Bland-Altman; b - a)
* mean absolute error, and bias as % of the pair mean
* Pearson r                                          (association only)
* Lin's concordance correlation coefficient (CCC)    (agreement on the 1:1 line)
* ICC(A,1): two-way, absolute agreement, single rater (McGraw & Wong 1996)

r can be high while tools disagree systematically; CCC/ICC and the bias
are what matter for "can I swap one tool for the other".
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

KEY = ["metric", "zone"]


def lins_ccc(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return np.nan
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(), b.var()  # population variances, as in Lin (1989)
    cov = ((a - ma) * (b - mb)).mean()
    den = va + vb + (ma - mb) ** 2
    return float(2 * cov / den) if den > 0 else np.nan


def icc_a1(a: np.ndarray, b: np.ndarray) -> float:
    """ICC(A,1) for two raters."""
    y = np.column_stack([a, b])
    n, k = y.shape
    if n < 2:
        return np.nan
    grand = y.mean()
    ss_rows = k * ((y.mean(axis=1) - grand) ** 2).sum()
    ss_cols = n * ((y.mean(axis=0) - grand) ** 2).sum()
    ss_err = ((y - grand) ** 2).sum() - ss_rows - ss_cols
    msr = ss_rows / (n - 1)
    msc = ss_cols / (k - 1)
    mse = ss_err / ((n - 1) * (k - 1))
    den = msr + (k - 1) * mse + k * (msc - mse) / n
    return float((msr - mse) / den) if den > 0 else np.nan


def agreement(a: np.ndarray, b: np.ndarray) -> dict:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    n = len(a)
    d = b - a
    bias = d.mean() if n else np.nan
    sd = d.std(ddof=1) if n > 1 else np.nan
    pair_mean = np.concatenate([a, b]).mean() if n else np.nan
    r = float(np.corrcoef(a, b)[0, 1]) if n > 2 and a.std() > 0 and b.std() > 0 else np.nan
    return dict(
        n=n,
        mean_a=a.mean() if n else np.nan,
        mean_b=b.mean() if n else np.nan,
        bias=bias,
        bias_pct=100 * bias / pair_mean if n and pair_mean else np.nan,
        sd_diff=sd,
        loa_low=bias - 1.96 * sd,
        loa_high=bias + 1.96 * sd,
        mae=np.abs(d).mean() if n else np.nan,
        pearson_r=r,
        ccc=lins_ccc(a, b),
        icc_a1=icc_a1(a, b),
    )


def compare(long: pd.DataFrame, reference: str | None = None, kinds=("behaviour",)) -> pd.DataFrame:
    """Agreement table.  With ``reference`` every tool is compared to it (a = reference);
    otherwise all tool pairs are compared."""
    df = long[long["kind"].isin(kinds)].copy()
    df["zone"] = df["zone"].fillna("").astype(str)
    wide = df.pivot_table(index=["video_id", *KEY], columns="tool", values="value", aggfunc="last")
    tools = sorted(wide.columns)
    if reference is not None:
        if reference not in tools:
            raise ValueError(f"Reference tool {reference!r} not in data (tools: {tools})")
        pairs = [(reference, t) for t in tools if t != reference]
    else:
        pairs = list(combinations(tools, 2))

    out = []
    for (metric, zone), g in wide.groupby(level=KEY):
        for ta, tb in pairs:
            out.append(dict(metric=metric, zone=zone, tool_a=ta, tool_b=tb, **agreement(g[ta].to_numpy(), g[tb].to_numpy())))
    res = pd.DataFrame(out)
    if res.empty:
        return res
    res = res[res["n"] > 0]
    return res.sort_values(["metric", "zone", "tool_a", "tool_b"]).reset_index(drop=True)
