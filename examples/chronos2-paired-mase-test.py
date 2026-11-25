"""Tests whether the thirty clients can tell Chronos-2's mean MASE from the global model's and from seasonal naive's:
the exact sign-flip p-value of each mean difference, and the smallest mean difference the test detects at 80% power.
The point rests on the p-value and detectable-difference lines, read beside each mean difference."""
from lib.compare import global_forecasts, recorded_forecasts, score, seasonal_naive
from lib.electricity import load
from lib.panel import ALL30
from lib.quirks import mask_clock_changes
from lib.settings import EVAL_CUTOFFS, show

from statistics import NormalDist

import numpy as np
import pandas as pd

raw = load(ALL30)
df = mask_clock_changes(raw)[["unique_id", "ds", "y"]]
cutoffs = pd.DatetimeIndex(EVAL_CUTOFFS)
forecasts = recorded_forecasts().merge(seasonal_naive(df, cutoffs), on=["unique_id", "ds"], validate="one_to_one")
forecasts = forecasts.merge(global_forecasts(df, cutoffs), on=["unique_id", "ds"], validate="one_to_one")
per_client, _ = score(raw, forecasts, ["Chronos", "SeasonalNaive", "Global"], ALL30)


def flip_sums(values):
    """Every sum of the values with each sign flipped or kept: 2**len(values) sums, sorted."""
    signs = 1 - 2 * ((np.arange(2 ** len(values))[:, None] >> np.arange(len(values))) & 1)
    return np.sort(signs @ values)


def share_at_least(halves, total):
    """Share of all 2**30 sign flips whose sum is at least `total` from zero; the halves meet in the middle."""
    low, high = halves
    above = len(high) - np.searchsorted(high, total - low - 1e-9)      # 1e-9: a tie counts as at least as far
    below = np.searchsorted(high, -total - low + 1e-9)
    return (above + below).sum() / (len(low) * len(high))


z_power = NormalDist().inv_cdf(0.80)                                   # 80% power, under a normal approximation
for other in ["Global", "SeasonalNaive"]:
    difference = (per_client["Chronos"] - per_client[other]).to_numpy()  # below zero: Chronos-2 scores lower
    assert len(difference) == 30
    halves = flip_sums(difference[:15]), flip_sums(difference[15:])
    show(f"mean MASE difference Chronos minus {other}", difference.mean(), 3)
    p_value = share_at_least(halves, abs(difference.sum()))
    show(f"sign-flip p-value Chronos vs {other}", f"{p_value:.4f}" if p_value >= 1e-4 else "below 0.0001")
    low, high = 0.0, np.abs(difference).sum()                         # the 5% two-sided critical sum, by bisection
    for _ in range(60):
        middle = (low + high) / 2
        low, high = (low, middle) if share_at_least(halves, middle) <= 0.05 else (middle, high)
    detectable = high / 30 + z_power * difference.std(ddof=1) / np.sqrt(30)
    show(f"smallest detectable mean difference Chronos vs {other}", detectable, 3)
