"""Shift tests for features: the history rows, and the target hours a day-ahead forecast must make.

    from lib.leakage import history_test, target_test, lag, peeking_mean, trailing_mean

history_test(y, feature) computes the feature from the history cut at each evaluation cutoff, and from the whole
series, and compares the rows of the history. target_test(y, feature) does the same for the HORIZON hours after each
cutoff, with every reading from the cutoff on removed first, so it asks whether the feature for each target hour can
be known at the cutoff. A lag of at least HORIZON passes it. A lag of one hour fails it on every target hour but the
first, because the hour it reads is in the day being forecast.
"""
import numpy as np
import pandas as pd

from lib.settings import EVAL_CUTOFFS, HISTORY_HOURS, HORIZON


def trailing_mean(s):
    """Honest: the mean of the 24 hours up to and including each hour, so it reads nothing after that hour."""
    return s.rolling(24).mean()


def peeking_mean(s):
    """Leaking: the same mean moved one hour forward, so each hour's value reads the next hour's reading."""
    return s.rolling(24).mean().shift(-1)


def lag(hours):
    """A feature that reads the reading `hours` hours earlier."""
    return lambda s: s.shift(hours)


def _same(cut, full):
    return np.isclose(cut, full, equal_nan=True)


def history_test(y, feature):
    """Rows compared and rows that differ, over the 8 weeks before each evaluation cutoff."""
    compared = differing = 0
    for cutoff in pd.to_datetime(EVAL_CUTOFFS):
        cut = feature(y[y.index < cutoff]).tail(HISTORY_HOURS)
        full = feature(y).loc[cut.index]
        compared += len(cut)
        differing += int((~_same(cut, full)).sum())
    return compared, differing


def target_test(y, feature):
    """Target hours compared and target hours that differ, over the HORIZON hours after each evaluation cutoff."""
    compared = differing = 0
    for cutoff in pd.to_datetime(EVAL_CUTOFFS):
        targets = y.index[(y.index >= cutoff) & (y.index < cutoff + pd.Timedelta(hours=HORIZON))]
        cut = feature(y.where(y.index < cutoff))[targets]
        full = feature(y)[targets]
        compared += len(targets)
        differing += int((~_same(cut, full)).sum())
    return compared, differing
