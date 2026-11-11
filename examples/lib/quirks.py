"""The clock-change rule: how the two clock-change hours are marked, filled and kept out of the scores.

    from lib.quirks import mask_clock_changes, fill_clock_change_hours, filled_history
    from lib.quirks import seasonal_naive_at, mae_of, median_client_ratio, year_scores, clock_ratios

The rule (decisions B-2, item 1):
- mask_clock_changes(df) sets y to NaN on the clock-change hours: the March gap (quirk clock_change_gap) and the
  October repeat (quirk clock_change_repeat). A masked hour is never scored and never a training row.
- fill_clock_change_hours(history) fills each missing hour with the mean of the hour before and the hour after.
  It is for models that cannot take missing values (SeasonalNaive copies NaN through). It reads only the history
  it is given, so pass it the rows before a cutoff, never the whole series.
- filled_history(df, cutoff) does both for one cutoff: the masked history of the cutoff's window, filled.

The alternative of halving the October hour is not used: it assumes both real hours were alike, and the data
does not show that. The book's choice is a rule of thumb, labelled as one, not a measured optimum.

The helpers after filled_history set up the examples of Chapter 3: the daily clock comparisons, the census of the
clock-change hours and the year of Sunday windows. They score only what they are given, and they do not change the
rule above.
"""
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import SeasonalNaive

from lib.backtest import history_before
from lib.baselines import SEASON, mase_scores
from lib.settings import HISTORY_HOURS, HORIZON

CLOCK_CHANGE_QUIRKS = ("clock_change_gap", "clock_change_repeat")
COLS = ["unique_id", "ds", "y"]


def mask_clock_changes(df):
    """A copy of df (unique_id, ds, y, quirk) with y set to NaN on the clock-change hours."""
    out = df.copy()
    out.loc[out["quirk"].isin(CLOCK_CHANGE_QUIRKS), "y"] = float("nan")
    return out


def fill_clock_change_hours(history):
    """A copy of history (unique_id, ds, y) with each NaN y replaced by the mean of its neighbouring hours.

    The neighbours are the hour before and the hour after, within the same client, and only the non-missing
    ones count. Raises ValueError if a missing hour has no reading on either side.
    """
    out = history.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    missing = out["y"].isna()
    before = out.groupby("unique_id")["y"].shift(1)
    after = out.groupby("unique_id")["y"].shift(-1)
    fill = pd.concat([before[missing], after[missing]], axis=1).mean(axis=1)
    if fill.isna().any():
        raise ValueError("a missing hour has no reading on either side in the history")
    out.loc[missing, "y"] = fill
    return out


def filled_history(df, cutoff, hours=HISTORY_HOURS):
    """The history of one cutoff, from a masked df, with its clock-change hours filled from that history only."""
    return fill_clock_change_hours(history_before(df, cutoff, hours))


def seasonal_naive_at(df, cutoffs, make_history):
    """Seasonal naive (season 24) for each cutoff, from the history make_history gives that cutoff.

    df holds unique_id, ds, y and quirk. Returns the forecast hours with SeasonalNaive, and the y and quirk of df
    for those hours. A NaN in y (a masked hour) stays NaN, so the caller decides what is scored.
    """
    sf = StatsForecast(models=[SeasonalNaive(season_length=SEASON)], freq="h", n_jobs=1)
    parts = [sf.forecast(df=make_history(c)[COLS], h=HORIZON) for c in cutoffs]
    return pd.concat(parts, ignore_index=True).merge(df[COLS + ["quirk"]], on=["unique_id", "ds"])


def mae_of(s, col="SeasonalNaive"):
    """The mean absolute error of the forecast in column col, over the rows of s, as the rows are given."""
    return (s["y"] - s[col]).abs().mean()


def median_client_ratio(s, hour, other, col="SeasonalNaive"):
    """The median over clients of one client's error at the rows `hour` divided by its error at the rows `other`.

    hour and other are boolean masks over s. Each client counts once, so the largest client does not set the answer.
    """
    err = (s["y"] - s[col]).abs()
    at_hour = err[hour].groupby(s.loc[hour, "unique_id"]).mean()
    elsewhere = err[other].groupby(s.loc[other, "unique_id"]).mean()
    return float((at_hour / elsewhere).median())


def clock_ratios(df):
    """Each clock-change row of df (unique_id, ds, y, quirk, the whole series) with its ratio to its neighbours.

    The neighbours are the hour before and the hour after, within the same client. The ratio is NaN where both
    neighbours are zero. Returns unique_id, ds, quirk, y and ratio for the clock-change rows only.
    """
    s = df.sort_values(["unique_id", "ds"]).copy()
    neighbours = (s.groupby("unique_id")["y"].shift(1) + s.groupby("unique_id")["y"].shift(-1)) / 2
    s["ratio"] = s["y"] / neighbours.where(neighbours != 0)
    return s.loc[s["quirk"].isin(CLOCK_CHANGE_QUIRKS), ["unique_id", "ds", "quirk", "y", "ratio"]]


def year_scores(scored, scale_rows, col="SeasonalNaive"):
    """Pooled MAE and RMSE, and per-client MASE, over the scored hours of a year of windows.

    scored holds unique_id, ds, y (a masked hour has y NaN), quirk and col. A late starter's hours before its first
    reading are left out of every score, and so are masked hours. The MASE scale is scale_rows (from
    scoring.book_scale), and a client with no scale rows is left out of MASE. Returns labelled values in print order.
    """
    late = scored["quirk"] == "before_first_reading"
    counted = scored[~late & scored["y"].notna()]
    err = (counted["y"] - counted[col]).abs()
    share = err.groupby(counted["unique_id"]).sum()
    in_mase = counted[counted["unique_id"].isin(scale_rows["unique_id"])]
    per_client = mase_scores(in_mase[COLS], in_mase[["unique_id", "ds", col]], [col], scale_rows)[col]
    return {
        "scored hours": len(counted),
        "scored hours before a first reading": int(late.sum()),
        "MAE": err.mean(),
        "RMSE": ((counted["y"] - counted[col]) ** 2).mean() ** 0.5,
        "largest client by absolute error": share.idxmax(),
        "largest client's share of absolute error, percent": 100 * share.max() / share.sum(),
        "clients in MASE": len(per_client),
        "mean MASE": per_client.mean(),
        "lowest client MASE": per_client.min(),
        "highest client MASE": per_client.max(),
    }
