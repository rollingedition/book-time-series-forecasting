"""Baselines and the MASE scale, shared by the chapters that set and score baselines.

    from lib.baselines import BASELINES, baseline_forecasts, scale_history, seasonal_scale, mase_scores

A baseline is a forecast with no fitted model in it. Three are used here:
    Naive            repeats the last reading for every hour of the horizon
    SeasonalNaive    repeats the reading at the same hour yesterday (season 24)
    AverageOfWeeks   the mean of the readings at the same hour on the same weekday, 1 to 4 weeks earlier;
                     computed by hand in pandas, not by a library method

The MASE scale is read from history only. scale_history() keeps the rows before the first test day, and
seasonal_scale() takes the mean absolute difference between each reading and the reading one season before it,
per client. mase_scores() scores with utilsforecast's mase, which takes its scale from the train_df it is given.
Every function here accepts one client or many; see each signature.
"""
import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import Naive, SeasonalNaive
from utilsforecast.losses import mase

from lib.backtest import forecast_windows, history_before
from lib.settings import HORIZON

SEASON = 24                                  # one day, in hours: the seasonal lag for MASE and SeasonalNaive
WEEKS = 4                                    # weeks averaged by AverageOfWeeks
BASELINES = ["Naive", "SeasonalNaive", "AverageOfWeeks"]


def average_of_weeks(df, cutoffs, weeks=WEEKS, h=HORIZON):
    """Forecast each cutoff's next h hours as the mean of the same hour on the same weekday, 1 to `weeks` weeks back.

    `df` holds one series (unique_id, ds, y). Each window reads only its own history (history_before), and every
    lagged reading is before its cutoff. A lag missing from the history raises KeyError rather than being skipped.
    Returns unique_id, ds and AverageOfWeeks.
    """
    parts = []
    for c in cutoffs:
        hist = history_before(df, c)
        series = hist.set_index("ds")["y"]
        targets = pd.date_range(c, periods=h, freq="h")
        lagged = [series.loc[targets - pd.Timedelta(weeks=w)].to_numpy() for w in range(1, weeks + 1)]
        parts.append(pd.DataFrame({"unique_id": hist["unique_id"].iloc[0], "ds": targets,
                                   "AverageOfWeeks": sum(lagged) / weeks}))
    return pd.concat(parts, ignore_index=True)


def baseline_forecasts(df, cutoffs, h=HORIZON):
    """The three baselines for every window. `df` (unique_id, ds, y) may hold one client or many.

    Returns unique_id, ds, then one column per name in BASELINES. Each window is forecast from its own history.
    """
    sf = StatsForecast(models=[Naive(), SeasonalNaive(season_length=SEASON)], freq="h", n_jobs=1)
    stat = forecast_windows(sf, df, cutoffs, h=h)
    hand = average_of_weeks(df, cutoffs, h=h)
    return stat.merge(hand, on=["unique_id", "ds"], how="inner", validate="one_to_one")


def scale_history(df, first_test_day):
    """The rows a MASE scale may read: every reading strictly before the first test day.

    `df` holds unique_id, ds, y and any other columns (quirk). Returns the rows as they are, one or many clients.
    """
    return df[df["ds"] < pd.Timestamp(first_test_day)]


def check_hourly(df):
    """Raise ValueError unless each client's rows in df are one unbroken hourly series.

    `df` needs unique_id and ds. Each client is checked on its own, so one frame can hold several. A gap or a
    duplicated hour fails. This is a raise, not an assert, so it still runs under python -O.
    """
    for uid, part in df.sort_values(["unique_id", "ds"]).groupby("unique_id"):
        steps = part["ds"].diff().dropna()
        if not (steps == pd.Timedelta(hours=1)).all():
            raise ValueError(f"{uid}: readings are not one unbroken hourly series")


def seasonal_scale(train, m=SEASON):
    """The seasonal MASE scale for each client: the mean of |y(t) - y(t - m)| over the rows of `train`.

    `train` needs unique_id, ds and y. A pair with a missing reading (NaN, such as a clock-change hour masked by
    mask_clock_changes) is left out, and nothing else is. Returns a Series indexed by unique_id, one value per
    client. Raises ValueError (via check_hourly) if a client is not an unbroken hourly series, and if no pair is
    left to average for a client.
    """
    check_hourly(train)
    out = {}
    for uid, part in train.sort_values(["unique_id", "ds"]).groupby("unique_id"):
        y = part["y"].to_numpy(dtype=float)
        diff = np.abs(y[m:] - y[:-m])
        diff = diff[~np.isnan(diff)]
        if diff.size == 0:
            raise ValueError(f"{uid}: no pair of readings {m} hours apart is left to scale by")
        out[uid] = float(diff.mean())
    return pd.Series(out, name="scale")


def mase_scores(actual, forecasts, models, train_df, m=SEASON):
    """MASE for each client and model: the mean of |actual - forecast| over the scored hours, over the scale.

    `actual` needs unique_id, ds, y. `forecasts` needs unique_id, ds and one column per name in `models`. It may
    hold one client or many, and several windows, but each hour at most once. Every forecast hour must have an
    actual reading. The scale comes from `train_df`, as passed: utilsforecast takes it from train_df with no time
    filter unless `forecasts` has a cutoff column. So pass only the rows the scale may read, and mask any clock-change
    hours first if they should not count. A pair with a NaN is left out of the scale, as in seasonal_scale.

    Raises ValueError if train_df is not an unbroken hourly series per client, if a forecast hour has no actual
    reading, or if an hour is forecast twice. Returns a DataFrame indexed by unique_id, one column per model.
    """
    models = list(models)
    check_hourly(train_df)
    if forecasts.duplicated(["unique_id", "ds"]).any():
        raise ValueError("windows overlap: some hours are forecast twice, and each hour must be scored once")
    joined = forecasts.merge(actual[["unique_id", "ds", "y"]], on=["unique_id", "ds"], how="left")
    if joined["y"].isna().any():
        raise ValueError("some forecast hours have no actual reading")
    train = train_df[["unique_id", "ds", "y"]].sort_values(["unique_id", "ds"])
    result = mase(df=joined, models=models, seasonality=m, train_df=train)
    return result.set_index("unique_id")[models]
