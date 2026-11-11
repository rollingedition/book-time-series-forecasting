"""Feature builders for the gradient-boosting chapters (Chapter 5 defines them; Chapter 6 reuses them).

    from lib.features import LGBM_PARAMS, CALENDAR, PT_HOLIDAYS, is_holiday, make_forecaster, mlforecast_windows
    from lib.features import lag_values, lag_table, rows_reading_missing, from_first_reading

A feature for an hour t is a reading that is known at the cutoff of the window that forecasts t. With a 24-hour
horizon, a lag of 24 or more hours reads an hour before the cutoff, so it is known; a lag of 1 is not.

- lag_values(y, k, index) reads y at each hour minus k. It reads only the readings in y, so a series cut at the
  cutoff gives the honest value, and the whole series gives the one a leak would use.
- lag_table(y, lags, index) stacks those lags and the hour of day into a feature table for the hours in index.
- is_holiday(times) is a date feature for mlforecast: 1 on a Portuguese national holiday, else 0. It is computed
  from the timestamp, so it needs no lag and is known in advance.
- make_forecaster(lags, date_features, scaled=True) is an MLForecast with one LGBMRegressor, set as LGBM_PARAMS says.
  With scaled=True each client's target is standardised by LocalStandardScaler. The scaler is fitted in fit, on the
  rows that fit receives (one client's history before the cutoff), and inverted in predict, so no statistic is read
  from after the cutoff. scaled=False gives the unscaled run that Chapter 5 shows as the failure that motivates it.
- mlforecast_windows(df, cutoffs, lags, date_features, h, scaled=True, filled=False) fits one forecaster per cutoff on
  that cutoff's history and forecasts HORIZON hours. It raises ValueError if a window does not start at its cutoff.
  Masked clock-change hours in the history make mlforecast refuse, unless filled=True: then each history is filled
  from its own readings by quirks.filled_history (decisions B-2 and B-5).
- fit_window_mae(y, lags, cutoffs, leaky) fits one LightGBM per window on the history and returns the MAE over the
  windows; with leaky=True its target rows read the whole series, which is the leak the book checks for.
- rows_reading_missing(y, lags) counts the rows whose lag reads a missing reading (a masked clock-change hour).
- from_first_reading(df) drops the rows flagged before_first_reading, so a late starter's leading zeros are not
  read as demand. It is the start rule's flag. score_feature_sets no longer uses it: its scale is scoring.book_scale.
- score_feature_sets(raw, cutoffs, lags, feature_sets, first_test_day) runs the Chapter 5 comparison: each feature
  set scaled and unscaled, with seasonal naive on the same windows, scored per client as MASE on book_scale.
"""
import holidays
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from mlforecast import MLForecast
from mlforecast.target_transforms import LocalStandardScaler
from statsforecast import StatsForecast
from statsforecast.models import SeasonalNaive

from lib.backtest import forecast_windows, history_before
from lib.baselines import SEASON, mase_scores
from lib.quirks import filled_history, mask_clock_changes
from lib.scoring import book_scale
from lib.settings import HISTORY_HOURS, HORIZON, SEED

DATA_YEARS = [2013, 2014]
PT_HOLIDAYS = holidays.country_holidays("PT", years=DATA_YEARS)  # 9 in each year: 18 in total at holidays 0.108
CALENDAR = ["hour", "dayofweek", "month"]  # mlforecast date features by name; dayofweek also carries the weekend
LGBM_PARAMS = dict(deterministic=True, force_row_wise=True, num_threads=1, random_state=SEED, verbosity=-1)


def is_holiday(times):
    """1 for each timestamp on a Portuguese national holiday, else 0. Works as an mlforecast date feature."""
    days = pd.DatetimeIndex(times).normalize()
    return np.array([day.date() in PT_HOLIDAYS for day in days], dtype=int)


def lag_values(y, k, index):
    """The reading k hours before each hour of index, read from the hourly series y (a Series indexed by ds).

    A reading missing from y, or an hour before y's first reading, is NaN.
    """
    return y.reindex(index - pd.Timedelta(hours=k)).to_numpy(dtype=float)


def lag_table(y, lags, index):
    """A feature table for the hours in index: one column per lag (lag_24, ...) and the hour of day."""
    table = pd.DataFrame({f"lag_{k}": lag_values(y, k, index) for k in lags}, index=index)
    table["hour"] = index.hour
    return table


def rows_reading_missing(y, lags):
    """How many hours of y have a lag that reads a missing reading in y. Used to count masked clock-change hours."""
    missing = y.isna().astype(float)
    hit = np.zeros(len(y), dtype=bool)
    for k in lags:
        hit |= missing.reindex(y.index - pd.Timedelta(hours=k)).to_numpy(dtype=float) == 1.0
    return int(hit.sum())


def make_forecaster(lags, date_features=(), scaled=True):
    """An MLForecast with one LGBMRegressor (LGBM_PARAMS) and one thread, on hourly data.

    scaled=True standardises each client's target: fitted in fit on the rows it receives, inverted in predict.
    """
    transforms = [LocalStandardScaler()] if scaled else None
    return MLForecast(models=[LGBMRegressor(**LGBM_PARAMS)], freq="h", lags=list(lags),
                      date_features=list(date_features), target_transforms=transforms, num_threads=1)


def mlforecast_windows(df, cutoffs, lags, date_features=(), h=HORIZON, scaled=True, filled=False):
    """Forecast h hours from each cutoff with a model fitted on that cutoff's history only. Returns them stacked.

    `df` holds unique_id, ds and y (masked, if clock-change hours are to be left out). The model's column is
    named LGBMRegressor; rename it before comparing windows of several feature sets. With filled=True a masked hour
    in a history is filled from that history (quirks.filled_history) instead of refused.
    """
    parts = []
    for c in cutoffs:
        hist = (filled_history(df, c) if filled else history_before(df, c))[["unique_id", "ds", "y"]]
        fcst = make_forecaster(lags, date_features, scaled=scaled)
        fcst.fit(hist, static_features=[])
        part = fcst.predict(h=h)
        if (part.groupby("unique_id")["ds"].min() != c).any():
            raise ValueError(f"forecast for cutoff {c} does not start at the cutoff")
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def fit_window_mae(y, lags, cutoffs, leaky, h=HORIZON):
    """MAE over the windows of one LightGBM on lag and hour features, fitted on each window's history.

    y is an hourly Series of one client. Training rows are the history hours, whose lags read only history. Target
    rows read the whole series if leaky, else the history cut at the cutoff. Returns a float.
    """
    errors = []
    for c in pd.DatetimeIndex(cutoffs):
        hist = pd.date_range(c - pd.Timedelta(hours=HISTORY_HOURS), c - pd.Timedelta(hours=1), freq="h")
        targets = pd.date_range(c, periods=h, freq="h")
        train = lag_table(y, lags, hist)
        train["y"] = y.reindex(hist).to_numpy()
        train = train.dropna()
        cols = [col for col in train.columns if col != "y"]
        model = LGBMRegressor(**LGBM_PARAMS).fit(train[cols], train["y"])
        source = y if leaky else y[y.index < c]
        pred = model.predict(lag_table(source, lags, targets)[cols])
        errors.append(np.abs(y.reindex(targets).to_numpy() - pred))
    return float(np.concatenate(errors).mean())


def from_first_reading(df):
    """The rows of df (quirk, unique_id, ds, y) from each client's first reading on. Drops before_first_reading rows."""
    return df[df["quirk"] != "before_first_reading"]


def score_feature_sets(raw, cutoffs, lags, feature_sets, first_test_day):
    """Fit each feature set scaled and unscaled on each cutoff, and score all of them on the same windows.

    `raw` is the loaded panel (unique_id, ds, y, quirk); clock-change hours are masked before scoring, and filled
    from each window's own history before fitting (B-2). `feature_sets` maps a name to its list of mlforecast date
    features. Seasonal naive is the baseline, and each set comes back twice: as its name (scaled, the default) and
    as unscaled_<name>. The MASE scale is scoring.book_scale over first_test_day: the rows of raw before it, with the
    start rule and the clock mask applied.

    Returns (forecasts, scored, per_client): the forecasts stacked over the windows, the forecasts joined to the
    actual readings, and the per-client MASE of each model, indexed by unique_id. Raises ValueError as the
    functions it calls do.
    """
    df = mask_clock_changes(raw)[["unique_id", "ds", "y"]]
    cutoffs = pd.to_datetime(cutoffs)
    naive = StatsForecast(models=[SeasonalNaive(season_length=SEASON)], freq="h", n_jobs=1)
    forecasts = forecast_windows(naive, df, cutoffs)
    for name, dates in feature_sets.items():
        part = mlforecast_windows(df, cutoffs, lags, dates, filled=True).rename(columns={"LGBMRegressor": name})
        forecasts = forecasts.merge(part, on=["unique_id", "ds"], validate="one_to_one")
        raw_part = mlforecast_windows(df, cutoffs, lags, dates, scaled=False, filled=True)
        raw_part = raw_part.rename(columns={"LGBMRegressor": f"unscaled_{name}"})
        forecasts = forecasts.merge(raw_part, on=["unique_id", "ds"], validate="one_to_one")
    models = ["SeasonalNaive", *feature_sets, *[f"unscaled_{name}" for name in feature_sets]]
    scored = forecasts.merge(df, on=["unique_id", "ds"], how="inner")
    scale_train = book_scale(raw, first_test_day)
    return forecasts, scored, mase_scores(df, forecasts, models, scale_train)
