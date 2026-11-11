"""Forecast windows and their scores, shared by every chapter that backtests.

A window is one cutoff and the HORIZON hours after it. A cutoff's history is the hours before it, and the reading
at the cutoff itself is not history, so no reading from the target day reaches the model.

    from lib.backtest import held_out_cutoffs, history_before, forecast_windows, score_windows
    from lib.backtest import scoring_counts, window_mae, chosen_forms

The checks below raise ValueError on misuse. They do not change any value a correct call returns.
"""
import pandas as pd
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import mae, rmse

from lib.settings import HELD_OUT_WEEK, HISTORY_HOURS, HORIZON


def held_out_cutoffs():
    """The daily cutoffs at 00:00 of the held-out week: seven of them."""
    return pd.date_range(HELD_OUT_WEEK[0], HELD_OUT_WEEK[1], freq="D")


def history_before(df, cutoff, hours=HISTORY_HOURS):
    """The rows in the `hours` hours before `cutoff`, with ds in [cutoff - hours, cutoff).

    Raises ValueError unless every client has exactly `hours` rows in that range. A missing row would otherwise
    shorten the history silently, and a duplicated row would lengthen it.
    """
    start = cutoff - pd.Timedelta(hours=hours)
    out = df[(df["ds"] >= start) & (df["ds"] < cutoff)]
    counts = out.groupby("unique_id").size()
    bad = counts[counts != hours]
    if len(bad):
        raise ValueError(f"history before {cutoff} has {dict(bad)} rows per client; expected {hours}")
    return out


def forecast_windows(sf, df, cutoffs, hours=HISTORY_HOURS, h=HORIZON):
    """Forecast h hours from each cutoff, fitting on that cutoff's history only. Returns the forecasts stacked.

    `sf` is a statsforecast StatsForecast object. `df` needs unique_id, ds and y. Raises ValueError if a window's
    first forecast hour is not its cutoff.
    """
    parts = []
    for c in cutoffs:
        part = sf.forecast(df=history_before(df, c, hours), h=h)
        first = part.groupby("unique_id")["ds"].min()
        if (first != c).any():
            raise ValueError(f"forecast for cutoff {c} does not start at the cutoff: the history is not what it should be")
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def _joined(actual, forecasts):
    """The forecast hours that have an actual reading, with both. Raises on several series or overlapping windows."""
    if forecasts["unique_id"].nunique() != 1:
        raise ValueError(f"these helpers score one series at a time; got {forecasts['unique_id'].nunique()}")
    if forecasts.duplicated(["unique_id", "ds"]).any():
        raise ValueError("windows overlap: some hours are forecast twice, and each hour must be scored once")
    return forecasts.merge(actual[["unique_id", "ds", "y"]], on=["unique_id", "ds"], how="inner")


def score_windows(actual, forecasts, model):
    """MAE and RMSE over every scored hour, pooled over the windows. Returns a Series indexed by metric.

    Written for one series (the lead client). Windows must not overlap in ds, so each hour is scored once.
    """
    scored = _joined(actual, forecasts)
    result = evaluate(scored, metrics=[mae, rmse], models=[model])
    return result.set_index("metric")[model]


def scoring_counts(actual, forecasts, model, h=HORIZON):
    """The counts behind a score, as a dict with windows, forecast_hours, missing_forecasts and scored_hours.

    A window is counted by the calendar day its forecast hours fall on, so windows must be `h` hours from 00:00.
    scored_hours excludes missing forecasts, as score_windows does. Raises if a window is not `h` hours long, or if
    a forecast hour has no actual reading, since then the two sides would not cover the same hours.
    """
    per_day = forecasts["ds"].dt.normalize().value_counts()
    if (per_day != h).any():
        raise ValueError(f"every window must hold {h} hours; the counts per day are {per_day.tolist()}")
    joined = _joined(actual, forecasts)
    unmatched = len(forecasts) - len(joined)
    if unmatched:
        raise ValueError(f"{unmatched} forecast hours have no actual reading")
    missing = int(joined[model].isna().sum())
    return {"windows": int(per_day.size), "forecast_hours": len(forecasts),
            "missing_forecasts": missing, "scored_hours": len(joined) - missing}


def window_mae(actual, forecasts, model):
    """The MAE of each window, by the calendar day its forecast hours fall on. Returns a Series indexed by day.

    Missing forecasts are left out, as in score_windows, so count them with scoring_counts first.
    """
    joined = _joined(actual, forecasts)
    joined = joined[joined[model].notna()]
    err = (joined["y"] - joined[model]).abs()
    return err.groupby(joined["ds"].dt.normalize()).mean()


def chosen_forms(sf, df, cutoffs, hours=HISTORY_HOURS):
    """How many windows each model form was chosen for, as a dict from form name to count.

    Refits the model at each cutoff from that cutoff's history only, so it costs as much as the forecasts do.
    Only models that report a form (such as AutoETS) can be counted this way.
    """
    counts = {}
    for c in cutoffs:
        sf.fit(df=history_before(df, c, hours))
        form = sf.fitted_[0, 0].model_["method"]
        counts[form] = counts.get(form, 0) + 1
    return counts
