"""The running loop of Chapter 10: a forecast made each day from its cutoff, logged when it is made, scored as its readings
arrive, and checked for drift against the backtest. Chapter 10's examples use it; no other chapter does.

    from lib.running import REFIT_EVERY, prepare, run_days, scale_from, with_actuals, band, daily_mase, coverage_by_day

Every time here is a data time, never the machine's clock. A cutoff is 00:00 of its day. A reading arrives at the end of
its hour (ds + 1 hour), except the late block (LATE_CLIENT, LATE_FROM to LATE_TO), which arrives at LATE_ARRIVES.

feed(df, now) is what the loop knows at a cutoff: the readings that have arrived by then. A reading that has not arrived
sits at the end of its client's history. Its value is filled for the forecast's inputs (LATE_FILL), and it is left out of
the refit. Any other gap raises ValueError, because a gap in the middle would shift every lag.

run_days(df, days) forecasts each day from its own cutoff and writes the log rows then, before the readings of that day
exist in the log. Refits fall on a fixed cycle: REFIT_ANCHOR, a Monday, and every REFIT_EVERY days after it. A day's
forecast uses the model fitted at the latest refit day on or before it, so the refit dates do not depend on where a run
starts. Between refits the model keeps its trees, and each forecast reads the readings known at its own cutoff.
"""
import pandas as pd

from lib.baselines import seasonal_scale
from lib.features import make_forecaster
from lib.intervals import bounds
from lib.panel import ALL30
from lib.quirks import fill_clock_change_hours, mask_clock_changes
from lib.scoring import book_scale
from lib.settings import HISTORY_HOURS, HORIZON

LAGS = [24, 168]                                             # the lags Chapter 5 and 6 kept
REFIT_EVERY = 7                                              # days between refits; set by refit_cadence_decision
REFIT_ANCHOR = pd.Timestamp("2014-11-24")                    # a Monday: the weekly refits fall on Mondays
LATE_FILL = "week"                                           # how a late reading is filled; set by late_fill_decision
WEEK_HOURS = 168
DECISION_DAYS = pd.to_datetime(["2014-11-25", "2014-11-26", "2014-12-02", "2014-12-03"])
FIRST_SCORED_DAY = DECISION_DAYS[0]                          # the chapter's one backtest starts here
HELD_OUT_DAYS = pd.date_range("2014-12-25", periods=7, freq="D")
FLAG_FACTOR = 2.0                                            # the chapter's drift rule of thumb
CAL_DAYS = pd.DatetimeIndex(pd.date_range("2014-12-09", periods=12, freq="D"))  # = intervals.CAL_CUTOFFS

LATE_CLIENT = "MT_014"                                       # the lead client
LATE_FROM = pd.Timestamp("2014-12-28 18:00")                 # its last six readings of Dec 28 ...
LATE_TO = pd.Timestamp("2014-12-29 00:00")
LATE_ARRIVES = pd.Timestamp("2014-12-29 09:00")              # ... arrive after the Dec 29 cutoff


def arrivals(df):
    """When each reading arrives: the end of its hour, or LATE_ARRIVES for the late block."""
    arrive = df["ds"] + pd.Timedelta(hours=1)
    late = (df["unique_id"] == LATE_CLIENT) & (df["ds"] >= LATE_FROM) & (df["ds"] < LATE_TO)
    return arrive.mask(late, LATE_ARRIVES)


def late_copy(df, day):
    """df with the lead client's last six readings of the day before `day` arriving at 09:00 on `day`, as on Dec 28."""
    out = df.copy()
    start = day - pd.Timedelta(days=1) + pd.Timedelta(hours=18)
    block = (out["unique_id"] == LATE_CLIENT) & (out["ds"] >= start) & (out["ds"] < day)
    out.loc[block, "arrives"] = day + pd.Timedelta(hours=9)
    return out


def refit_day(day, every=REFIT_EVERY):
    """The cutoff of the model that forecasts day: the latest refit day on or before it. Refit days are REFIT_ANCHOR and
    every `every` days after it."""
    steps = (pd.Timestamp(day) - REFIT_ANCHOR).days // every
    return REFIT_ANCHOR + pd.Timedelta(days=steps * every)


def prepare(raw):
    """The readings the loop reads: unique_id, ds, y, the client identifier and arrives, with the clock-change hours
    set to NaN (quirks.mask_clock_changes). feed fills them from the history before each cutoff, as Chapter 3 does."""
    df = mask_clock_changes(raw)[["unique_id", "ds", "y"]].copy()
    df["client"] = pd.Categorical(df["unique_id"], categories=ALL30)
    df["arrives"] = arrivals(df)
    return df


def scale_from(raw, first_scored_day=FIRST_SCORED_DAY):
    """The MASE scale per client, read only from readings before the first scored day. Every run in this chapter uses
    this one scale, so a decision day and a held-out day are scored on the same yardstick."""
    return seasonal_scale(book_scale(raw, first_scored_day))


def feed(df, now, hours=HISTORY_HOURS, fill=LATE_FILL):
    """What the loop knows at the cutoff now. Returns (train, known, pending, read_by).

    train: the arrived readings of the `hours` hours before now, clock-change hours filled from them, for a refit.
    known: train plus the pending readings, filled for a forecast. With fill "forward" each takes its client's last
           arrived reading; with "week" it takes the reading of the same hour a week earlier.
    pending: how many readings had not arrived by now.
    read_by: the latest arrival among the readings in train, which is the latest time the loop's inputs were in hand.
    """
    win = df[(df["ds"] >= now - pd.Timedelta(hours=hours)) & (df["ds"] < now)].sort_values(["unique_id", "ds"])
    late = win["arrives"] > now
    arrived, pending = win[~late], win[late]
    last_arrived = pending["unique_id"].map(arrived.groupby("unique_id")["ds"].max())
    if (pending["ds"] <= last_arrived).any():
        raise ValueError(f"a reading that has not arrived by {now} sits inside a client's history")
    train = fill_clock_change_hours(arrived[["unique_id", "ds", "y"]].reset_index(drop=True))
    train["client"] = pd.Categorical(train["unique_id"], categories=ALL30)
    last_value = train.groupby("unique_id")["y"].last()
    value = pd.Series(pending["unique_id"].map(last_value).to_numpy())
    if fill == "week":
        week_ago = pd.MultiIndex.from_arrays([pending["unique_id"], pending["ds"] - pd.Timedelta(hours=WEEK_HOURS)])
        prior = pd.Series(train.set_index(["unique_id", "ds"])["y"].reindex(week_ago).to_numpy())
        value = prior.fillna(value)
    elif fill != "forward":
        raise ValueError(f"unknown fill rule {fill!r}")
    fill_rows = pending[["unique_id", "ds"]].assign(y=value.to_numpy())
    fill_rows["client"] = pd.Categorical(fill_rows["unique_id"], categories=ALL30)
    known = pd.concat([train, fill_rows], ignore_index=True).sort_values(["unique_id", "ds"]).reset_index(drop=True)
    if (known.groupby("unique_id").size() != hours).any():
        raise ValueError(f"the history at {now} does not hold {hours} hours for every client")
    read_by = arrived["arrives"].max()
    return train, known, len(pending), read_by


def fit_at(df, fit_day, leak=False):
    """The model re-estimated on the history known at fit_day. Returns the model and the latest arrival among the readings
    it read. With leak=True it reads the history known one day later, so it has seen the readings of the day it
    forecasts. That is the leak the checks must catch."""
    source = pd.Timestamp(fit_day) + pd.Timedelta(days=1) if leak else pd.Timestamp(fit_day)
    train, _, _, read_by = feed(df, source)
    model = make_forecaster(LAGS, (), scaled=True)
    model.fit(train[["unique_id", "ds", "y", "client"]], static_features=["client"])
    return model, read_by


def forecast_at(model, df, day, fill=LATE_FILL):
    """The forecast from day's cutoff with a fitted model, from the readings known at that cutoff. Returns the forecast
    (unique_id, ds, f), the number of readings filled for it, and the latest arrival among the readings it read."""
    _, known, pending, read_by = feed(df, day, fill=fill)
    part = model.predict(h=HORIZON, new_df=known[["unique_id", "ds", "y", "client"]])
    return part.rename(columns={"LGBMRegressor": "f"}), pending, read_by


def run_days(df, days, refit_every=None, leak=False, fill=LATE_FILL):
    """Forecast each day from its own cutoff and log every forecast when it is made.

    Returns the log: one row per client and hour, with cutoff, written_at (the cutoff, the label of the row's write),
    refit (the cutoff the model was fitted at), fit_read_by (the latest arrival among the readings the model was fitted
    on), read_by (the latest arrival among the readings the forecast read), filled (readings filled for that forecast)
    and f (the forecast).
    """
    every = REFIT_EVERY if refit_every is None else refit_every
    models, parts = {}, []
    for day in pd.DatetimeIndex(days):
        fit_day = day if leak else refit_day(day, every)
        if fit_day not in models:
            models[fit_day] = fit_at(df, fit_day, leak=leak)
        model, fit_read_by = models[fit_day]
        part, pending, read_by = forecast_at(model, df, day, fill=fill)
        parts.append(part.assign(cutoff=day, written_at=day, refit=fit_day, fit_read_by=fit_read_by,
                                 read_by=read_by, filled=pending))
    return pd.concat(parts, ignore_index=True)


def with_actuals(log, df):
    """The log joined to the readings: y, when each arrived, the hour of the day, the error e and its absolute value ae.
    Each client-hour appears once. Readings that are NaN (clock-change hours) have no error."""
    out = log.merge(df[["unique_id", "ds", "y", "arrives"]], on=["unique_id", "ds"], validate="one_to_one")
    out["hour"] = out["ds"].dt.hour
    out["e"] = out["y"] - out["f"]
    out["ae"] = out["e"].abs()
    return out


def band(scored, cal, scale, level=80):
    """Each logged hour's interval: its forecast plus the scaled calibration residuals of its hour, pooled over the
    clients (Chapter 8's rule), with cal from the calibration days, which are earlier than every scored day."""
    lo, hi = bounds(scored, cal.dropna(subset=["e"]), level, scale=scale)
    return scored.assign(lo=lo, hi=hi)


def coverage_by_day(bands):
    """The share of each day's readings inside its interval, by cutoff."""
    inside = ((bands["y"] >= bands["lo"]) & (bands["y"] <= bands["hi"])).astype(float)
    return inside.groupby(bands["cutoff"]).mean()


def daily_mase(scored, scale):
    """Each day's MASE: for each client, the mean absolute error over the day's hours divided by the client's scale,
    then the mean over the clients. Indexed by cutoff."""
    per = scored.dropna(subset=["y"]).groupby(["cutoff", "unique_id"])["ae"].mean().reset_index()
    per["mase"] = per["ae"] / per["unique_id"].map(scale)
    return per.groupby("cutoff")["mase"].mean()


def flagged(daily, reference, factor=FLAG_FACTOR):
    """The days whose MASE is above factor times the reference, which is the median daily MASE of earlier days."""
    return daily[daily > factor * reference].index
