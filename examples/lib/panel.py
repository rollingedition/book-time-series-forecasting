"""One global model over all thirty clients: the panel, its windows and their checks (Chapter 6). The MASE scale is in
scoring.py.

    from lib.panel import ALL30, LAGS, panel_history, panel_windows, DECISION_CUTOFFS, FEBRUARY_CUTOFFS

A panel is the long-format frame of every client (unique_id, ds, y). For each cutoff one LightGBM is fitted on the
panel's history before that cutoff, with each client's target scaled (make_forecaster, scaled=True), and it forecasts
HORIZON hours for every client. Rows are sorted on unique_id and ds before each fit, so the fit order is fixed.

Two histories are possible for each client (see starts.py for the rule):
- start_rule=True: the history starts at the client's first reading, so a late starter's leading zeros are not rows.
  history_from_start reads the first reading from the rows before the cutoff only. It raises ValueError if a client
  has no non-zero reading before the cutoff, or fewer than 48 hours from its first reading.
- start_rule=False: every client gets the same HISTORY_HOURS hours before the cutoff, zeros included.

In both, a clock-change hour masked by quirks.mask_clock_changes is filled from the history before the cutoff only
(quirks.fill_clock_change_hours), as Chapter 3 does, so a window across a clock change is fitted, not refused.

The clients' lags and the optional client identifier are the only features. Calendar and holiday features are not
used: Chapter 5 found they did not lower the error over these 8 weeks of history.
"""
import pandas as pd

from lib.backtest import history_before
from lib.baselines import mase_scores, scale_history, seasonal_scale
from lib.electricity import CLIENTS
from lib.features import is_holiday, make_forecaster
from lib.quirks import fill_clock_change_hours, mask_clock_changes
from lib.scoring import actual_hours, book_scale
from lib.settings import HELD_OUT_WEEK, HISTORY_HOURS, HORIZON
from lib.starts import FLAG, first_reading, history_from_start

ALL30 = CLIENTS                                  # the thirty clients, in the order of electricity.py
LAGS = [24, 168]                                 # one day back and one week back: the lags Chapter 5 kept
LATE_STARTERS = ["MT_039", "MT_115", "MT_179"]   # first readings: 2013-10-03, 2014-01-14, 2013-12-20
FEBRUARY_CUTOFFS = pd.to_datetime(["2014-02-01", "2014-02-15", "2014-02-28"])  # Chapter 4's cutoffs for MT_115
# Four ordinary decision cutoffs before the held-out week (from 2014-12-25). Each scores 24 hours of an ordinary day:
# no Portuguese holiday (8 December is a holiday in 2014, and 25 December is the held-out week's first day).
DECISION_CUTOFFS = pd.to_datetime(["2014-11-25", "2014-11-26", "2014-12-02", "2014-12-03"])
# Holiday decision cutoffs (B-9, B-10): the 2014 Portuguese national holidays before Dec 25 whose 8-week history is
# complete for all thirty clients (Jan 1 is not: MT_115's first reading is Jan 14). A holiday choice is decided on these
# together with DECISION_CUTOFFS. Each forecasts a holiday's 24 hours from 00:00.
HOLIDAY_DECISION_CUTOFFS = pd.to_datetime(["2014-04-18", "2014-04-20", "2014-04-25", "2014-05-01", "2014-06-10",
                                           "2014-08-15", "2014-12-08"])
# The chapter's final model uses the client identifier: client-identifier-check keeps it (mean MASE falls on these
# windows). Set this by hand from that check's verdict; the chapter's model and Chapter 7 read it from here.
IDENTIFIER_KEPT = True
CHECK_POISONS = (0.0, 1e6)                       # the values a leak check writes over every reading after a cutoff


def panel_history(df, cutoff, start_rule=True):
    """The panel's history before cutoff, sorted on unique_id and ds, with clock-change hours filled.

    See the module docstring for the two rules. The fill reads only the history it is given.
    """
    hist = history_from_start(df, cutoff) if start_rule else history_before(df, cutoff)
    return fill_clock_change_hours(hist.sort_values(["unique_id", "ds"]).reset_index(drop=True))


def panel_windows(df, cutoffs, lags=LAGS, start_rule=True, scaled=True, client_id=False, name="global"):
    """Forecast HORIZON hours for every client from each cutoff, with one panel model fitted per cutoff.

    `df` holds unique_id, ds and y (clock-change hours masked, if they are to stay out). With client_id=True the
    client is a static categorical feature, the identifier of the identifier check. Returns unique_id, ds and one
    column named `name`. Raises ValueError if a window does not start at its cutoff.
    """
    parts = []
    for c in cutoffs:
        hist = panel_history(df, c, start_rule)[["unique_id", "ds", "y"]]
        static = []
        if client_id:
            hist = hist.assign(client=pd.Categorical(hist["unique_id"], categories=ALL30))
            static = ["client"]
        fcst = make_forecaster(lags, (), scaled=scaled)
        fcst.fit(hist, static_features=static)
        part = fcst.predict(h=HORIZON)
        if (part.groupby("unique_id")["ds"].min() != pd.Timestamp(c)).any():
            raise ValueError(f"panel forecast for cutoff {c} does not start at the cutoff")
        parts.append(part)
    out = pd.concat(parts, ignore_index=True)
    return out.rename(columns={"LGBMRegressor": name}).sort_values(["unique_id", "ds"]).reset_index(drop=True)


def with_and_without_start_rule(df, cutoffs, client_id=False):
    """The panel's forecasts with the start rule and without it, over the same windows, joined to the readings.

    Returns unique_id, ds, start, raw and y: both runs fit the same model on the same cutoffs, and differ only in
    their histories. Each hour appears once, because the windows do not overlap.
    """
    fc = panel_windows(df, cutoffs, start_rule=True, client_id=client_id, name="start").merge(
        panel_windows(df, cutoffs, start_rule=False, client_id=client_id, name="raw"),
        on=["unique_id", "ds"], validate="one_to_one")
    actual = df[(df["ds"] >= cutoffs[0]) & (df["ds"] < cutoffs[-1] + pd.Timedelta(hours=HORIZON))]
    return fc.merge(actual, on=["unique_id", "ds"], validate="one_to_one")


def mae_by_group(scored, model, late=LATE_STARTERS):
    """MAE of one model's forecasts: the late starters as a group, the other clients, and each late starter."""
    err = (scored["y"] - scored[model]).abs()
    is_late = scored["unique_id"].isin(late)
    out = {"late starters": err[is_late].mean(), "other clients": err[~is_late].mean()}
    for uid in late:
        out[uid] = err[scored["unique_id"] == uid].mean()
    return out


def mase_by_run(raw, scored, runs):
    """Per-client MASE of each run in `runs`, over the same hours, on the book scale read from the first scored cutoff."""
    first = scored["ds"].min()
    return mase_scores(scored[["unique_id", "ds", "y"]], scored[["unique_id", "ds", *runs]], list(runs),
                       book_scale(raw, first))


def counterfactual_scale_rows(raw, first_cutoff):
    """The rows a MASE scale would read without the start rule: the clock mask only, so a late starter's zeros stay.
    This is a labelled counterfactual for the chapter. The book's rule is scoring.book_scale."""
    return scale_history(mask_clock_changes(raw), first_cutoff)[["unique_id", "ds", "y"]]


def late_starter_facts(raw, df):
    """Labelled first readings and February zero shares of the late starters, as global-model-raw-starts prints them.

    df is the masked panel frame (unique_id, ds, y). A share is printed as a whole number of percent.
    """
    facts = {}
    firsts = first_reading(df)
    for uid in LATE_STARTERS:
        facts[f"first reading {uid}"] = f"{firsts[uid]:%Y-%m-%d}"
    for c in FEBRUARY_CUTOFFS:
        hist = history_before(df, c)
        for uid in ["MT_115", "MT_179"]:
            share = (hist.loc[hist["unique_id"] == uid, "y"] == 0).mean()
            facts[f"zero share of history {uid}, {c:%Y-%m-%d}"] = f"{100 * share:.0f}"
    return facts


def scale_facts(raw, cutoffs, client_id):
    """The late starters' MASE scales with and without the start rule, and the mean MASE of the standard-cutoff forecasts
    under each scale, as global-model-raw-starts prints them. Both scales are read from the first cutoff."""
    df = mask_clock_changes(raw)[["unique_id", "ds", "y"]]
    actual = actual_hours(df, cutoffs)
    fc = panel_windows(df, cutoffs, client_id=client_id)
    starts = book_scale(raw, cutoffs[0])
    no_rule = counterfactual_scale_rows(raw, cutoffs[0])
    facts = {}
    for uid in LATE_STARTERS:
        facts[f"MASE scale {uid}, start rule"] = seasonal_scale(starts)[uid]
        facts[f"MASE scale {uid}, counterfactual raw"] = seasonal_scale(no_rule)[uid]
    facts["mean MASE global, counterfactual raw scale, standard cutoffs"] = \
        mase_scores(actual, fc, ["global"], no_rule)["global"].mean()
    facts["mean MASE global, book scale, standard cutoffs"] = mase_scores(actual, fc, ["global"], starts)["global"].mean()
    return facts


def check_decision_hours(actual):
    """The number of holiday hours among a decision window's scored hours. Raises ValueError if one reaches the held-out
    week, or if one falls on a Portuguese holiday, so a decision cannot use either."""
    if actual["ds"].max() >= pd.Timestamp(HELD_OUT_WEEK[0]):
        raise ValueError("a decision window reaches the held-out week")
    holiday_hours = int(is_holiday(actual["ds"]).sum())
    if holiday_hours:
        raise ValueError(f"{holiday_hours} decision hours fall on a Portuguese holiday")
    return holiday_hours


def mae_change(scored, plain, other):
    """Each client's share of the pooled MAE change from `plain` to `other`, in kWh: its summed change over all hours."""
    d = (scored["y"] - scored[other]).abs() - (scored["y"] - scored[plain]).abs()
    return d.groupby(scored["unique_id"]).sum() / len(scored)


def _outcome(history_fn, frame, c):
    """The history of cutoff c, or None if the history function refuses it."""
    try:
        return history_fn(frame, c)
    except ValueError:
        return None


def start_rule_checks(raw, cutoffs, history_fn, removed_rows=False):
    """The refusal, leak and count checks of start-rule-leak-check, run on history_fn over cutoffs.

    raw holds the whole series with the quirk column. The refusals are compared with those the data's flag predicts:
    a cutoff is expected to be refused when a client's first non-flagged reading is under 48 hours before it. Every
    cutoff, refused or not, is poisoned from the cutoff on, with each value in CHECK_POISONS, and its history must not
    change. An accepted cutoff must hold every client. With removed_rows=True the rows the rule removes are counted too.
    Returns a dict of counts.
    """
    df = mask_clock_changes(raw)[["unique_id", "ds", "y"]]
    first_real = raw[raw["quirk"] != FLAG].groupby("unique_id")["ds"].min()
    r = dict(refused=0, expected=0, mismatch=0, accepted=0, missing=0, changed=0, removed=0, removed_nonzero=0)
    for c in cutoffs:
        hist = _outcome(history_fn, df, c)
        expected = any(c < first_real[uid] + pd.Timedelta(hours=48) for uid in ALL30)
        r["expected"] += int(expected)
        r["mismatch"] += int((hist is None) != expected)
        changed = False
        for poison in CHECK_POISONS:
            poisoned = df.assign(y=df["y"].where(df["ds"] < c, poison))
            other = _outcome(history_fn, poisoned, c)
            if (hist is None) != (other is None) or (hist is not None and not hist.equals(other)):
                changed = True
        r["changed"] += int(changed)
        if hist is None:
            r["refused"] += 1
            continue
        r["accepted"] += 1
        r["missing"] += int(hist["unique_id"].nunique() < len(ALL30))
        if removed_rows:
            before = history_before(df, c)
            gone = before.merge(hist[["unique_id", "ds"]], on=["unique_id", "ds"], how="left", indicator=True)
            gone = gone[gone["_merge"] == "left_only"]
            r["removed"] += len(gone)
            r["removed_nonzero"] += int((gone["y"] != 0).sum())
    return r


def leaky_history(frame, cutoff):
    """Deliberately broken, for the checks to catch: each first reading is looked up in the whole series, and a client
    with no rows in the window is dropped without a message."""
    cutoff = pd.Timestamp(cutoff)
    firsts = first_reading(frame)
    before = frame[frame["ds"] < cutoff]
    window = before[before["ds"] >= cutoff - pd.Timedelta(hours=HISTORY_HOURS)]
    hist = window[window["ds"] >= window["unique_id"].map(firsts)]
    counts = hist.groupby("unique_id").size()
    short = counts[counts < 48]
    if len(short):
        raise ValueError(f"history before {cutoff} has fewer than 48 hours for {dict(short)}")
    return hist.sort_values(["unique_id", "ds"]).reset_index(drop=True)
