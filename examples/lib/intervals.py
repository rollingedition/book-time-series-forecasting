"""Prediction intervals and buy-ahead quantiles for Chapter 8. Nothing else in lib/ changes for them.

    from lib.intervals import (CAL_CUTOFFS, COST_EXCESS, COST_SHORT, DECISION_CAL, bounds, buy_report,
                               builtin_intervals, calibration_facts, coverage, critical_fractile, design_coverage,
                               interval_score, paired, signed_ceiling, symmetric_ceiling, total_of)

The method, in words:
- paired(df, cutoffs) gives one row per client and hour: the Chapter 6 model's forecast f (panel_windows, one fit per
  cutoff, the client identifier kept), the reading y, the cutoff, the hour of the day and the residual e = y - f.
  Cutoffs are at 00:00 and the horizon is 24 hours, so the hour of the day is the horizon step.
- CAL_CUTOFFS are twelve calibration windows for the held-out evaluation: holiday-free days before the held-out week.
  DECISION_CAL are twelve holiday-free days before the decision windows. calibration_facts checks both.
- bounds(scored, cal, level) takes, for each hour of the day, the quantiles of the calibration residuals, pooled over
  the windows and clients. A level of L percent gets the bounds at (1 - L/100)/2 and (1 + L/100)/2. With scale (a Series
  by unique_id), each residual is first divided by its client's scale, so clients of very different size count alike,
  and the bound is multiplied back by the scored client's scale. With finite=True the levels are the finite-sample ones:
  floor((n + 1) * a) / n for the lower bound and ceil((n + 1) * (1 - a)) / n for the upper, for n residuals per hour
  and tail share a.
- total_of(pairs) sums the clients by hour: the total the supplier buys.
- builtin_intervals(df, cutoffs) runs mlforecast's own conformal intervals (PredictionIntervals, n_windows=3).

Scores (lower is better): interval_score is Gneiting and Raftery's interval score; quantile_loss is the quantile rule
at level tau; buy_cost is the supplier's two-cost loss, which equals (cu + co) times quantile_loss at cu / (cu + co).
"""
import numpy as np
import pandas as pd
from mlforecast.conformal_prediction import PredictionIntervals

from lib.features import make_forecaster
from lib.panel import ALL30, IDENTIFIER_KEPT, LAGS, check_decision_hours, panel_windows
from lib.scoring import actual_hours
from lib.settings import HISTORY_HOURS, HORIZON

CAL_CUTOFFS = pd.date_range("2014-12-09", periods=12, freq="D")  # Dec 9 to Dec 20, 2014: no holiday, before the held-out week
DECISION_CAL = pd.date_range("2014-11-09", periods=12, freq="D")  # Nov 9 to Nov 20, 2014: no holiday, before the decision windows
CLEAN_FROM = "2014-10-27"   # after the Oct 26, 2014 clock change, so no masked hour is inside a builtin window
COST_SHORT = 3.0    # illustrative cost of a kWh bought short (a stated assumption, not a price)
COST_EXCESS = 1.0   # illustrative cost of a kWh bought and not used (a stated assumption, not a price)


def critical_fractile(cu=COST_SHORT, co=COST_EXCESS):
    """The quantile that minimises the expected cost: cu / (cu + co)."""
    return cu / (cu + co)


def signed_ceiling(n):
    """The highest two-sided coverage that n signed residuals allow, if they are exchangeable. A band from the smallest
    to the largest residual covers n - 1 of the n + 1 places a new residual can take among them."""
    return (n - 1) / (n + 1)


def symmetric_ceiling(n):
    """The highest coverage of a band of plus and minus the largest of n absolute residuals: n of the n + 1 places."""
    return n / (n + 1)


def paired(df, cutoffs):
    """Forecasts, readings and residuals for every client and hour of the windows at cutoffs."""
    fc = panel_windows(df, cutoffs, client_id=IDENTIFIER_KEPT, name="f")
    act = actual_hours(df, cutoffs)[["unique_id", "ds", "y"]]
    out = fc.merge(act, on=["unique_id", "ds"], validate="one_to_one")
    out["cutoff"] = out["ds"].dt.floor("D")
    out["hour"] = out["ds"].dt.hour
    out["e"] = out["y"] - out["f"]
    return out


def total_of(pairs):
    """The hourly total over the clients: forecast, reading and residual, one row per window and hour."""
    t = pairs.groupby(["cutoff", "ds", "hour"], as_index=False)[["y", "f"]].sum()
    t["e"] = t["y"] - t["f"]
    return t


def _levels(alpha, n, finite):
    """The lower and upper quantile levels. With finite=True, the (n + 1) correction for n residuals per hour."""
    if not finite:
        return alpha, 1 - alpha
    return np.floor((n + 1) * alpha) / n, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)


def bounds(scored, cal, level, scale=None, finite=False):
    """Lower and upper bounds on the scored rows at `level` percent: the forecast plus the calibration quantiles of
    its hour. With scale (a Series by unique_id) the residuals are divided by the client's scale first."""
    alpha = (1 - level / 100) / 2
    n = int(cal.groupby("hour").size().iloc[0])
    lo_p, hi_p = _levels(alpha, n, finite)
    if scale is None:
        z, mult = cal["e"], 1.0
    else:
        z, mult = cal["e"] / cal["unique_id"].map(scale), scored["unique_id"].map(scale)
    lo = scored["f"] + mult * scored["hour"].map(z.groupby(cal["hour"]).quantile(lo_p))
    hi = scored["f"] + mult * scored["hour"].map(z.groupby(cal["hour"]).quantile(hi_p))
    return lo, hi


def coverage(y, lo, hi):
    """The share of readings inside [lo, hi]."""
    return float(((y >= lo) & (y <= hi)).mean())


def interval_score(y, lo, hi, level):
    """Gneiting and Raftery's interval score for a central (level)% interval: the width, plus 2 / alpha times the miss."""
    alpha = 1 - level / 100
    miss = np.maximum(lo - y, 0) + np.maximum(y - hi, 0)
    return float(((hi - lo) + (2 / alpha) * miss).mean())


def quantile_loss(y, q, tau):
    """The quantile rule at level tau (Gneiting and Raftery 2007): tau * (y - q) when y is at or above q, else (1 - tau) * (q - y)."""
    d = y - q
    return float(np.mean(np.maximum(tau * d, (tau - 1) * d)))


def buy_cost(y, q, cu=COST_SHORT, co=COST_EXCESS):
    """The supplier's cost per hour of buying q when y is read: cu per kWh short, co per kWh unused."""
    return cu * np.maximum(y - q, 0) + co * np.maximum(q - y, 0)


def level_bound(sc, cal, t):
    """The purchase at quantile t of the calibration residuals: each scored hour's forecast plus the t-quantile of the
    residuals in its hour of the day."""
    return sc["f"] + sc["hour"].map(cal.groupby("hour")["e"].quantile(t))


def client_sum_bound(pairs_sc, pairs_cal, tau):
    """The total's bound at tau when every client gets its own bound: the sum of the client bounds, by hour."""
    q = pairs_cal.groupby(["unique_id", "hour"])["e"].quantile(tau).rename("q").reset_index()
    m = pairs_sc.merge(q, on=["unique_id", "hour"], how="left", validate="many_to_one")
    return (m["f"] + m["q"]).groupby(m["ds"]).sum()


def design_coverage(level, n, construction, draws=200_000):
    """Coverage of a band built from n calibration residuals, by simulation under exchangeable normal errors (seed 0).
    "builtin" is mlforecast's band: the forecast plus and minus each absolute residual, 2n paths, np.quantile at the
    level. "signed" is the plain quantiles of n signed residuals, as bounds() takes them."""
    rng = np.random.default_rng(0)
    e_cal, e_test = rng.standard_normal((draws, n)), rng.standard_normal(draws)
    tail = (100 - level) / 200
    if construction == "builtin":
        s = np.abs(e_cal)
        lo, hi = np.quantile(np.concatenate([-s, s], axis=1), [tail, 1 - tail], axis=1)
    else:
        lo, hi = np.quantile(e_cal, [tail, 1 - tail], axis=1)
    return float(((e_test >= lo) & (e_test <= hi)).mean())


def calibration_facts(df, scored_cutoffs, cal_cutoffs=CAL_CUTOFFS):
    """Labelled checks on calibration windows: their count, their holiday hours (raises if any), and the hours they
    share with the scored windows (must be 0)."""
    cal_hours = actual_hours(df, cal_cutoffs)
    holidays = check_decision_hours(cal_hours)
    shared = set(cal_hours["ds"]) & set(actual_hours(df, scored_cutoffs)["ds"])
    return {"calibration windows": len(cal_cutoffs), "holiday hours in calibration": holidays,
            "calibration hours shared with scored": len(shared)}


def buy_report(pairs_cal, pairs_sc, tau=None):
    """The figures of the purchase example, as (label, value, digits) rows. The bands and the purchase are for the total.
    The finite-sample row is an outcome on the scored days only: the choice was made on the decision windows."""
    tau = critical_fractile() if tau is None else tau
    cal, sc = total_of(pairs_cal), total_of(pairs_sc)
    n = int(cal.groupby("hour").size().iloc[0])
    lo, hi = bounds(sc, cal, 80)
    lo_f, hi_f = bounds(sc, cal, 80, finite=True)
    day = sc["ds"].dt.floor("D")
    rows = [("scored hours", len(sc), 0),
            ("residuals per hour for the total", n, 0),
            ("highest coverage the total's residuals allow", signed_ceiling(n), 3),
            ("simulated coverage at 80, exchangeable errors", design_coverage(80, n, "signed"), 3),
            ("mean absolute error of the total, calibration days", cal["e"].abs().mean(), 2),
            ("mean absolute error of the total, scored days", sc["e"].abs().mean(), 2),
            ("total coverage at 80", coverage(sc["y"], lo, hi), 3),
            ("total coverage at 80 with finite-sample levels, not used", coverage(sc["y"], lo_f, hi_f), 3)]
    inside = (sc["y"] >= lo) & (sc["y"] <= hi)
    rows += [(f"total coverage at 80 on {d:%Y-%m-%d}", s, 3) for d, s in inside.groupby(day).mean().items()]
    rows += [(f"total mean absolute error on {d:%Y-%m-%d}", e, 2) for d, e in sc["e"].abs().groupby(day).mean().items()]
    for t in (0.5, tau, 0.9):
        q = level_bound(sc, cal, t)
        rows += [(f"mean cost at {t:.2f}", buy_cost(sc["y"], q).mean(), 2),
                 (f"hours short at {t:.2f}", (sc["y"] > q).mean(), 3)]
    q_tot = level_bound(sc, cal, tau)
    q_sum = sc["ds"].map(client_sum_bound(pairs_sc, pairs_cal, tau))
    rows += [("quantile loss at critical fractile, times both costs",
              (COST_SHORT + COST_EXCESS) * quantile_loss(sc["y"], q_tot, tau), 2),
             ("mean hourly total bound from the total", q_tot.mean(), 2),
             ("mean hourly total bound from the sum of client bounds", q_sum.mean(), 2),
             ("hours short at critical fractile, sum of client bounds", (sc["y"] > q_sum).mean(), 3),
             ("mean cost at critical fractile, sum of client bounds", buy_cost(sc["y"], q_sum).mean(), 2)]
    return rows


def builtin_intervals(df, cutoffs):
    """mlforecast's conformal_distribution intervals, PredictionIntervals(n_windows=3, h=24), refitted at each cutoff.
    The calibration for each cutoff is the three cutoffs before it (mlforecast's rule). Returns the cross-validation
    frame: the forecast, the reading and the bounds at 80 and 95, as mlforecast names them."""
    cut = pd.DatetimeIndex(cutoffs)
    d = df[(df["ds"] >= CLEAN_FROM) & (df["ds"] < cut[-1] + pd.Timedelta(hours=HORIZON))]
    d = d.assign(client=pd.Categorical(d["unique_id"], categories=ALL30))
    pi = PredictionIntervals(n_windows=3, h=HORIZON, method="conformal_distribution")
    return make_forecaster(LAGS, (), scaled=True).cross_validation(
        d, n_windows=len(cut), h=HORIZON, step_size=HORIZON, refit=True, static_features=["client"],
        keep_last_n=HISTORY_HOURS, prediction_intervals=pi, level=[80, 95])
