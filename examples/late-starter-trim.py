"""Backtest AutoETS for MT_115, a late starter, on three February cutoffs: once on its raw eight weeks and once on its history from its first reading.
The start-rule counts and the three-window MAE lines, printed last, are the figures the prose quotes."""
import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoETS

from lib.backtest import history_before
from lib.electricity import load
from lib.quirks import fill_clock_change_hours, mask_clock_changes
from lib.settings import HORIZON, SEED, SUBSET8, show
from lib.starts import check_start_rule, history_from_start

LATE = "MT_115"                                             # the one late starter in SUBSET8, first non-zero reading on Jan 14, 2014
# February cutoffs: at the December cutoffs every hour of the history is a reading, so the rule changes nothing there
CUTOFFS = pd.to_datetime(["2014-02-01", "2014-02-15", "2014-02-28"])

np.random.seed(SEED)                                        # the seed from lib/settings.py, set before any fit
dropped, flagged = check_start_rule(load(SUBSET8))           # raises unless the rule drops exactly the flagged rows, so the counts agree
show("start rule drops, SUBSET8", dropped)
show("flagged before_first_reading, SUBSET8", flagged)
masked = mask_clock_changes(load([LATE]))[["unique_id", "ds", "y", "quirk"]]


def fit_and_score(hist, c):
    """Fit AutoETS on one history, forecast the day after c, and score it on that day."""
    sf = StatsForecast(models=[AutoETS(season_length=24)], freq="h", n_jobs=1)
    sf.fit(df=fill_clock_change_hours(hist)[["unique_id", "ds", "y"]])   # clock hours filled from the history's neighbours, as elsewhere
    fc = sf.predict(h=HORIZON)
    actual = masked[(masked["ds"] >= c) & (masked["ds"] < c + pd.Timedelta(hours=HORIZON))]   # the day's 24 readings, to score against
    joined = fc.merge(actual[["ds", "y"]], on="ds", how="left")
    # the form's method name, the mean forecast, the day's MAE, and the mean of the readings
    return sf.fitted_[0, 0].model_["method"], float(joined["AutoETS"].mean()), float((joined["y"] - joined["AutoETS"]).abs().mean()), float(joined["y"].mean())


maes = {"raw": [], "start": []}
for c in CUTOFFS:
    raw_hist = history_before(masked, c)                     # the eight weeks before c, zeros included; it insists on exactly 1,344 rows
    # the share of hours that read zero, including the leading zeros before the first reading
    show(f"zero percent before {c:%Y-%m-%d}", round(100 * (raw_hist["y"] == 0).mean()))
    for label, hist in [("raw", raw_hist), ("start", history_from_start(masked, c))]:
        form, fc_mean, window_mae, actual_mean = fit_and_score(hist, c)   # the start history is shorter where the client starts late
        show(f"{label} history hours {c:%Y-%m-%d}", len(hist))
        show(f"{label} form {c:%Y-%m-%d}", form)
        show(f"{label} mean forecast {c:%Y-%m-%d}", fc_mean)
        show(f"{label} mean actual {c:%Y-%m-%d}", actual_mean)
        show(f"MAE {label} {c:%Y-%m-%d}", window_mae)
        maes[label].append(window_mae)
for label in maes:
    show(f"MAE {label}, three windows", np.mean(maes[label]))
    show(f"MAE {label}, 2014-02-15 and 2014-02-28", np.mean(maes[label][1:]))   # [1:] skips Feb 1, the first cutoff
