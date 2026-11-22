"""Draw the lead client's actual load over the three standard cutoffs, with the forecast from honest lags and from the leaky
set. The forecasts are made as lag-leak-check makes them, and the two MAE lines match the ones that example prints."""
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from lib.electricity import load
from lib.features import LGBM_PARAMS, fit_window_mae, lag_table
from lib.figures import ACCENT, INK, MUTED, new_figure, save
from lib.settings import EVAL_CUTOFFS, HISTORY_HOURS, HORIZON, LEAD, show

y = load([LEAD]).set_index("ds")["y"].asfreq("h")  # asfreq gives every hour a row, so a missing hour is NaN
HONEST, LEAKY = [24, 168], [1, 24, 168]            # the same lag sets as lag-leak-check
CUTOFFS = pd.to_datetime(EVAL_CUTOFFS)


def window_forecasts(lags, leaky):
    """The 24 hours after each cutoff: their actual readings, and the forecast of one LightGBM fitted on that cutoff's
    history. The steps match fit_window_mae; leaky=True reads the target lags from the whole series."""
    times, actual, pred = [], [], []
    for c in CUTOFFS:
        hist = pd.date_range(c - pd.Timedelta(HISTORY_HOURS, unit="h"), c - pd.Timedelta(1, unit="h"), freq="h")
        targets = pd.date_range(c, periods=HORIZON, freq="h")
        train = lag_table(y, lags, hist)
        train["y"] = y.reindex(hist).to_numpy()
        train = train.dropna()                       # a training row whose lag reads a missing hour is left out
        cols = [col for col in train.columns if col != "y"]
        model = LGBMRegressor(**LGBM_PARAMS).fit(train[cols], train["y"])
        source = y if leaky else y[y.index < c]      # the honest model reads the history cut at the cutoff only
        times.append(targets)
        actual.append(y.reindex(targets).to_numpy())
        pred.append(model.predict(lag_table(source, lags, targets)[cols]))
    return pd.DatetimeIndex(np.concatenate(times)), np.concatenate(actual), np.concatenate(pred)


times, actual, honest = window_forecasts(HONEST, leaky=False)
_, _, leaky_pred = window_forecasts(LEAKY, leaky=True)
mae_honest = np.abs(actual - honest).mean()
mae_leaky = np.abs(actual - leaky_pred).mean()
assert np.isclose(mae_honest, fit_window_mae(y, HONEST, CUTOFFS, leaky=False))  # the same numbers lag-leak-check prints
assert np.isclose(mae_leaky, fit_window_mae(y, LEAKY, CUTOFFS, leaky=True))

show("cutoffs", len(CUTOFFS))
show("hours forecast", len(actual))
show("mean actual load, kWh", actual.mean())
show("MAE honest, lags 24 and 168", mae_honest)
show("MAE leaky, lags 1, 24 and 168", mae_leaky)
show("correlation of the honest forecast with the actuals", np.corrcoef(actual, honest)[0, 1], digits=3)
show("correlation of the leaky forecast with the actuals", np.corrcoef(actual, leaky_pred)[0, 1], digits=3)

fig, ax = new_figure(height=2.8)
ax.plot(times, actual, color=INK, linewidth=1.2, label="actual load")
ax.plot(times, honest, color=MUTED, linestyle="--", linewidth=1.2, label="forecast, lags 24 and 168")
ax.plot(times, leaky_pred, color=ACCENT, linewidth=1.6, label="forecast, lags 1, 24 and 168")
ax.set_ylabel("kWh in the hour")
ax.set_ylim(bottom=0)
fig.legend(loc="outside upper left", ncols=3, frameon=False, labelcolor=INK)
ax.xaxis.set_major_locator(mdates.DayLocator())          # one tick at each midnight, so each day is labelled once
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %-d"))
save(fig, "leaky-lag-forecast")
