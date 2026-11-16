"""Choose the refit cadence on the four decision days, Nov 25 to Dec 3, 2014: a refit every day or every seven days,
with the same model. The cadence with the lower `mean MASE` line is the one the rule takes, and the `model aged` lines
show how far a model's age alone moves that score. The held-out week is not used."""
import pandas as pd

from lib.electricity import load
from lib.panel import ALL30
from lib.running import DECISION_DAYS, daily_mase, feed, fit_at, forecast_at, prepare, run_days, scale_from, with_actuals
from lib.settings import show

raw = load(ALL30)
df = prepare(raw)
scale = scale_from(raw)
show("decision days", len(DECISION_DAYS))
show("rows read by one refit", len(feed(df, DECISION_DAYS[0])[0]))  # feed returns (train, known, pending, read_by); [0] is train
names = {1: "daily", 7: "weekly"}  # the keys are the refit interval in days
means = {}
for every in names:
    scored = with_actuals(run_days(df, DECISION_DAYS, refit_every=every), df)
    means[every] = daily_mase(scored, scale).mean()  # the mean over the four decision days of each day's MASE
    show(f"refits, {names[every]}", scored["refit"].nunique())
    show(f"MAE, {names[every]}", scored["ae"].mean())
    show(f"mean MASE, {names[every]}", means[every], digits=3)
show("mean MASE, daily minus weekly", means[1] - means[7], digits=3)
show("chosen cadence", names[min(means, key=means.get)])  # the lower mean wins; the close-call tie-break is made in the prose
models = {}
for age in range(7):
    parts = []
    for day in DECISION_DAYS:
        fit_day = day - pd.Timedelta(days=age)  # a model aged `age` days, fitted that many days before the decision day
        if fit_day not in models:
            models[fit_day] = fit_at(df, fit_day)[0]  # fit_at returns (model, read_by); [0] is the model
        part, _, _ = forecast_at(models[fit_day], df, day)  # forecast_at returns (forecast, pending, read_by); [0] is the forecast
        parts.append(part.assign(cutoff=day))
    show(f"mean MASE, model aged {age} days", daily_mase(with_actuals(pd.concat(parts), df), scale).mean(), digits=3)
