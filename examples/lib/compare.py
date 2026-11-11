"""Chapter 9's shared code: Prophet, the recorded neural forecasts (Chronos-2 and NHITS), and one scoring path.

    from lib.compare import (PROPHET_LEVEL, prophet_windows, seasonal_naive, global_forecasts, score,
                             recorded_forecasts, check_recording, report_range, report_pinball, report_subset)

prophet_windows(df, cutoffs) fits Prophet once per client and cutoff. Each fit reads that client's history before the
cutoff (history_from_start, so the start rule holds), leaves out the missing hours (Prophet drops them from the fit),
and uses the Portuguese national holidays from features.PT_HOLIDAYS when use_holidays is true. PROPHET_HOLIDAYS is the
setting the book's runs use; prophet-holiday-decision chose it on the holiday decision windows (B-9). Its interval is PROPHET_LEVEL
percent (95), not Prophet's default of 80. Prophet draws its interval samples from numpy's global generator, so SEED
is set before each predict. Returns unique_id, ds, Prophet, ProphetLo and ProphetHi.

The recordings are read, never made. recorded_forecasts(name, model) checks every output's SHA-256 against the
recording's manifest.json and refuses to return the forecasts if one does not match. It returns the recorded quantiles
as q025, q10, q50, q90 and q975, and the median (q50) under the model's name. A quantile the package interpolated is
listed in the manifest's "interpolated" levels; a quantile trained directly is not.

score(raw, forecasts, models, clients, cutoffs) is the one scoring path. It returns the per-client MASE on the book's
scale (scoring.book_scale over the first cutoff) and the scored hours joined to their readings. The cutoffs default to
the standard ones; prophet-holiday-decision passes the holiday decision windows. The report_ helpers print labelled lines
through lib.settings.show.
"""
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from prophet import Prophet
from statsforecast import StatsForecast
from statsforecast.models import SeasonalNaive

from lib.backtest import forecast_windows
from lib.baselines import SEASON, mase_scores
from lib.features import PT_HOLIDAYS
from lib.intervals import coverage, interval_score, quantile_loss
from lib.panel import IDENTIFIER_KEPT, panel_windows
from lib.scoring import actual_hours, book_scale
from lib.settings import EVAL_CUTOFFS, HORIZON, SEED, SUBSET8, show
from lib.starts import history_from_start

PROPHET_LEVEL = 95   # percent: the interval Prophet is set to, the level of the recordings' 0.025 and 0.975 quantiles
PROPHET_HOLIDAYS = False  # prophet-holiday-decision: on the holiday decision windows the table raised mean MASE, so it is off
PROPHET_OPTIONS = dict(interval_width=PROPHET_LEVEL / 100, daily_seasonality=True, weekly_seasonality=True,
                       yearly_seasonality=False)
RECORDED = Path(__file__).resolve().parent.parent / "data" / "recorded"
QUANTILES = {"q0.025": "q025", "q0.1": "q10", "q0.5": "q50", "q0.9": "q90", "q0.975": "q975"}


def holiday_table():
    """The Portuguese national holidays for the data's years, in the table Prophet takes."""
    return pd.DataFrame({"holiday": "pt_national", "ds": pd.to_datetime(sorted(PT_HOLIDAYS.keys()))})


def _fit_predict(task):
    """One client's fit and predict for one cutoff. Top level, so that a worker process can run it."""
    uid, series, future, holidays = task
    model = Prophet(holidays=holidays, **PROPHET_OPTIONS)
    model.fit(series[["ds", "y"]])
    np.random.seed(SEED)
    pred = model.predict(future)
    return pd.DataFrame({"unique_id": uid, "ds": future["ds"], "Prophet": pred["yhat"].to_numpy(),
                         "ProphetLo": pred["yhat_lower"].to_numpy(), "ProphetHi": pred["yhat_upper"].to_numpy()})


def prophet_windows(df, cutoffs, use_holidays=PROPHET_HOLIDAYS, h=HORIZON, workers=1):
    """Prophet's forecast of the next h hours for every client, one fit per client and cutoff. df holds unique_id, ds, y.
    With workers above 1 the fits run in that many processes; each fit is the same either way."""
    holidays = holiday_table() if use_holidays else None
    tasks = []
    for cutoff in pd.DatetimeIndex(cutoffs):
        hist = history_from_start(df, cutoff)
        future = pd.DataFrame({"ds": pd.date_range(cutoff, periods=h, freq="h")})
        tasks += [(uid, series, future, holidays) for uid, series in hist.groupby("unique_id")]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            parts = list(pool.map(_fit_predict, tasks))
    else:
        parts = [_fit_predict(task) for task in tasks]
    return pd.concat(parts, ignore_index=True).sort_values(["unique_id", "ds"]).reset_index(drop=True)


def seasonal_naive(df, cutoffs):
    """Seasonal naive from statsforecast, one forecast per client and hour, each window fitted on its own history."""
    sf = StatsForecast(models=[SeasonalNaive(season_length=SEASON)], freq="h", n_jobs=1)
    return forecast_windows(sf, df, pd.DatetimeIndex(cutoffs))[["unique_id", "ds", "SeasonalNaive"]]


def global_forecasts(df, cutoffs):
    """Chapter 6's global model: one LightGBM over the clients, the client identifier kept, the start rule applied."""
    out = panel_windows(df, pd.DatetimeIndex(cutoffs), client_id=IDENTIFIER_KEPT, name="Global")
    return out[["unique_id", "ds", "Global"]]


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(name):
    return json.loads((RECORDED / name / "manifest.json").read_text())


def _read_forecasts(name):
    return pd.read_csv(RECORDED / name / "forecasts.csv.gz", parse_dates=["cutoff", "ds"])


def check_recording(name="chronos-2"):
    """Each output's SHA-256 against the recording's manifest.json, and the shape of its forecast file. Returns labelled
    counts. A recording with no interpolated levels reports 0."""
    manifest = _manifest(name)
    mismatches = [out for out, digest in manifest["outputs"].items() if _digest(RECORDED / name / out) != digest]
    f = _read_forecasts(name)
    q = f[["q0.025", "q0.1", "q0.5", "q0.9", "q0.975"]].to_numpy()
    return {
        "outputs checked": len(manifest["outputs"]),
        "hash mismatches": len(mismatches),
        "rows": len(f),
        "clients": f["unique_id"].nunique(),
        "windows": f["cutoff"].nunique(),
        "interpolated levels": len(manifest["quantiles"].get("interpolated", [])),
        "rows with quantiles out of order": int((np.diff(q, axis=1) < 0).any(axis=1).sum()),
        "largest quantile crossing, kWh": float(max(0.0, -np.diff(q, axis=1).min())),
        "inverted 80 or 95 ranges": int(((f["q0.1"] > f["q0.9"]) | (f["q0.025"] > f["q0.975"])).sum()),
    }


def recorded_forecasts(name="chronos-2", model="Chronos"):
    """The forecasts of recording `name` on the standard cutoffs, read only after every output matches its manifest hash.

    Returns unique_id, ds, the five quantile columns and `model`, the median. The cutoff column is dropped, so that
    the MASE scale is not filtered per cutoff (see baselines.mase_scores).
    """
    for out, digest in _manifest(name)["outputs"].items():
        if _digest(RECORDED / name / out) != digest:
            raise ValueError(f"{out} does not match its hash in the {name} manifest.json")
    f = _read_forecasts(name)
    if set(f["cutoff"]) != set(pd.DatetimeIndex(EVAL_CUTOFFS)):
        raise ValueError("the recorded cutoffs are not the standard cutoffs")
    f = f.rename(columns=QUANTILES).drop(columns="cutoff")
    f[model] = f["q50"]
    return f


def score(raw, forecasts, models, clients, cutoffs=EVAL_CUTOFFS):
    """Per-client MASE on the book's scale, and the scored hours joined to their readings, on the given cutoffs.

    raw holds unique_id, ds, y and quirk for every client. forecasts holds unique_id, ds and the model columns, and any
    range columns. Returns (per_client, scored): per_client is indexed by unique_id, one column per model; scored has
    one row per client and hour, with the forecast columns and y.
    """
    cutoffs = pd.DatetimeIndex(cutoffs)
    actual = actual_hours(raw, cutoffs)
    actual = actual[actual["unique_id"].isin(clients)][["unique_id", "ds", "y"]]
    fc = forecasts[forecasts["unique_id"].isin(clients)]
    scale = book_scale(raw, cutoffs[0])
    scale = scale[scale["unique_id"].isin(clients)]
    per_client = mase_scores(actual, fc[["unique_id", "ds", *models]], models, scale)
    scored = fc.merge(actual, on=["unique_id", "ds"], validate="one_to_one")
    return per_client, scored


def report_range(scored, name, lo, hi, level):
    """Coverage and interval score of the range from column lo to column hi, a central `level` percent range."""
    show(f"coverage {level} {name}", coverage(scored["y"], scored[lo], scored[hi]), 3)
    show(f"interval score {level} {name}", interval_score(scored["y"], scored[lo], scored[hi], level))


def report_pinball(scored, name, column, tau):
    """The quantile loss at level tau of one quantile column."""
    show(f"quantile loss {tau} {name}", quantile_loss(scored["y"], scored[column], tau))


def report_subset(per_client, models):
    """The mean per-client MASE of each model over SUBSET8, one labelled line each."""
    for model in models:
        show(f"mean MASE {model} SUBSET8", per_client.loc[SUBSET8, model].mean())
