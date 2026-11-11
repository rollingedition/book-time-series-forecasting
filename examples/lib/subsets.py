"""Scores for several clients over the same windows, shared by the backtests of SUBSET8.

    from lib.subsets import score_clients

score_clients(forecasts, actual, models, scale_train, interval_model=None) scores each client over its own scored
hours and then averages over clients, so each client counts once. It returns a dict of labelled values, and the
per-client MASE table (a DataFrame indexed by unique_id, one column per model):
    MAE and RMSE, the mean over clients, for each model
    MASE, the mean over clients, for each model; the scale comes from scale_train (history before the test days)
    coverage 95, for interval_model only: the share of scored hours inside its 95% interval, pooled over clients
Raises ValueError if a forecast hour has no actual reading (for example a masked clock-change hour).
"""
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import mae, rmse

from lib.baselines import mase_scores
from lib.settings import show


def score_clients(forecasts, actual, models, scale_train, interval_model=None):
    models = list(models)
    joined = forecasts.merge(actual[["unique_id", "ds", "y"]], on=["unique_id", "ds"], how="left",
                             validate="one_to_one")
    if joined["y"].isna().any():
        raise ValueError("some forecast hours have no actual reading")
    per_client = evaluate(joined[["unique_id", "ds", "y", *models]], metrics=[mae, rmse], models=models)
    errors = per_client.groupby("metric")[models].mean()
    mase_table = mase_scores(actual, forecasts[["unique_id", "ds", *models]], models, scale_train)
    out = {"clients": int(joined["unique_id"].nunique()),
           "scored hours per client": int(joined.groupby("unique_id").size().min())}
    for m in models:
        out[f"MAE {m}"] = errors.loc["mae", m]
        out[f"RMSE {m}"] = errors.loc["rmse", m]
        out[f"MASE {m}"] = mase_table[m].mean()
    if interval_model:
        inside = (joined["y"] >= joined[f"{interval_model}-lo-95"]) & (joined["y"] <= joined[f"{interval_model}-hi-95"])
        out[f"coverage 95 {interval_model}"] = inside.mean()
    return out, mase_table


def show_client_lines(forms_by_client, mase_table, models):
    """Print one line per client with its chosen form at each cutoff, then its MASE for each model.

    forms_by_client maps unique_id to a list of forms, one per cutoff. mase_table is the per-client MASE table
    that score_clients returns. Uses settings.show, so the lines are the ones the prose binds.
    """
    for uid in sorted(forms_by_client):
        show(f"forms {uid}, one per cutoff", " ".join(forms_by_client[uid]))
        for m in models:
            show(f"MASE {uid} {m}", mase_table.loc[uid, m])
