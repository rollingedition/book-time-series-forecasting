"""The book's one MASE scale (decisions B-4 and B-5), and the scoring shared by the Chapter 6 examples.

    from lib.scoring import book_scale, clients_left_out, actual_hours, score_windows, report_scores

book_scale(df, first_cutoff) returns the rows a MASE scale may read for a backtest whose first scored cutoff is
first_cutoff. It applies three steps in this order:
1. the start rule (starts.start_at_first_reading): a late starter's readings begin at its first reading;
2. the clock-change mask (quirks.mask_clock_changes): the two clock-change hours are NaN and are not counted;
3. baselines.scale_history: only the rows strictly before first_cutoff.
The result has unique_id, ds and y, sorted, and holds one or many clients. Pass it to baselines.mase_scores as
train_df. A client with no non-zero reading before first_cutoff has no rows in the result, so it is left out of
MASE. clients_left_out(df, first_cutoff) lists those clients, so the caller can report them; it is not an error.

The start rule looks up each first reading in the whole series, but it changes no row before first_cutoff unless a
non-zero reading exists before first_cutoff, which is a fact about the past. A client with none has all its
pre-cutoff rows dropped, so the scale is the same as if the first reading were looked up before the cutoff only.

df needs unique_id, ds, y and quirk, and holds the whole series. Chapter 6 introduces the function; Chapters 3 to 5
use it for the same scale.
"""
import pandas as pd

from lib.baselines import mase_scores, scale_history
from lib.quirks import mask_clock_changes
from lib.settings import HORIZON, show
from lib.starts import start_at_first_reading

SCALE_COLUMNS = ["unique_id", "ds", "y"]


def book_scale(df, first_cutoff):
    """The rows the book's MASE scale reads: the start rule, the clock mask, then the rows before first_cutoff."""
    started = start_at_first_reading(df)
    masked = mask_clock_changes(started)
    before = scale_history(masked, first_cutoff)
    return before[SCALE_COLUMNS].sort_values(["unique_id", "ds"]).reset_index(drop=True)


def clients_left_out(df, first_cutoff):
    """The clients with no non-zero reading before first_cutoff, sorted. Each is left out of MASE, not an error."""
    cutoff = pd.Timestamp(first_cutoff)
    before = df[(df["ds"] < cutoff) & df["y"].notna() & (df["y"] != 0)]
    return sorted(set(df["unique_id"]) - set(before["unique_id"]))


def left_out_facts(raw, df, first_cutoff, check_cutoff="2013-12-15"):
    """The left-out counts of global-model-thirty-clients. The check counts the same clients at an earlier cutoff, where
    two clients have no reading yet, and the broken count would count any row, which gives 0 there."""
    return {
        "clients left out of MASE": len(clients_left_out(raw, first_cutoff)),
        f"clients with no reading before {check_cutoff}, check": len(clients_left_out(raw, check_cutoff)),
        f"clients with any row before {check_cutoff}, broken count": df[df["ds"] < check_cutoff]["unique_id"].nunique(),
    }


def actual_hours(df, cutoffs, h=HORIZON):
    """The readings of the hours the windows forecast: from the first cutoff to the last cutoff's h-th hour."""
    cutoffs = pd.DatetimeIndex(cutoffs)
    return df[(df["ds"] >= cutoffs[0]) & (df["ds"] < cutoffs[-1] + pd.Timedelta(hours=h))]


def score_windows(actual, forecasts, models, scale_rows):
    """The forecasts joined to the readings, and each model's per-client MASE on the scale rows."""
    scored = forecasts.merge(actual[["unique_id", "ds", "y"]], on=["unique_id", "ds"], validate="one_to_one")
    per_client = mase_scores(actual, forecasts, models, scale_rows)
    return scored, per_client


def report_scores(scored, per_client, models):
    """One line each for MAE, then RMSE, then mean MASE, for each model."""
    for model in models:
        show(f"MAE {model}", (scored["y"] - scored[model]).abs().mean())
    for model in models:
        show(f"RMSE {model}", ((scored["y"] - scored[model]) ** 2).mean() ** 0.5)
    for model in models:
        show(f"mean MASE {model}", per_client[model].mean())
