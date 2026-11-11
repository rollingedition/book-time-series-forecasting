"""The total of the thirty clients, forecast two ways, and the client shares that split it (Chapter 7).

    from lib.totals import TOTAL, METHODS, total_frame, total_forecasts, total_scale_rows, pooled_mae, plain_mae
    from lib.totals import total_checks, split_scored, split_checks, client_mae

The total is the hourly sum of ALL30. Its methods:
- bottom-up: the sum of the thirty client forecasts from panel_windows (Chapter 6's model, identifier kept), by hour.
- direct: one model on the total's own history, with the clients' features (lags 24 and 168). Its target is scaled
  only if DIRECT_SCALED is set, which the decision windows decide (the example prints both).
- top-down: the direct total forecast times each client's share, where a share is read from history before the cutoff.
- seasonal naive on the total, from the same total history as the direct model.

The start rule is applied to each client before the sum, so the total is the sum of the clients that have started. A
total hour is NaN if any client's reading in it is NaN (a clock-change hour). The forecasts read the total's history
through panel_history, which looks up each first reading before the cutoff only, so no later reading reaches a model.
"""
import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import SeasonalNaive

from lib.baselines import SEASON
from lib.features import make_forecaster
from lib.panel import ALL30, IDENTIFIER_KEPT, LAGS, panel_history, panel_windows
from lib.quirks import CLOCK_CHANGE_QUIRKS, mask_clock_changes
from lib.scoring import actual_hours, book_scale, score_windows
from lib.settings import HORIZON
from lib.starts import start_at_first_reading

TOTAL = "total"                                  # the unique_id of the total series
METHODS = ["SeasonalNaive", "bottom-up", "direct"]
SPLIT_METHODS = ["bottom-up", "top-down"]
POISONS = (0.0, 1e6)                             # the values a leak check writes over every reading after a cutoff
DIRECT_SCALED = False                            # the direct total's target is unscaled: lower MAE on the decision windows


def total_frame(raw):
    """The total series (unique_id, ds, y, quirk) of the thirty clients, each read from its first reading.

    The start rule is applied to each client first (starts.start_at_first_reading), then the clock-change hours are
    masked, then the readings are summed by hour.
    """
    masked = mask_clock_changes(start_at_first_reading(raw))
    total = masked.groupby("ds")["y"].sum()
    total[masked["y"].isna().groupby(masked["ds"]).any()] = np.nan
    clock = masked[masked["quirk"].isin(CLOCK_CHANGE_QUIRKS)].drop_duplicates("ds").set_index("ds")["quirk"]
    return pd.DataFrame({"unique_id": TOTAL, "ds": total.index, "y": total.to_numpy(),
                         "quirk": clock.reindex(total.index).fillna("normal").to_numpy()})


def total_scale_rows(raw, first_cutoff):
    """The rows the total's MASE scale may read: the same rule as every client's (scoring.book_scale), applied to the
    total. The start rule is a no-op on the total, whose first reading is its first hour."""
    return book_scale(total_frame(raw), first_cutoff)


def total_history(raw, cutoff):
    """The total's history before cutoff: each client's history under the start rule (panel_history), summed by hour.

    The clock-change hours are filled per client from that client's own history, as Chapter 6 does, so the total has
    no missing hour in its history.
    """
    df = mask_clock_changes(raw)[["unique_id", "ds", "y"]]
    hist = panel_history(df, cutoff, start_rule=True)
    out = hist.groupby("ds", as_index=False)["y"].sum()
    out.insert(0, "unique_id", TOTAL)
    return out


def naive_windows(raw, cutoffs):
    """Seasonal naive on the total, from each cutoff's total history. Returns unique_id, ds and SeasonalNaive."""
    sf = StatsForecast(models=[SeasonalNaive(season_length=SEASON)], freq="h", n_jobs=1)
    parts = []
    for c in pd.to_datetime(cutoffs):
        part = sf.forecast(df=total_history(raw, c), h=HORIZON)
        if part["ds"].min() != c:
            raise ValueError(f"seasonal naive for cutoff {c} does not start at the cutoff")
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def direct_windows(raw, cutoffs, scaled=DIRECT_SCALED):
    """One model on the total's own history per cutoff: lags 24 and 168, target scaled only if `scaled`. Returns direct."""
    parts = []
    for c in pd.to_datetime(cutoffs):
        fcst = make_forecaster(LAGS, (), scaled=scaled)
        fcst.fit(total_history(raw, c), static_features=[])
        part = fcst.predict(h=HORIZON)
        if part["ds"].min() != c:
            raise ValueError(f"direct forecast for cutoff {c} does not start at the cutoff")
        parts.append(part)
    return pd.concat(parts, ignore_index=True).rename(columns={"LGBMRegressor": "direct"})


def bottom_up_windows(raw, cutoffs, client_id=IDENTIFIER_KEPT):
    """The bottom-up total: the sum of the thirty client forecasts from panel_windows, hour by hour. Returns bottom-up."""
    df = mask_clock_changes(raw)[["unique_id", "ds", "y"]]
    fc = panel_windows(df, cutoffs, client_id=client_id, name="client")
    if (fc.groupby("ds").size() != len(ALL30)).any():
        raise ValueError("a total hour has forecasts from fewer than thirty clients")
    out = fc.groupby("ds", as_index=False)["client"].sum().rename(columns={"client": "bottom-up"})
    out.insert(0, "unique_id", TOTAL)
    return out


def total_forecasts(raw, cutoffs, client_id=IDENTIFIER_KEPT):
    """Seasonal naive, bottom-up and direct on the same windows, joined on unique_id and ds."""
    keys = ["unique_id", "ds"]
    return (naive_windows(raw, cutoffs)
            .merge(bottom_up_windows(raw, cutoffs, client_id), on=keys, validate="one_to_one")
            .merge(direct_windows(raw, cutoffs), on=keys, validate="one_to_one"))


def pooled_mae(actual, forecasts, models):
    """The MAE over every scored hour, pooled over the series, for each model. Returns a Series indexed by model."""
    joined = forecasts.merge(actual[["unique_id", "ds", "y"]], on=["unique_id", "ds"], validate="one_to_one")
    if joined[["y", *models]].isna().any().any():
        raise ValueError("a scored hour has no reading or no forecast; a masked hour is never scored as zero")
    return pd.Series({m: float((joined["y"] - joined[m]).abs().mean()) for m in models})


def plain_mae(raw, cutoffs):
    """The pooled MAE on the total of bottom-up without the client identifier: the model Chapter 6 compares with."""
    actual = actual_hours(total_frame(raw), cutoffs)
    plain = bottom_up_windows(raw, cutoffs, client_id=False).rename(columns={"bottom-up": "plain"})
    return pooled_mae(actual, plain, ["plain"])["plain"]


def total_checks(raw, first_cutoff, cutoffs=None):
    """Two counts. The total equals the sum of the readings in every hour without a clock-change flag. For each cutoff
    (default: first_cutoff alone), the total's scale for that window does not move when every reading from that cutoff
    on is replaced. A scale that read its own window would move."""
    cutoffs = [first_cutoff] if cutoffs is None else cutoffs
    clock_hours = raw.loc[raw["quirk"].isin(CLOCK_CHANGE_QUIRKS), "ds"].unique()
    parts = raw[~raw["ds"].isin(clock_hours)].groupby("ds")["y"].sum()
    whole = total_frame(raw).set_index("ds")["y"].reindex(parts.index)
    moved = sum(not book_scale(total_frame(poisoned(raw, c, v)), c).equals(book_scale(total_frame(raw), c))
                for c in pd.to_datetime(cutoffs) for v in POISONS)
    return {"clock-free hours where the total differs from the sum of the thirty clients":
            int((~np.isclose(whole, parts)).sum()),
            "scale changes after poisoning readings from each scored cutoff": int(moved)}


def shares(raw, cutoff):
    """Each client's share of the total over the history before cutoff: its readings summed over its own history,
    divided by the sum of all clients' readings over the same history. The shares add to 1."""
    df = mask_clock_changes(raw)[["unique_id", "ds", "y"]]
    per_client = panel_history(df, cutoff, start_rule=True).groupby("unique_id")["y"].sum()
    return per_client / per_client.sum()


def top_down_windows(raw, cutoffs, direct):
    """Each client's top-down forecast: the direct total in each hour times the client's share from the history before
    that window's cutoff. Returns unique_id, ds and top-down. `direct` holds the total's forecasts (direct_windows)."""
    parts = []
    for c in pd.to_datetime(cutoffs):
        win = direct[(direct["ds"] >= c) & (direct["ds"] < c + pd.Timedelta(hours=HORIZON))]
        share = shares(raw, c).rename("share").reset_index()
        part = win[["ds", "direct"]].merge(share, how="cross")
        part["top-down"] = part["direct"] * part["share"]
        parts.append(part[["unique_id", "ds", "top-down"]])
    return pd.concat(parts, ignore_index=True)


def split_scored(raw, cutoffs):
    """Each client's bottom-up and top-down forecasts, scored against the readings on the book's MASE scale.

    Returns (scored, per_client, gap): scored holds the forecasts with the readings; per_client is each client's MASE for
    each method; gap counts the hours where the top-down clients do not add up to the direct total.
    """
    direct = direct_windows(raw, cutoffs)
    top = top_down_windows(raw, cutoffs, direct)
    masked = mask_clock_changes(raw)[["unique_id", "ds", "y"]]
    bottom = panel_windows(masked, cutoffs, client_id=IDENTIFIER_KEPT, name="bottom-up")
    gap = ~np.isclose(top.groupby("ds")["top-down"].sum(), direct.set_index("ds")["direct"])
    fc = bottom.merge(top, on=["unique_id", "ds"], validate="one_to_one")
    scored, per_client = score_windows(actual_hours(masked, cutoffs), fc, SPLIT_METHODS,
                                       book_scale(raw, pd.to_datetime(cutoffs)[0]))
    return scored, per_client, int(gap.sum())


def split_checks(raw, cutoffs):
    """The count of comparisons in which a share moves when every reading from its own cutoff on is replaced: a share
    read from the past alone must not move. The shares add to 1, and the top-down clients add to the direct total, by
    construction, so neither is counted: an identity like that cannot fail."""
    moved = sum(not shares(poisoned(raw, c, v), c).equals(shares(raw, c))
                for c in pd.to_datetime(cutoffs) for v in POISONS)
    return {"shares changed after poisoning readings from their cutoff": int(moved)}


def client_mae(scored, model):
    """Each client's MAE over its scored hours, from split_scored's scored frame. Returns a Series indexed by unique_id."""
    return (scored["y"] - scored[model]).abs().groupby(scored["unique_id"]).mean()


def poisoned(raw, cutoff, value):
    """raw with every reading from cutoff on replaced by value. A leak check: a method that reads only the past gives
    the same answer on the poisoned frame and on raw."""
    return raw.assign(y=raw["y"].where(raw["ds"] < pd.Timestamp(cutoff), value))


def oracle_split_mae(raw, cutoffs):
    """The mean per-client MAE of the top-down split given the true total: each client's share times the total's actual
    reading, in the same hours. It separates the split's error from the total's forecast error."""
    total = total_frame(raw).set_index("ds")["y"]
    act = actual_hours(mask_clock_changes(raw)[["unique_id", "ds", "y"]], cutoffs)
    parts = []
    for c in pd.to_datetime(cutoffs):
        win = act[(act["ds"] >= c) & (act["ds"] < c + pd.Timedelta(hours=HORIZON))].copy()
        win["split"] = win["ds"].map(total) * win["unique_id"].map(shares(raw, c))
        parts.append(win)
    both = pd.concat(parts)
    return float((both["y"] - both["split"]).abs().groupby(both["unique_id"]).mean().mean())
