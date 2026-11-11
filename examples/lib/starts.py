"""The start rule for late starters: a series begins at its first reading. Every chapter that meets one uses it.

    from lib.starts import first_reading, check_start_rule, start_at_first_reading, history_from_start

A late starter is a client whose first reading falls inside the data window. Before its first reading the meter
reports zeros, which the data flags before_first_reading (lib/electricity.py). Those zeros are not low use: a model
fitted on them learns a level near zero. The start rule drops every reading before a client's first non-zero reading.

- first_reading(df) gives each client's first non-zero reading in df. Pass the rows you want the first reading
  from: the whole series for a check, or the rows before a cutoff for a forecast. Raises ValueError for a client
  with no non-zero reading in df.
- check_start_rule(df) raises ValueError unless the rule drops exactly the rows flagged before_first_reading, and
  returns the two counts, (dropped, flagged). Run it on the whole series, before any cut.
- start_at_first_reading(df) returns df without the readings before each client's first reading, after the check.
- history_from_start(df, cutoff, hours=HISTORY_HOURS, min_hours=48) is one cutoff's history under the rule: the
  readings in the `hours` hours before cutoff, from the client's first reading onward. The first reading is looked
  up in the rows before cutoff only, so nothing after the cutoff reaches the model. Raises ValueError if a client
  has no reading before cutoff, or fewer than min_hours hours of history. Its length varies by client and cutoff,
  so it is not history_before, which requires an exact count.

The rule removes only the leading zeros of a series. A window that starts after a client's first reading keeps
every hour in it, including real zeros, because those sit inside the series.
"""
import pandas as pd

from lib.settings import HISTORY_HOURS

FLAG = "before_first_reading"


def first_reading(df):
    """Each client's first non-zero reading in df, as a Series indexed by unique_id."""
    nonzero = df[df["y"].notna() & (df["y"] != 0)]
    firsts = nonzero.groupby("unique_id")["ds"].min()
    missing = sorted(set(df["unique_id"]) - set(firsts.index))
    if missing:
        raise ValueError(f"no non-zero reading for {missing}, so no first reading to start from")
    return firsts


def check_start_rule(df):
    """Raise ValueError unless the start rule drops exactly the rows flagged before_first_reading.

    df needs unique_id, ds, y and quirk, and holds the whole series. Returns (dropped, flagged) row counts.
    """
    if "quirk" not in df.columns:
        raise ValueError("the start rule check needs the quirk column")
    dropped = df["ds"] < df["unique_id"].map(first_reading(df))
    flagged = df["quirk"] == FLAG
    disagree = int((dropped != flagged).sum())
    if disagree:
        raise ValueError(f"the start rule and the flag disagree on {disagree} rows")
    return int(dropped.sum()), int(flagged.sum())


def start_at_first_reading(df):
    """df without the readings before each client's first reading. Runs check_start_rule first."""
    check_start_rule(df)
    keep = df["ds"] >= df["unique_id"].map(first_reading(df))
    return df[keep].reset_index(drop=True)


def history_from_start(df, cutoff, hours=HISTORY_HOURS, min_hours=48):
    """One cutoff's history under the start rule: the readings in the `hours` hours before cutoff, from each
    client's first reading onward. See the module docstring for the leakage rule and the errors raised."""
    cutoff = pd.Timestamp(cutoff)
    before = df[df["ds"] < cutoff]
    firsts = first_reading(before)
    window = before[before["ds"] >= cutoff - pd.Timedelta(hours=hours)]
    hist = window[window["ds"] >= window["unique_id"].map(firsts)]
    counts = hist.groupby("unique_id").size()
    short = counts[counts < min_hours]
    if len(short):
        raise ValueError(f"history before {cutoff} has fewer than {min_hours} hours for {dict(short)}")
    return hist
