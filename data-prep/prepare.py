"""Cut the UCI ElectricityLoadDiagrams20112014 file into the book's hourly subset.

Usage, from the book root, with the examples runtime (pandas and numpy only):

    python -I data-prep/prepare.py RAW_TXT examples/data/electricity_hourly.csv.gz

RAW_TXT is LD2011_2014.txt from the UCI zip (https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip).
The script refuses any file whose SHA-256 is not RAW_SHA256. Output bytes are deterministic (gzip mtime 0).

What it does, in order:
  1. Reads the 15-minute readings (kW, decimal comma, ';' separated) for the window below.
  2. Drops clients with no non-zero reading in the window.
  3. Picks N_CLIENTS of the rest: sort by numeric id, take evenly spaced positions (np.linspace, rounded).
  4. Makes hourly energy: each hour's four readings are averaged. A reading in kW held for 15 minutes is
     kW x 0.25 kWh, so the four together give kWh for the hour, which equals their mean kW.
     A reading's label is taken as the start of its interval (the labels run 00:00 to 23:45 on each day).
  5. Flags quirks in column `quirk` and keeps the values as recorded (no fixing):
       before_first_reading  hours before the client's first non-zero hour (late-starting clients)
       clock_change_gap      01:00 on the March clock-change day (zero for most clients)
       clock_change_repeat   01:00 on the October clock-change day (about two hours of energy)
     A flagged hour that matches more than one rule takes the first rule in that order.
  6. Rounds y to DECIMALS places (kWh), writes long format: unique_id, ds, y, quirk.
"""
import argparse
import gzip
import hashlib
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd

RAW_SHA256 = "d51565f2cb5a6b768d06ba1bbd3c084c6e2f3aab07f00c6f2dcb80e90175124b"
WINDOW_START = "2013-01-01"   # first hour kept (inclusive)
WINDOW_END = "2014-12-31"     # last day kept (inclusive, whole day)
N_CLIENTS = 30
DECIMALS = 3                  # kWh to the watt-hour


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def last_sunday(year, month):
    """The last Sunday of a month: the EU clock-change day for March and October (Portugal included)."""
    d = date(year + (month == 12), month % 12 + 1, 1) - timedelta(days=1)
    while d.weekday() != 6:
        d -= timedelta(days=1)
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("raw", help="path to LD2011_2014.txt")
    ap.add_argument("out", help="output path, e.g. examples/data/electricity_hourly.csv.gz")
    args = ap.parse_args()

    digest = sha256_file(args.raw)
    print("raw sha256", digest)
    if digest != RAW_SHA256:
        sys.exit("raw file is not the UCI file this script was written for")

    raw = pd.read_csv(args.raw, sep=";", decimal=",", index_col=0, low_memory=False)
    raw.index = pd.to_datetime(raw.index, format="%Y-%m-%d %H:%M:%S")
    start = pd.Timestamp(WINDOW_START)
    end = pd.Timestamp(WINDOW_END) + pd.Timedelta(days=1)
    win = raw[(raw.index >= start) & (raw.index < end)]
    days_in_window = (end - start).days
    if len(win) != days_in_window * 96:
        sys.exit(f"unexpected row count in window: {len(win)}")

    # 2. drop all-zero clients, 3. choose evenly spaced clients in id order
    dropped = sorted(c for c in win.columns if not (win[c] != 0).any())
    eligible = sorted((c for c in win.columns if c not in dropped), key=lambda c: int(c.split("_")[1]))
    pos = np.round(np.linspace(0, len(eligible) - 1, N_CLIENTS)).astype(int)
    chosen = [eligible[i] for i in pos]

    # 4. hourly kWh: mean of the four 15-minute kW readings
    kw = win[chosen]
    groups = kw.groupby(kw.index.floor("h"))
    if not (groups.size() == 4).all():
        sys.exit("an hour does not have four readings")
    hourly = groups.mean()
    hourly.index.name = "ds"
    if len(hourly) != days_in_window * 24:
        sys.exit(f"unexpected hour count: {len(hourly)}")

    # 5. quirks
    years = sorted({ts.year for ts in hourly.index})
    gap_days = {last_sunday(y, 3) for y in years}
    repeat_days = {last_sunday(y, 10) for y in years}
    long = pd.concat(
        [pd.DataFrame({"unique_id": c, "ds": hourly.index, "y": hourly[c].to_numpy()}) for c in chosen],
        ignore_index=True,
    )
    day = long["ds"].dt.normalize().dt.date
    at_one = long["ds"].dt.hour.eq(1)
    is_gap = at_one & day.isin(gap_days)
    is_repeat = at_one & day.isin(repeat_days)
    started = long.groupby("unique_id")["y"].transform(lambda s: (s > 0).cummax())
    before = ~started
    long["quirk"] = np.select(
        [before, is_gap, is_repeat],
        ["before_first_reading", "clock_change_gap", "clock_change_repeat"],
        default="normal",
    )

    # Diagnostics for the card: what the flagged hours look like against their neighbours.
    neighbours = (hourly.shift(1) + hourly.shift(-1)) / 2
    for label, days in (("gap", gap_days), ("repeat", repeat_days)):
        mask = pd.Series(hourly.index.hour == 1, index=hourly.index) & pd.Series(
            [ts.date() in days for ts in hourly.index], index=hourly.index
        )
        ratio = (hourly[mask] / neighbours[mask].where(neighbours[mask] > 0)).stack().dropna()
        print(f"{label} hour (01:00) on {int(mask.sum())} day(s): {ratio.size} client-hours with non-zero neighbours; "
              f"median ratio to neighbours {ratio.median():.3f}, share below 0.05 {(ratio < 0.05).mean():.3f}")

    # 6. output. Rounding is explicit (half up, by floor) so ties such as 155.3125 never depend on the pandas version.
    scale = 10 ** DECIMALS
    long["y"] = np.floor(long["y"] * scale + 0.5) / scale
    if (long["y"] < 0).any() or long["y"].isna().any():
        sys.exit("negative or missing values after cutting")
    text = long.assign(ds=long["ds"].dt.strftime("%Y-%m-%d %H:%M:%S")).to_csv(index=False, lineterminator="\n")
    data = gzip.compress(text.encode("utf-8"), compresslevel=9, mtime=0)
    with open(args.out, "wb") as f:
        f.write(data)

    late = [c for c in chosen if (hourly[c] > 0).idxmax() > start]
    print("window", WINDOW_START, "to", WINDOW_END, "hours", len(hourly), "clients in window", win.shape[1])
    print("dropped all-zero", len(dropped), dropped)
    print("chosen", len(chosen), ", ".join(chosen))
    print("late starters chosen", len(late), {c: str((hourly[c] > 0).idxmax()) for c in late})
    print("rows", len(long), "quirk counts", long["quirk"].value_counts().to_dict())
    print("clock_change_gap days", sorted(str(d) for d in gap_days), "clock_change_repeat days", sorted(str(d) for d in repeat_days))
    print("y kWh per hour (min, median, max)", float(long["y"].min()), float(long["y"].median()), float(long["y"].max()))
    print("output", args.out, "bytes", len(data), "sha256", hashlib.sha256(data).hexdigest())


if __name__ == "__main__":
    main()
