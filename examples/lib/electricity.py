"""The book's electricity data, read the same way in every chapter.

    from lib.electricity import load, CLIENTS

load() returns long-format hourly energy: unique_id, ds, y (kWh in that hour), quirk.
ds is local Portuguese clock time with no time zone. quirk is one of:
    normal                  an ordinary hour
    clock_change_gap        01:00 on a March clock-change day; zero or near zero for most clients
    clock_change_repeat     01:00 on an October clock-change day; about two hours of energy in one slot
    before_first_reading    a client's hours before its first reading (a late-starting client)
Values are kept as recorded. See ../data/README.md.
"""
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data" / "electricity_hourly.csv.gz"
CLIENTS = [f"MT_{i:03d}" for i in (1, 14, 26, 39, 52, 64, 77, 90, 103, 115, 128, 141, 153, 166, 179,
                                    191, 204, 217, 230, 243, 256, 268, 281, 294, 307, 319, 332, 345, 357, 370)]


def load(clients=None, start=None, end=None):
    """Hourly kWh for the chosen clients (default: all 30) between start and end, inclusive."""
    df = pd.read_csv(DATA, parse_dates=["ds"])
    if clients is not None:
        df = df[df["unique_id"].isin(clients)]
    if start is not None:
        df = df[df["ds"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["ds"] <= pd.Timestamp(end)]
    return df.reset_index(drop=True)
