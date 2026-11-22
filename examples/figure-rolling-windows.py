"""Draw the first four Sunday windows of 2014: each window's fixed 1,344-hour history and its 24 targets, and the expanding
history that cross_validation takes when input_size is not passed. Drawn from the dates alone, with no model fitted.
The lines print the dates and lengths the prose quotes."""
import matplotlib.dates as mdates
import pandas as pd

from lib.backtest import history_before
from lib.electricity import load
from lib.figures import ACCENT, INK, MUTED, new_figure, save
from lib.settings import HISTORY_HOURS, HORIZON, LEAD, show

cutoffs = pd.date_range("2014-01-05", periods=4, freq="W-SUN")  # the first four Sunday cutoffs of the sample
lead = load([LEAD])[["unique_id", "ds", "y"]]
first_reading = lead["ds"].min()                                  # where an expanding history starts

fixed = [history_before(lead, c) for c in cutoffs]               # each window's fixed history; raises unless 1,344 rows
assert all(len(h) == HISTORY_HOURS for h in fixed)
expanding = [int(((lead["ds"] >= first_reading) & (lead["ds"] < c)).sum()) for c in cutoffs]  # hours from the first reading

show("first cutoff", f"{cutoffs[0]:%b %-d, %Y}")
show("last cutoff drawn", f"{cutoffs[-1]:%b %-d, %Y}")
show("step between cutoffs, days", (cutoffs[1] - cutoffs[0]).days)
show("fixed history hours per window", len(fixed[0]))
show("first history timestamp, first window", f"{fixed[0]['ds'].min():%b %-d, %Y %H:%M}")
show("target hours per window", HORIZON)
show("first reading of the series", f"{first_reading:%b %-d, %Y %H:%M}")
show("expanding history hours at the first window", expanding[0])
show("expanding history hours at the last drawn window", expanding[-1])

fig, (top, bottom) = new_figure(rows=2, height=5.4, sharex=False)
rows = [4, 3, 2, 1]                                              # the first window on top
handles, labels = [], []
for y, c in zip(rows, cutoffs):
    # top: the fixed history, 1,344 hours back from the cutoff, and the 24 targets from the cutoff, to scale
    start = c - pd.Timedelta(HISTORY_HOURS, unit="h")
    top.barh(y, width=HISTORY_HOURS / 24, left=mdates.date2num(start), height=0.56, color=ACCENT,
             label="fixed history, 1,344 hours" if y == 4 else None)
    top.barh(y, width=HORIZON / 24, left=mdates.date2num(c), height=0.56, color=INK,
             label="24 targets" if y == 4 else None)
    # bottom: the expanding history, from the first reading up to the cutoff; the cutoff is a tick
    span = (c - first_reading) / pd.Timedelta(1, unit="D")
    bottom.barh(y, width=span, left=mdates.date2num(first_reading), height=0.56, color=MUTED,
                label="expanding history, from the first reading" if y == 4 else None)
    bottom.plot([mdates.date2num(c)] * 2, [y - 0.4, y + 0.4], color=INK, linewidth=2.0,
                label="cutoff" if y == 4 else None)

for ax, lo in ((top, cutoffs[0] - pd.Timedelta(HISTORY_HOURS + 72, unit="h")), (bottom, first_reading - pd.Timedelta(5, unit="D"))):
    ax.set_xlim(mdates.date2num(lo), mdates.date2num(cutoffs[-1] + pd.Timedelta(3, unit="D")))
    ax.set_yticks(rows)
    ax.set_yticklabels([f"{c:%b %-d}" for c in cutoffs])
    ax.set_ylim(0.4, 4.6)
    ax.set_ylabel("cutoff")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %-d, %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1 if ax is top else 3))  # few enough ticks to read
    ax.tick_params(axis="x", labelsize=8)
top.set_title("the same eight weeks before each cutoff", loc="left", color=MUTED)
bottom.set_title("the history that grows, from the first reading", loc="left", color=MUTED)
for ax in (top, bottom):
    h, lab = ax.get_legend_handles_labels()
    handles += h
    labels += lab
fig.legend(handles, labels, loc="outside upper left", ncols=2, frameon=False, labelcolor=INK)
save(fig, "rolling-windows")
