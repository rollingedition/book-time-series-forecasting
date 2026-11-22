"""Draw MT_115's eight weeks before the Feb 1, 2014 cutoff: zeros until its first reading, then readings, and the start rule
keeps only the rows from that reading on. The prints give the zero share the prose quotes, the first reading and the rows kept."""
import matplotlib.dates as mdates
import pandas as pd

from lib.backtest import history_before
from lib.electricity import load
from lib.figures import ACCENT, INK, MUTED, new_figure, save
from lib.settings import show
from lib.starts import first_reading, history_from_start

CUTOFF = pd.Timestamp("2014-02-01")                    # the first February cutoff, where MT_115's zeros are most of its history
raw = load(["MT_115"])
hist = history_before(raw, CUTOFF)                     # the 1,344 hours before the cutoff, every row, zeros included
kept = history_from_start(raw, CUTOFF)                 # the start rule: from the first reading on, read before the cutoff only
first = first_reading(hist)["MT_115"]                  # the first non-zero reading in those rows
dropped = hist[hist["ds"] < first]                     # the rows before it, which the rule removes

show("rows in the history before 2014-02-01", len(hist))
show("zero share of history MT_115, 2014-02-01", int(round(100 * (hist["y"] == 0).mean())))
show("first reading MT_115", f"{first:%Y-%m-%d}")
show("rows before the first reading, all zero", len(dropped))
assert (dropped["y"] == 0).all()                       # the rule removes zeros only; a reading there would fail here
show("rows the start rule keeps", len(kept))
show("zeros the start rule keeps", int((kept["y"] == 0).sum()))

fig, ax = new_figure(height=2.6)
ax.plot(hist["ds"], hist["y"], color=MUTED, linewidth=0.8, label="every hour before the cutoff")
ax.plot(kept["ds"], kept["y"], color=ACCENT, linewidth=1.2, label="kept by the start rule")
ax.axvline(first, color=INK, linestyle="--", linewidth=1.0, label="first reading")
ax.set_ylabel("kWh in the hour")
ax.set_ylim(-4, 64)                                # room below 0, so the zeros draw above the axis line
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %-d"))
fig.legend(loc="outside upper left", ncols=3, frameon=False, labelcolor=INK, handlelength=2.5)
save(fig, "late-starter-history")
