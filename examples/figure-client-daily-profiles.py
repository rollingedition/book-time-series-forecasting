"""Draw the average day of the total of the thirty clients, and of its largest client and the client whose day differs most,
each divided by its own daily mean, so the shapes share one scale. The history is the one before the first standard cutoff.
The lines print each peak hour and the mean absolute difference from the total's day, which the prose quotes."""
import pandas as pd

from lib.electricity import load
from lib.figures import ACCENT, INK, MUTED, new_figure, save
from lib.panel import ALL30, panel_history
from lib.quirks import mask_clock_changes
from lib.settings import EVAL_CUTOFFS, show
from lib.totals import shares, total_history

raw = load(ALL30)
cut = pd.Timestamp(EVAL_CUTOFFS[0])                               # the first standard cutoff: shares and profiles read before it


def normalised(hist):
    """Mean load at each hour of the day, divided by the mean of those 24 means: the shape, on one scale."""
    by_hour = hist.groupby(hist["ds"].dt.hour)["y"].mean()
    return by_hour / by_hour.mean()


total_day = normalised(total_history(raw, cut))                   # the total's history has no missing hour
clients = panel_history(mask_clock_changes(raw)[["unique_id", "ds", "y"]], cut, start_rule=True)
days = pd.DataFrame({uid: normalised(g) for uid, g in clients.groupby("unique_id")}).T   # one row per client, 24 hours
largest = shares(raw, cut).idxmax()                               # the largest client by share of the total
gap = (days - total_day).abs().mean(axis=1)                       # mean absolute difference from the total's day
differs = gap.drop(largest).idxmax()                              # the most different shape, other than the largest client's

show("largest client share of the total, first cutoff", shares(raw, cut).max())
show("peak hour of the total's average day", f"{total_day.idxmax():02d}:00")
show("peak hour of the largest client's average day", f"{days.loc[largest].idxmax():02d}:00")
show("mean absolute difference from the total's day, largest client", gap[largest], digits=3)
show("peak hour of the most different client's average day", f"{days.loc[differs].idxmax():02d}:00")
show("mean absolute difference from the total's day, most different client", gap[differs], digits=3)
show("clients whose day differs more than the largest client's", int((gap > gap[largest]).sum()))

fig, ax = new_figure(height=2.8)
ax.plot(total_day.index, total_day.values, color=ACCENT, linewidth=2.0, label="total of the thirty clients")
ax.plot(days.columns, days.loc[largest].values, color=INK, label=f"largest client, {largest}")
ax.plot(days.columns, days.loc[differs].values, color=MUTED, linestyle="--", label=f"most different shape, {differs}")
ax.set_xlabel("hour of the day")
ax.set_ylabel("load over its daily mean")
ax.set_xticks(range(0, 24, 3))
ax.set_ylim(bottom=0)
fig.legend(loc="outside upper left", ncols=1, frameon=False, labelcolor=INK)
save(fig, "client-daily-profiles")
