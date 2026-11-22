"""Draw the lead client's readings around the March 2014 clock change, and the seasonal naive forecast for the next day with
no clock-change rule: the forecast copies the zero the gap hour reads. The lines print the gap reading, the next-day 1am
forecast with and without the rule, and the actual at that hour."""
import matplotlib.dates as mdates
import pandas as pd

from lib.backtest import history_before
from lib.electricity import load
from lib.figures import ACCENT, INK, MUTED, new_figure, save
from lib.quirks import filled_history, mask_clock_changes, seasonal_naive_at
from lib.settings import LEAD, show

df = load([LEAD])[["unique_id", "ds", "y", "quirk"]]
gap = pd.Timestamp("2014-03-30 01:00")          # the hour that does not occur on the clock
next_1am = pd.Timestamp("2014-03-31 01:00")     # the forecast that copies the gap hour, one day later
cutoff = pd.Timestamp("2014-03-31")             # the cutoff that forecasts the day after the change
assert df.loc[df["ds"] == gap, "quirk"].item() == "clock_change_gap"

gap_reading = df.loc[df["ds"] == gap, "y"].item()
assert gap_reading == 0                         # this client's gap hour reads zero, which the figure is about
show("client", LEAD)
show("gap hour reading, kWh", gap_reading)
show("reading the hour after the gap, kWh", df.loc[df["ds"] == gap + pd.Timedelta(1, unit="h"), "y"].item())

raw = seasonal_naive_at(df, [cutoff], lambda c: history_before(df, c))         # no rule: the zero stays in the history
masked = mask_clock_changes(df)
fixed = seasonal_naive_at(masked, [cutoff], lambda c: filled_history(masked, c))  # the rule: the gap hour filled first
actual = raw.loc[raw["ds"] == next_1am, "y"].item()
show("next-day forecast at 1am, no rule, kWh", raw.loc[raw["ds"] == next_1am, "SeasonalNaive"].item())
show("next-day forecast at 1am, with the rule, kWh", fixed.loc[fixed["ds"] == next_1am, "SeasonalNaive"].item())
show("actual at next-day 1am, kWh", actual)

readings = df[(df["ds"] >= "2014-03-29") & (df["ds"] < "2014-04-01")]   # three days around the change
fig, ax = new_figure(height=2.9)
ax.plot(readings["ds"], readings["y"], color=MUTED, linewidth=1.0, label="readings, as the file has them")
ax.plot(raw["ds"], raw["SeasonalNaive"], color=ACCENT, linewidth=2.0, label="seasonal naive for Mar 31, no rule")
ax.scatter([gap], [gap_reading], color=INK, marker="x", s=36, zorder=3, label="gap hour, Mar 30 01:00")
ax.set_ylabel("kWh in the hour")
ax.set_ylim(bottom=0)
fig.legend(loc="outside upper left", ncols=2, frameon=False, labelcolor=INK)
ax.xaxis.set_major_locator(mdates.DayLocator())          # one tick at each midnight
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %-d"))
save(fig, "march-gap-copied")
