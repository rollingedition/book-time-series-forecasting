"""Draw the lead client's Dec 25, 2014 hour by hour beside the average Thursday and the average Sunday of the eight weeks
before it. Holidays are left out of both averages. The mean absolute differences show which average day Dec 25 is closer to."""
import numpy as np
import pandas as pd

from lib.electricity import load
from lib.features import is_holiday
from lib.figures import ACCENT, INK, MUTED, new_figure, save
from lib.quirks import mask_clock_changes
from lib.settings import HISTORY_HOURS, LEAD, show

DAY = pd.Timestamp("2014-12-25")
raw = mask_clock_changes(load([LEAD]))                          # Chapter 3's clock mask: gap and repeat hours become missing
y = raw.set_index("ds")["y"].asfreq("h")
before = pd.date_range(DAY - pd.Timedelta(HISTORY_HOURS, unit="h"), DAY - pd.Timedelta(1, unit="h"), freq="h")  # eight weeks before Dec 25 00:00
hist = y.reindex(before).to_frame("y")
hist["hour"] = hist.index.hour
hist["weekday"] = hist.index.dayofweek                          # Monday is 0, Thursday 3, Sunday 6
hist["holiday"] = is_holiday(hist.index)                        # 1 on a Portuguese national holiday, else 0
ordinary = hist[hist["holiday"] == 0]                           # holidays are left out of both averages

thursday = ordinary[ordinary["weekday"] == 3].groupby("hour")["y"].mean()  # the average Thursday: mean load at each hour
sunday = ordinary[ordinary["weekday"] == 6].groupby("hour")["y"].mean()    # the average Sunday, the same way
day = y.reindex(pd.date_range(DAY, periods=24, freq="h"))       # Dec 25, hours 00:00 to 23:00
day_values = day.to_numpy()

show("hours in the eight weeks before", len(before))
show("holiday days left out of the averages", int(hist.loc[hist["holiday"] == 1].index.normalize().nunique()))
show("Thursdays averaged", int(ordinary.loc[ordinary["weekday"] == 3].index.normalize().nunique()))
show("Sundays averaged", int(ordinary.loc[ordinary["weekday"] == 6].index.normalize().nunique()))
show("mean load on Dec 25, kWh", day.mean())
show("mean load of the average Thursday, kWh", thursday.mean())
show("mean load of the average Sunday, kWh", sunday.mean())
mad_thursday = np.abs(day_values - thursday.to_numpy()).mean()  # hour by hour, then averaged over the 24 hours
mad_sunday = np.abs(day_values - sunday.to_numpy()).mean()
show("mean absolute difference from the average Thursday, kWh", mad_thursday)
show("mean absolute difference from the average Sunday, kWh", mad_sunday)
assert mad_sunday < mad_thursday  # the prose says Dec 25 is closer to a Sunday than to a Thursday

hours = np.arange(24)
fig, ax = new_figure(height=2.8)
ax.plot(hours, sunday.to_numpy(), color=MUTED, linestyle="--", label="average Sunday")
ax.plot(hours, thursday.to_numpy(), color=INK, label="average Thursday")
ax.plot(hours, day_values, color=ACCENT, linewidth=2.0, label="Dec 25, 2014")  # drawn last, so it sits on top
ax.set_xlabel("hour of the day")
ax.set_ylabel("kWh in the hour")
ax.set_xticks(range(0, 24, 3))
ax.set_ylim(bottom=0)
fig.legend(loc="outside upper left", ncols=3, frameon=False, labelcolor=INK)
save(fig, "holiday-like-sunday")
