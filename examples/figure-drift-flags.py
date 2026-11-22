"""Draw the daily MASE of the held-out week as recorded, and after the stated test in which every reading from Dec 27 is
multiplied by 0.6, against the flag line at twice the reference. The flags are counted as drift-flag-rolling and
drift-after-a-change count them, and the two counts must match those examples."""
import matplotlib.dates as mdates
import pandas as pd

from lib.electricity import load
from lib.figures import ACCENT, INK, MUTED, new_figure, save
from lib.panel import ALL30
from lib.running import (CAL_DAYS, FLAG_FACTOR, HELD_OUT_DAYS, daily_mase, flagged, prepare, run_days, scale_from,
                         with_actuals)
from lib.settings import show

raw = load(ALL30)
df = prepare(raw)
scale = scale_from(raw)
reference = daily_mase(with_actuals(run_days(df, CAL_DAYS), df), scale).median()    # calibration days, before the change
recorded = daily_mase(with_actuals(run_days(df, HELD_OUT_DAYS), df), scale)

changed = df.copy()
after = changed["ds"] >= pd.Timestamp("2014-12-27")                                 # the stated test, from Dec 27 on
changed.loc[after, "y"] = changed.loc[after, "y"] * 0.6
tested = daily_mase(with_actuals(run_days(changed, HELD_OUT_DAYS), changed), scale)

flags_recorded = flagged(recorded, reference)
flags_tested = flagged(tested, reference)
show("reference, median daily MASE", reference)
show("flag line, MASE", FLAG_FACTOR * reference)                                    # twice the reference
show("held-out days flagged, as recorded", len(flags_recorded))
show("flagged days, as recorded", " and ".join(f"{day:%b %-d}" for day in flags_recorded))
show("held-out days flagged, after the stated test", len(flags_tested))
assert len(flags_recorded) == 2 and len(flags_tested) == 7    # the counts drift-flag-rolling and drift-after-a-change print

fig, ax = new_figure(height=3.2)
ax.plot(recorded.index, recorded.values, color=INK, marker="o", markersize=3, label="as recorded")
ax.plot(tested.index, tested.values, color=ACCENT, marker="o", markersize=3,
        label="after the stated test: readings from Dec 27 at 60%")
ax.axhline(FLAG_FACTOR * reference, color=MUTED, linestyle="--", linewidth=1.2, label="flag line, twice the reference")
ax.set_ylabel("daily MASE")
ax.set_ylim(bottom=0)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %-d"))
fig.legend(loc="outside upper left", ncols=1, frameon=False, labelcolor=INK)
save(fig, "drift-flags")
