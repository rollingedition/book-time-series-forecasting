"""Draw the lead client's three scored days: the recorded Chronos-2 median, its 80% and 95% ranges, and the actual load.
The ranges are the recording's own quantiles; the 0.025 and 0.975 quantiles are interpolated by the package, and the
printed counts are for the model and the package together."""
import matplotlib.dates as mdates

from lib.compare import recorded_forecasts, score
from lib.electricity import load
from lib.figures import ACCENT, INK, MUTED, new_figure, save
from lib.intervals import coverage
from lib.panel import ALL30
from lib.settings import LEAD, show

raw = load(ALL30)
# the recording, read only after each output matches its manifest hash; scored against the readings of the lead only
_, scored = score(raw, recorded_forecasts(), ["Chronos"], [LEAD])
lead = scored.sort_values("ds")                                 # the 72 hours of the three standard cutoffs, in time order
lo80, hi80, lo95, hi95 = lead["q10"], lead["q90"], lead["q025"], lead["q975"]

show("hours scored", len(lead), 0)
show("readings inside the 80% range", int(((lead["y"] >= lo80) & (lead["y"] <= hi80)).sum()), 0)
show("readings inside the 95% range", int(((lead["y"] >= lo95) & (lead["y"] <= hi95)).sum()), 0)
show("coverage 80, lead Chronos", coverage(lead["y"], lo80, hi80), 3)
show("coverage 95, lead Chronos", coverage(lead["y"], lo95, hi95), 3)
show("MAE, lead Chronos median, kWh", (lead["y"] - lead["Chronos"]).abs().mean())

fig, ax = new_figure(height=3.2)
ax.fill_between(lead["ds"], lo95, hi95, color=MUTED, alpha=0.15, linewidth=0, label="95% range")   # lighter: the wider band
ax.fill_between(lead["ds"], lo80, hi80, color=MUTED, alpha=0.35, linewidth=0, label="80% range")
ax.plot(lead["ds"], lead["Chronos"], color=ACCENT, linewidth=1.6, label="Chronos-2 median")
ax.plot(lead["ds"], lead["y"], color=INK, linewidth=1.2, label="actual load")
ax.set_ylabel("kWh in the hour")
ax.set_ylim(bottom=0)
ax.xaxis.set_major_locator(mdates.DayLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %-d"))
fig.legend(loc="outside upper left", ncols=2, frameon=False, labelcolor=INK)
save(fig, "chronos-ranges")
