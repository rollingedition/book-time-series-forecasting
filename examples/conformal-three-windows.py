"""mlforecast's own conformal intervals, with three calibration windows, refitted at each of the three standard cutoffs.
The point rests on the coverage at 80 and 95 falling short of their labels, and on the highest coverage three windows allow."""
import pandas as pd

from lib.electricity import load
from lib.intervals import builtin_intervals, design_coverage, symmetric_ceiling
from lib.panel import ALL30
from lib.quirks import mask_clock_changes
from lib.settings import EVAL_CUTOFFS, show

raw = load(ALL30)
# The Oct 26, 2014 clock change is masked as NaN; the windows here start after it, so no masked hour is scored.
df = mask_clock_changes(raw)[["unique_id", "ds", "y"]]
# One row per client and hour. The bounds are columns named by model and level, such as LGBMRegressor-lo-80.
cv = builtin_intervals(df, pd.to_datetime(EVAL_CUTOFFS))

show("scored hours", len(cv))
show("calibration windows per cutoff", 3)
show("residuals per client and hour", 3)
# The band is plus and minus the largest of three absolute residuals, so it covers 3 of the 4 places a new one can take.
show("highest coverage three windows allow, any level", symmetric_ceiling(3), 3)
for level in (80, 95):
    # Simulated under exchangeable normal errors, with mlforecast's construction, to set against the measured coverage.
    show(f"simulated coverage at {level}, exchangeable errors", design_coverage(level, 3, "builtin"), 3)
for level in (80, 95):
    lo, hi = cv[f"LGBMRegressor-lo-{level}"], cv[f"LGBMRegressor-hi-{level}"]
    inside = (cv["y"] >= lo) & (cv["y"] <= hi)
    show(f"coverage at {level}", inside.mean(), 3)
    show(f"mean width at {level}", (hi - lo).mean())
    if level == 80:
        # The by-day lines are printed for the 80 level only. Each hour is grouped by its calendar day.
        day = cv["ds"].dt.floor("D")
        for d, share in inside.groupby(day).mean().items():
            show(f"coverage at 80 on {d:%Y-%m-%d}", share, 3)
        for d, err in (cv["y"] - cv["LGBMRegressor"]).abs().groupby(day).mean().items():
            show(f"mean absolute error on {d:%Y-%m-%d}", err)
