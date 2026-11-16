"""The buy level for the total, set from the two costs: the critical fractile, cu / (cu + co), is the quantile that minimises
the expected cost. The point rests on the mean cost and the hours short at each level, and the costs are illustrative, not prices."""
import pandas as pd

from lib.electricity import load
from lib.intervals import CAL_CUTOFFS, COST_EXCESS, COST_SHORT, buy_report, critical_fractile, paired
from lib.panel import ALL30
from lib.quirks import mask_clock_changes
from lib.settings import EVAL_CUTOFFS, show

raw = load(ALL30)
# The Oct 26, 2014 clock change is masked as NaN, as in every example of this chapter.
df = mask_clock_changes(raw)[["unique_id", "ds", "y"]]
# Calibration on Dec 9 to 20, 2014; scored on the three standard cutoffs.
pairs_cal, pairs_sc = paired(df, CAL_CUTOFFS), paired(df, pd.to_datetime(EVAL_CUTOFFS))

show("cost of a kWh short", COST_SHORT)
show("cost of a kWh unused", COST_EXCESS)
# With the illustrative costs, 3 / (3 + 1) = 0.75.
show("critical fractile", critical_fractile())
# buy_report returns the printed rows as (label, value, decimal places).
for label, value, digits in buy_report(pairs_cal, pairs_sc):
    show(label, value, digits)
