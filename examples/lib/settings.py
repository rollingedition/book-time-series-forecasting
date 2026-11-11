"""Settings every chapter shares: the seed, the horizon, the history length, the clients and the test weeks.

    from lib.settings import LEAD, HORIZON, HISTORY_HOURS, show

show() prints one labelled line, `label: value`. It is how an example prints a number the prose quotes (W-26).
Floats print to 2 decimals (MAE, RMSE, MASE); pass digits=3 for coverage. Integers print as they are.
"""
import numbers

from lib.electricity import CLIENTS

SEED = 0                # np.random.seed before any predict that samples
HORIZON = 24            # hours ahead: a day-ahead forecast is made at 00:00 and covers that day
HISTORY_HOURS = 1344    # 8 weeks of hours before each cutoff, unless a chapter says otherwise
LEAD = "MT_014"         # the lead client, the first client of SUBSET8; no late start
SUBSET8 = CLIENTS[1::4]  # every fourth client of the thirty, starting with the second
HELD_OUT_WEEK = ("2014-12-25", "2014-12-31")  # 7 daily cutoffs at 00:00
EVAL_CUTOFFS = ("2014-12-29", "2014-12-30", "2014-12-31")  # the three standard evaluation cutoffs


def show(label, value, digits=2):
    """Print `label: value` on one line. Strings print as they are; integers print without decimals."""
    if isinstance(value, str):
        print(f"{label}: {value}")
    elif isinstance(value, numbers.Integral):
        print(f"{label}: {int(value)}")
    else:
        print(f"{label}: {value:.{digits}f}")
