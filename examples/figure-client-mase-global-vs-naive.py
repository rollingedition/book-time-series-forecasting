"""Draw each client's MASE under seasonal naive against the global model's, on the three standard cutoffs, with the equal line.
Clients above the line are the ones seasonal naive wins; they are labelled. The forecasts and scale are global-model-thirty-clients's."""
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import SeasonalNaive

from lib.backtest import forecast_windows
from lib.baselines import SEASON
from lib.electricity import load
from lib.figures import ACCENT, INK, MUTED, new_figure, save
from lib.panel import ALL30, IDENTIFIER_KEPT, panel_windows
from lib.quirks import mask_clock_changes
from lib.scoring import actual_hours, book_scale, score_windows
from lib.settings import EVAL_CUTOFFS, show

raw = load(ALL30)
df = mask_clock_changes(raw)[["unique_id", "ds", "y"]]        # clock-change hours masked, as in global-model-thirty-clients
cutoffs = pd.to_datetime(EVAL_CUTOFFS)
actual = actual_hours(df, cutoffs)
naive = forecast_windows(StatsForecast(models=[SeasonalNaive(season_length=SEASON)], freq="h", n_jobs=1), df, cutoffs)
global_fc = panel_windows(df, cutoffs, client_id=IDENTIFIER_KEPT)
forecasts = naive.merge(global_fc, on=["unique_id", "ds"], validate="one_to_one")
models = ["SeasonalNaive", "global"]
_, per_client = score_windows(actual, forecasts, models, book_scale(raw, cutoffs[0]))   # one MASE per client over all 72 hours
wins = per_client["global"] < per_client["SeasonalNaive"]     # True where the global model's MASE is lower

show("clients scored", len(per_client))
show("clients where global beats seasonal naive", int(wins.sum()))
show("clients where seasonal naive wins", ", ".join(sorted(per_client.index[~wins])))

lo = float(per_client.min().min()) * 0.9                       # the equal line runs across the whole plot
hi = float(per_client.max().max()) * 1.05
fig, ax = new_figure(height=3.4)
ax.plot([lo, hi], [lo, hi], color=INK, linestyle="--", linewidth=1.0, label="equal MASE")
ax.scatter(per_client.loc[wins, "SeasonalNaive"], per_client.loc[wins, "global"], s=18, color=MUTED,
           label="global model lower")
ax.scatter(per_client.loc[~wins, "SeasonalNaive"], per_client.loc[~wins, "global"], s=22, color=ACCENT,
           label="seasonal naive lower")
for uid in per_client.index[~wins]:                            # name each client where seasonal naive wins
    point = (per_client.at[uid, "SeasonalNaive"], per_client.at[uid, "global"])
    if uid == "MT_294":                                        # its label would sit on other points, so it goes to open space with a leader
        ax.annotate(uid, point, xytext=(1.5, 1.3), textcoords="data", fontsize=8, color=ACCENT,
                    arrowprops=dict(arrowstyle="-", color=ACCENT, linewidth=0.6))
    else:
        ax.annotate(uid, point, xytext=(4, -3), textcoords="offset points", fontsize=8, color=ACCENT)
ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)
ax.set_xlabel("MASE under seasonal naive")
ax.set_ylabel("MASE under the global model")
fig.legend(loc="outside upper left", ncols=3, frameon=False, labelcolor=INK)
save(fig, "client-mase-global-vs-naive")
