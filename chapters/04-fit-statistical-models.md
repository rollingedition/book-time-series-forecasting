---
chapter_id: fit-statistical-models-per-client
---

# Fit a statistical model for each client

By the end of this chapter you can fit AutoETS and a fixed seasonal ARIMA for each client over the same backtest, score both against seasonal naive, and start a late-starting client at its first reading. The chapter assumes the baselines and backtests of the two chapters before it.

The supplier needs a day-ahead forecast for each of its clients. Chapter 2's seasonal naive gives every client the same rule: the same hour yesterday. That rule holds no estimate of a client's level or its trend, and the clients differ in both. This chapter fits a statistical model for each client, and asks whether the extra work pays.

## Why a statistical model comes next {#why-a-statistical-model-comes-next}

[Chapter 2](#beat-the-naive-baselines) set the bar. Seasonal naive forecasts each hour as the same hour yesterday, and the scaled error, MASE, compares clients with different loads on one scale. [Chapter 3](#backtest-without-looking-ahead) backtested that forecast over many cutoffs. A cutoff is the time a forecast is made, and its history ends there. A window is one cutoff and the 24 hours after it, which is the horizon. Each window's history ends at its cutoff, so no hour is forecast with data from its own day.

A statistical model adds to that loop. A series has a state. Its level is the typical reading, its trend is the slow drift of that level, its season is the daily shape of the load, and what none of these explain is error. A statistical model holds an estimate of each part, updates the estimates as each reading arrives, and projects them over the horizon. Seasonal naive has no such state. It copies yesterday, so it gets a client's level only as well as yesterday did.

A per-client model is one fit for each client, on that client's history alone. The cost is one fit per client per window. The subset this chapter uses, SUBSET8, is eight clients: every fourth of the thirty, starting with the second. Over the three standard cutoffs, Dec 29, 2014 to Dec 31, 2014, that is 24 fits for each model. The examples run the fits in one worker (`n_jobs=1`), which keeps the output the same from run to run. Every model in this chapter gets the same three cutoffs and the same 1,344 hours (eight weeks) of history before each one. The cutoffs fall on the last days of 2014, the test period the whole book uses, and the subset holds the one late starter that the later section needs.

**Working rule:** Fit one model per client only after a baseline and a backtest are in place. Count the fits before you choose the number of windows: clients, times windows, times refits if you run the model every day. Use the same cutoffs and history for every model in a comparison. Otherwise the comparison measures the windows, not the models.

## Fit exponential smoothing {#fit-exponential-smoothing}

The first statistical model to try is exponential smoothing, which keeps a level where seasonal naive keeps only yesterday. Exponential smoothing keeps a level that moves toward each new reading. The simplest form updates the level as `level_t = alpha * y_t + (1 - alpha) * level_(t-1)`. The constant alpha, between 0 and 1, sets how fast the level follows the data. The forecast for every hour of the horizon is the last level. Recent readings count more than older ones, and the weights fall off with age.

ETS, for error, trend and season, extends that form. Each part is chosen from a short list. The trend can be none, additive or damped. The error can be additive or multiplicative, and the season can be none, additive or multiplicative. A name such as ETS(A,N,A) reads in that order: additive error, no trend, additive season. ETS(M,Ad,M) has a multiplicative error, a damped trend and a multiplicative season.

AutoETS chooses the form for you. Its default criterion is AICc, the corrected Akaike information criterion, which charges a model for each parameter and corrects for small samples (statsforecast {{version:statsforecast}}, `statsforecast/models.py`, the `AutoETS` docstring). Two of its defaults matter for hourly demand. The season length defaults to 1, which means no season at all, and the model gives no warning when you leave it that way. The `model` argument defaults to `"ZZZ"`, which lets the model choose all three parts. For hourly demand with a daily pattern, the call sets `season_length=24`.

Here is the call on the subset. The example fits AutoETS for each client at each cutoff, prints the form the model chose each time, and scores the forecasts beside seasonal naive.

[Example: autoets-subset-backtest](../examples/autoets-subset-backtest.py)

Across the 24 fits, AutoETS chose a form with no season, ETS(A,N,N), for {{out:autoets-subset-backtest:fits that chose ETS(A,N,N)}} of them, and an additive season, ETS(A,N,A), for {{out:autoets-subset-backtest:fits that chose ETS(A,N,A)}}. One chose a damped trend with an additive season, ETS(A,Ad,A), and the other {{out:autoets-subset-backtest:fits with a multiplicative part}} used a multiplicative part, in the error, the season or both. The choice changes by client, and for some clients it changes from one cutoff to the next. The lines printed under the totals give each client's three forms. Three clients, MT_014, MT_064 and MT_370, take the flat form at all three cutoffs. MT_014 is the lead client, the one [Chapter 1](#forecast-tomorrows-demand) met.

A flat form is a result to explain, because the seasonal form contains it. ETS(A,N,N) is ETS(A,N,A) with the seasonal part switched off, so the best seasonal fit cannot have a lower likelihood than the flat fit on the same history. The flat-form check runs both forms on each of the 24 windows.

[Example: flat-form-check](../examples/flat-form-check.py)

In {{out:flat-form-check:seasonal fit has the lower log-likelihood}} of the {{out:flat-form-check:windows}} windows, the seasonal fit has the lower log-likelihood, and in {{out:flat-form-check:seasonal fit has the higher AICc}} it has the higher AICc. Those are the same nine windows in which the default run chose the flat form. AICc charges a model for each parameter, and the seasonal form has 24 more. That charge adds to the gap; it does not cause it, because the seasonal fit's likelihood is already lower before any charge is made. The seasonal fit most likely stopped short of its best value, and the flat form won by default. The book does not chase the optimiser further. It tests the choice instead, with a forced season.

Chapter 1 forced an additive season on the lead client, with `model="ZZA"`, and seasonal naive still had the lower error there. The forced run is the second example, beside the default.

[Example: autoets-forced-season-backtest](../examples/autoets-forced-season-backtest.py)

With the season forced, AutoETS has a mean MASE of {{out:autoets-forced-season-backtest:MASE ZZA}}, against {{out:autoets-subset-backtest:MASE AutoETS}} for the default and {{out:autoets-subset-backtest:MASE SeasonalNaive}} for seasonal naive. Its intervals cover {{out:autoets-forced-season-backtest:coverage 95 ZZA}} of hours. The lead client's MASE goes from {{out:autoets-subset-backtest:MASE MT_014 AutoETS}} to {{out:autoets-forced-season-backtest:MASE MT_014 ZZA}}, and MT_064's from {{out:autoets-subset-backtest:MASE MT_064 AutoETS}} to {{out:autoets-forced-season-backtest:MASE MT_064 ZZA}}. The forced season is not better for every client. MT_370 goes from {{out:autoets-subset-backtest:MASE MT_370 AutoETS}} to {{out:autoets-forced-season-backtest:MASE MT_370 ZZA}}, so forcing the season is a change to test on each client, not a fix to apply to all of them.

**Working rule:** Start with AutoETS for hourly demand with `season_length=24`, and print the chosen forms in every backtest. When a client takes a flat form, check that its seasonal form is not failing. If the seasonal fit has the lower likelihood, force the season with `model="ZZA"`, and score both forms on earlier windows before you keep either: cutoffs before the held-out week that the report does not score, such as the four ordinary days in late November and early December 2014 that Chapter 6 uses. The standard cutoffs show the comparison here, but they do not decide it. A form that changes from one cutoff to the next is the first thing to check when a forecast gets worse.

## Fit a fixed seasonal ARIMA {#fit-a-fixed-seasonal-arima}

The second candidate models what seasonal naive leaves behind. Its differencing at lag 24 starts from the same comparison seasonal naive makes, and the backtests of Chapter 3 are the test for what it adds. An ARIMA model writes the next reading as a weighted sum of earlier readings and earlier errors, after differencing: subtracting a reading from the reading one season earlier. The model is written ARIMA(p,d,q)(P,D,Q)[m]. The non-seasonal part has p autoregressive terms, d differences and q moving-average terms. The seasonal part has the same three, at lags that are multiples of the season length m.

The book's model is ARIMA(1,0,0)(1,1,0)[24]. It has one autoregressive term, no ordinary difference, no moving-average term, one seasonal difference at lag 24, and one seasonal autoregressive term at lag 24. Written out, with B the step back operator, so that B y_t is y_(t-1):

`(1 - phi1 B)(1 - Phi1 B^24)(1 - B^24) y_t = e_t`

The term `(1 - B^24)` replaces each reading with its change from the same hour yesterday. The lag of 24 is the point of the model for hourly demand. Each hour is compared with the same hour yesterday, which is the comparison seasonal naive makes, and the autoregressive terms then model what that comparison leaves behind. The seasonal difference removes the level, so the model has no mean term. statsforecast's `ARIMA` leaves the mean off for differenced series, by default (`statsforecast/models.py`, the `ARIMA` docstring).

We fix the order rather than search for it. [Why the book avoids automatic ARIMA](#why-the-book-avoids-autoarima) explains why. The example below fits the same form with statsmodels {{version:statsmodels}}, on the lead client's history before the first standard cutoff. It prints the names of the fitted parameters, so you can check that the model has the terms you wrote down.

[Example: arima-summary-look](../examples/arima-summary-look.py)

The list has three names: `ar.L1`, `ar.S.L24` and `sigma2`. The first is the autoregressive term, which says how much an hour's change from the same hour yesterday carries into the next hour. The second is the seasonal term at lag 24, which says how much a day's change at an hour carries into the same hour the next day. The third is the variance of the error. The list has no intercept, and that tells you nothing about the differencing. statsmodels adds no intercept to a seasonal ARIMA unless `trend` is set, so the list has none whether or not the series is differenced. The seasonally differenced model needs none, and statsforecast's `ARIMA` leaves the mean off for differenced series by default. A full statsmodels summary also gives each value. This book does not quote them. The values change with the window and with the library version, and a reader needs to know what each term means more than its current number.

The fixed model is then one more candidate in the backtest, beside AutoETS and seasonal naive. Its scores come in the [comparison section](#compare-models-on-the-same-cutoffs) below.

**Working rule:** When a search is too slow to run at every cutoff, fix the order and write it in the call. Print the summary once, and check that the terms it lists are the ones you meant. Refit the fixed order at every cutoff, as you would refit any model. The order stays fixed. The coefficients do not.

## Compare models on the same cutoffs {#compare-models-on-the-same-cutoffs}

Three models now forecast the same 24 hours at the same three cutoffs, from the same eight weeks of history, for the same eight clients, and the forced-season AutoETS is a fourth row. The AutoETS and ARIMA backtests are two examples, not one, and the forced run is a third, so that each finishes well inside the 45 seconds the book allows an example on two CPUs.

[Example: arima-subset-backtest](../examples/arima-subset-backtest.py)

The scores are these. MAE is the mean absolute error in kilowatt hours, and RMSE is the root mean squared error, which punishes large misses more. Both are computed for each client over its 72 scored hours, then averaged over the eight clients, so each client counts once. MASE divides each client's MAE by the MAE that seasonal naive makes in its own history, over every reading before Dec 29, 2014, from each client's first reading and with the clock-change hours left out, then averages over clients. A MASE of 1 means the model is as accurate as seasonal naive was on its own past. The coverage of AutoETS is the share of scored hours that fall inside its 95% prediction interval, the range the model gives for each hour.

| Model | MAE (kWh) | RMSE (kWh) | MASE | Clients beating seasonal naive on MASE |
| --- | --- | --- | --- | --- |
| SeasonalNaive | {{out:autoets-subset-backtest:MAE SeasonalNaive}} | {{out:autoets-subset-backtest:RMSE SeasonalNaive}} | {{out:autoets-subset-backtest:MASE SeasonalNaive}} | not applicable |
| AutoETS | {{out:autoets-subset-backtest:MAE AutoETS}} | {{out:autoets-subset-backtest:RMSE AutoETS}} | {{out:autoets-subset-backtest:MASE AutoETS}} | {{out:autoets-subset-backtest:clients where AutoETS has lower MASE}} |
| AutoETS, season forced (ZZA) | {{out:autoets-forced-season-backtest:MAE ZZA}} | {{out:autoets-forced-season-backtest:RMSE ZZA}} | {{out:autoets-forced-season-backtest:MASE ZZA}} | {{out:autoets-forced-season-backtest:clients where ZZA has lower MASE}} |
| ARIMA | {{out:arima-subset-backtest:MAE ARIMA}} | {{out:arima-subset-backtest:RMSE ARIMA}} | {{out:arima-subset-backtest:MASE ARIMA}} | {{out:arima-subset-backtest:clients where ARIMA has lower MASE}} |

Read the last column as a count out of the eight clients, not as a score. The AutoETS intervals cover {{out:autoets-subset-backtest:coverage 95 AutoETS}} of the scored hours, and the forced-season intervals cover {{out:autoets-forced-season-backtest:coverage 95 ZZA}}. The nominal share is 0.95, so the intervals are a little narrow on this sample. Coverage checks the intervals, not the point forecast. An interval that covers too little is too narrow, and a planner who buys to it will be short more often than its label says. Chapter 8 returns to intervals and shows how to calibrate them on earlier windows.

The two measures disagree about AutoETS. Its mean MAE is the lowest of the four rows, and its mean MASE is the highest. MAE in kilowatt hours gives more weight to the clients with large loads, because their errors are large. MASE scales each client's error by that client's own history, so every client counts about the same. The per-client lines in the AutoETS example show where the MASE comes from. MT_064 has a MASE of {{out:autoets-subset-backtest:MASE MT_064 AutoETS}}, against {{out:autoets-subset-backtest:MASE MT_064 SeasonalNaive}} for seasonal naive, and it is one of the flat-form clients. MT_217 has {{out:autoets-subset-backtest:MASE MT_217 AutoETS}}, against {{out:autoets-subset-backtest:MASE MT_217 SeasonalNaive}}, and its form changes from one cutoff to the next. The ranking depends on the measure, and the choice of measure is part of the decision.

Three windows are a small sample. AutoETS beats seasonal naive on {{out:autoets-subset-backtest:clients where AutoETS has lower MASE}} of the eight clients, and ARIMA on {{out:arima-subset-backtest:clients where ARIMA has lower MASE}}. ARIMA is better than seasonal naive on both measures, and on MASE by {{out:arima-subset-backtest:MASE gap, SeasonalNaive minus ARIMA}}. Three windows can't settle a gap that small.

**Working rule:** Score every candidate on the same cutoffs, with the same history and the same scale, and report MAE and MASE side by side. When the two disagree, say which one the decision needs. We recommend more windows before choosing between two models that differ by less than about a tenth of MASE. That threshold is a judgement from this sample, not a measurement.

## Meet the late starters {#meet-the-late-starters}

The subset has one late starter. MT_115's first non-zero reading is at 00:00 on Jan 14, 2014. Before that, the meter reports zeros, and the data flags those hours `before_first_reading`. Those zeros are not low use. A model that reads them learns a level the client does not have. A late starter is a client whose first reading falls inside the history its model reads, and the start rule is the fix: a series begins at its first reading. This matters for the supplier's decision. A level pulled low means buying too little for that client, and the case says buying too little costs more than buying too much.

The standard December cutoffs do not meet this case. At those cutoffs, the eight weeks before each one are all readings. So this section uses February cutoffs: Feb 1, 2014, Feb 15, 2014 and Feb 28, 2014. Their eight-week histories still reach back before the first reading, so the case is in them. [Chapter 6](#forecast-many-clients-at-once) applies the same rule to all thirty clients.

### A history that starts with zeros {#a-history-that-starts-with-zeros}

Start with the raw history: the eight weeks before each February cutoff, zeros included. The share of zero hours in that history is {{out:late-starter-trim:zero percent before 2014-02-01}}% on Feb 1, 2014, {{out:late-starter-trim:zero percent before 2014-02-15}}% on Feb 15, 2014 and {{out:late-starter-trim:zero percent before 2014-02-28}}% on Feb 28, 2014. The share falls as the cutoff moves past the first reading.

The example below runs both fits on the same three cutoffs.

[Example: late-starter-trim](../examples/late-starter-trim.py)

AutoETS fits the raw history without an error and forecasts without a warning. The failure is in the level. On Feb 1, 2014 it chooses the flat form, {{out:late-starter-trim:raw form 2014-02-01}}, and its forecast averages {{out:late-starter-trim:raw mean forecast 2014-02-01}} kWh an hour over the day. The readings that followed average {{out:late-starter-trim:raw mean actual 2014-02-01}} kWh. The MAE on that day is {{out:late-starter-trim:MAE raw 2014-02-01}} kWh. The likely cause is the zeros: they pull the level down, and the model cannot tell that they are not use.

On the other two days the raw history's error is much smaller: {{out:late-starter-trim:MAE raw 2014-02-15}} kWh on Feb 15, 2014 and {{out:late-starter-trim:MAE raw 2014-02-28}} kWh on Feb 28, 2014. So the damage depends on how many zeros the window holds. The rule is right whatever the zero share, because leading zeros are not use.

### Start each series at its first reading {#start-each-series-at-its-first-reading}

The start rule says that a series begins at its first reading. It drops every reading before the client's first non-zero reading, and it keeps every reading after that, zeros included. The zeros after the start are real low use, and a model needs them. The rule removes leading zeros only, so a window that begins after the first reading keeps all its hours. The rule lives in `examples/lib/starts.py`, so later chapters call the same code. Its function `history_from_start(df, cutoff)` returns one cutoff's history under the rule: the readings in the eight weeks before the cutoff, from the client's first reading onward. It finds that first reading from the rows before the cutoff only, so no reading after the cutoff reaches the model. It raises an error if a client has no reading before the cutoff, or fewer than 48 hours of history.

Before the rule is used, check that it does what the flag says. `check_start_rule` counts the rows the rule would drop across the whole series, and raises an error unless those rows are exactly the rows flagged `before_first_reading`. On the subset, the rule drops {{out:late-starter-trim:start rule drops, SUBSET8}} hours, and the flag marks {{out:late-starter-trim:flagged before_first_reading, SUBSET8}}. They match. The only late starter in the subset is MT_115.

With the rule, the history at Feb 1, 2014 is {{out:late-starter-trim:start history hours 2014-02-01}} hours instead of 1,344. It grows with the cutoff: {{out:late-starter-trim:start history hours 2014-02-15}} hours on Feb 15, 2014 and {{out:late-starter-trim:start history hours 2014-02-28}} hours on Feb 28, 2014. The start history picks {{out:late-starter-trim:start form 2014-02-01}} on Feb 1, 2014, a form with multiplicative error, damped trend and additive season, and the same form on the other two days. The raw history changed its form between days: {{out:late-starter-trim:raw form 2014-02-01}} on Feb 1, 2014, then {{out:late-starter-trim:raw form 2014-02-15}} on Feb 15, 2014 and {{out:late-starter-trim:raw form 2014-02-28}} on Feb 28, 2014. The three-window MAE falls from {{out:late-starter-trim:MAE raw, three windows}} kWh to {{out:late-starter-trim:MAE start, three windows}} kWh.

The gain is not even across the windows, and all of the three-window gain comes from Feb 1, 2014. On that day the MAE falls from {{out:late-starter-trim:MAE raw 2014-02-01}} to {{out:late-starter-trim:MAE start 2014-02-01}} kWh. On Feb 15, 2014 the rule does worse, {{out:late-starter-trim:MAE start 2014-02-15}} kWh against {{out:late-starter-trim:MAE raw 2014-02-15}}, and on Feb 28, 2014 it does a little better, {{out:late-starter-trim:MAE start 2014-02-28}} against {{out:late-starter-trim:MAE raw 2014-02-28}}. Over the two days after the first together, the raw history does slightly better, with a mean MAE of {{out:late-starter-trim:MAE raw, 2014-02-15 and 2014-02-28}} kWh against {{out:late-starter-trim:MAE start, 2014-02-15 and 2014-02-28}}. The fits show how it loses. On Feb 15, 2014 the start history is 768 hours, and its damped trend forecasts lower than the raw fit does: {{out:late-starter-trim:start mean forecast 2014-02-15}} kWh against {{out:late-starter-trim:raw mean forecast 2014-02-15}}, when the readings averaged {{out:late-starter-trim:raw mean actual 2014-02-15}} kWh. That low forecast is the cost the case warns about, and it is the cost of the rule on that day. One client and three days are a small sample, so the figures illustrate the rule and do not carry it. The rule removes only the zeros before a client's first reading, which are not use.

**Working rule:** When a client's history holds zeros, check whether its first non-zero reading falls inside the history. If it does, apply the start rule before any model sees the history, whatever the zero share, because leading zeros are not use. Run `check_start_rule` on the whole series first, and if it fails, find out why before you trim anything. Print the history length at each cutoff, because a short history is the first sign that the rule has taken too much.

## Why the book avoids automatic ARIMA at season 24 {#why-the-book-avoids-autoarima}

AutoARIMA searches for the orders instead of fixing them. It runs a stepwise search over candidate orders, picks the best by AICc, which is its default, and stops after at most 94 candidate models (statsforecast {{version:statsforecast}}, `statsforecast/models.py`, and the stop message in `statsforecast/arima.py`). Each candidate is a full state-space fit. With a season of 24, each candidate that has seasonal terms at lag 24 is a slow fit, and the search tries dozens of them, so it takes far longer than the fits in this chapter. On the book's pinned setup, two CPUs with statsforecast {{version:statsforecast}}, one window for one client took about 75 seconds. That was one run on MT_014 at the first standard cutoff. The subset needs 24 such windows, so one backtest of it would take about half an hour, against the 45 seconds the book allows an example. That is this book's own measurement, not an example in it, and it depends on the setup, so time it on your own machine before you plan a schedule around it. The fixed order costs one fit per window, which is the difference that matters here. Its weak point is a client whose pattern differs from the order's assumptions. The backtest is the check for that, and the [scores in the comparison section](#compare-models-on-the-same-cutoffs) are the place to look.

A fixed order is a judgement, not a finding. We chose (1,0,0)(1,1,0) with a season of 24 because the data has a daily pattern and the backtests in this chapter need a model that finishes. The order sits in the call, where a reader can see it and change it. Later chapters compare against it on the same windows, so a change to the order shows up in the scores.

**Working rule:** Do not run an automatic search with a season of 24 on hourly data in a schedule that refits every day. Time the search on one client before you commit to it. If it is too slow, fix the order and test that order in the backtest.

## What goes wrong in production {#what-goes-wrong-ch4}

- **A missing season.** A season length left at 1 gives a model with no daily pattern, and no error. The sign is that every client chooses a form with no season, such as ETS(A,N,N). A few clients doing so is a different case: a seasonal fit that fails, which the flat-form check in [the exponential smoothing section](#fit-exponential-smoothing) tests. Log the chosen forms at each refit.
- **A late starter's zeros.** A client that begins inside the history has its level pulled down until its first reading leaves the window. The sign is a forecast that runs low for days after the start. Count the zero share of each client's history before each refit, and apply the start rule to any client whose first reading falls inside the history.
- **Forms that change.** The chosen form can change between refits for one client. That is not wrong in itself, but a switch from a seasonal form to one with no season is worth checking against the readings. Keep the forms in the run log.
- **Costs that add up.** Fits scale with clients, times windows, times refits. Count them before you add clients to a schedule.
- **Intervals that miss.** The AutoETS intervals covered {{out:autoets-subset-backtest:coverage 95 AutoETS}} of hours on this sample, short of the nominal 0.95. Chapter 8 shows how to calibrate bands on earlier windows, and until a band is calibrated that way, do not present it as a 95% range.
- **How you would notice.** Run seasonal naive beside every statistical model on each day's windows, and log the MASE gap. A gap that changes sign from day to day is the signal to look at the model, not a result to report.

## Decisions this chapter made {#decisions-ch4}

- **AutoETS with a season of 24.** The default `model="ZZZ"` and the forced-season run, `model="ZZA"`, are both reported, because a flat choice can be a failed seasonal fit. This chapter makes no choice between the two forms: the comparison on the standard cutoffs is reported, not used to choose, and the choice belongs on earlier windows. Chapter 1 forced `ZZA` on the lead client, and this chapter shows the forced run for all eight.
- **A fixed ARIMA(1,0,0)(1,1,0)[24], with no search.** This is a stated judgement, and the backtest is what tests it.
- **Eight clients, not thirty.** Chapter 6 fits all thirty clients with one model. A per-client fit of AutoETS for all thirty over three windows takes about 97 seconds on the pinned setup, against the 45 seconds the book allows an example, so this chapter uses the subset.
- **The subset, the cutoffs and the scale.** SUBSET8, the three standard cutoffs from Dec 29, 2014 to Dec 31, 2014, 1,344 hours of history, and seasonal naive as the bar in every comparison. The MASE scale reads each client from its first reading, with the clock-change hours left out.
- **The clock-change hours.** Every subset example masks them and fills them from history, as the backtest chapter sets out. None of the three windows holds a clock-change hour, so the masking changes no forecast and no scored hour. It does change the MASE scale, because the scale reads all history before Dec 29, 2014, and it counts MT_115 from Jan 14, 2014.
- **Late starters: February cutoffs and the start rule.** The December cutoffs do not meet MT_115's late start, so the February cutoffs do. Chapter 6 applies the rule to all thirty clients.
- **Coverage for the two AutoETS runs only.** The ARIMA backtest is scored without intervals here.
- **No fitted values quoted.** The ARIMA example prints parameter names only, and the flat-form check prints counts only. Errors are rounded to two decimals and coverage to three.
- **A statistical baseline for Chapter 5.** The per-client AutoETS and ARIMA scores on these windows are the bar the gradient boosting model must beat.
