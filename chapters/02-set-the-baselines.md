---
chapter_id: beat-the-naive-baselines
---

# Set the baselines a model must beat

By the end of this chapter you can compute three baselines for a client's day-ahead forecast, naive, seasonal naive and average of weeks, and scale a client's error with MASE, using a scale that reads only the readings before the forecast week.

Chapter 1 ended with two scores on the held-out week, and neither can be judged yet, because an error in kWh means nothing on its own. The supplier has to set each model's error against what the simplest forecast scores on the same hours, and against how much the client's demand moves from one day to the next. This chapter builds both comparisons. The supplier needs them before it can say that a model is worth its cost.

Every error is a miss in kilowatt hours for one hour, and the scores here are hourly averages of those misses. A baseline's score is the same kind of number. It says how many kilowatt hours of error the simplest rule leaves for an average hour of the day, and a model that cannot beat that figure has not earned its extra cost. So each score is read against the lead client's mean demand as well as against the other rows.

## Why a baseline comes first {#why-a-baseline-comes-first}

A baseline is a forecast that needs no fitting. It copies a value the history already holds, so it costs almost nothing to run, and it sets a bar that any model has to clear. A model earns its place only by beating a forecast that cost nothing. Fitting time, tuning and keeping a model running are all costs, and the baseline's score measures what those costs buy.

Three terms matter here. The cutoff is the hour at which a forecast is made. It is the first hour the forecast covers, and the history is every reading before it. The horizon is how far ahead a forecast reaches, here 24 hours. A window is one cutoff together with the horizon it covers, and a backtest is the score of many windows, each made as it would have been made on its day.

The comparison has to be fair, or it measures the setup and not the model. A baseline scored on different hours, with a different history or a different scoring routine, says nothing about the model beside it. So every method in this chapter takes its cutoffs from `held_out_cutoffs()`, the seven daily cutoffs at 00:00 of the held-out week, reads its history through `history_before()`, which takes the eight weeks of hours before a cutoff, and is scored by the same code. These are the seven windows Chapter 1 used.

**Working rule:** Score every baseline on the same windows, histories and scoring code as the model it will be compared with, and count each baseline's missing forecasts before you read its score. A model that beats a baseline scored on different hours has not beaten it.

## Three baselines to beat {#three-baselines-to-beat}

Each baseline copies the history in a different way. Naive carries the last hour it has seen forward across the whole day. Seasonal naive copies the day before, hour for hour. Average of weeks takes the mean of that hour on that weekday over the four weeks before.

Every rule covers the 24 hours after a cutoff. Call the cutoff `T`, the hour at which the forecast is made, and let `k` run from 0 to 23 across the hours the forecast covers. The three rules are these:

- Naive repeats the last reading before the cutoff: `forecast(T + k) = y(T - 1)`.
- Seasonal naive repeats the reading one day earlier: `forecast(T + k) = y(T + k - 24)`.
- Average of weeks takes the mean of the same hour on the same weekday over the four weeks before: `forecast(T + k) = (y(T + k - 168) + y(T + k - 336) + y(T + k - 504) + y(T + k - 672)) / 4`.

The lags are in hours, so 168 is one week. Every reading a rule uses comes from before the cutoff: naive's is the hour before it, seasonal naive's are the 24 hours before it, and average of weeks reaches back one to four weeks. Take the window at 00:00 on Dec 25, 2014 as a worked case. Naive gives every hour of Dec 25 the reading at 23:00 on Dec 24, so the forecast is flat. Seasonal naive gives 00:00 on Dec 25 the reading at 00:00 on Dec 24, and so copies the whole previous day's shape. Average of weeks gives 00:00 on Dec 25 the mean of the 00:00 readings on Dec 18, Dec 11, Dec 4 and Nov 27. That mean smooths out one unusual day, but it also averages in any level change across the four weeks.

Each rule fails in a known way. Naive gives every hour of the day the same value, so it cannot follow the daily peak at all. Seasonal naive copies one day, so an unusual day is copied into the forecast for the next day, as the per-day errors later in this section show on the held-out week. Average of weeks needs four clean weeks, and a holiday in any of them enters the mean with the same weight as an ordinary day. A baseline's score therefore describes its rule on the windows it was run on, and these failures show up only on the windows where they happen.

Naive and seasonal naive come from statsforecast ({{version:statsforecast}}), whose classes `Naive` and `SeasonalNaive` are in `statsforecast/models.py`. The seasonal naive rule is the one [Hyndman and Athanasopoulos](https://otexts.com/fpp3/toolbox.html#sec-simple-methods) give for seasonal data: the forecast is the last observed value from the same season. Average of weeks is not a library method, so `examples/lib/baselines.py` computes it in pandas, where the rule can be read, and each window reads only its own history. A lag missing from that history raises an error instead of being skipped. One script scores all three rules through those seven windows, and its first lines count the windows, scored hours and missing forecasts, so you can check them before you read a score.

[Example: naive-baselines-held-out-week](../examples/naive-baselines-held-out-week.py)

The three scores are comparable only because they cover the same hours: {{out:naive-baselines-held-out-week:scored hours}} scored hours in {{out:naive-baselines-held-out-week:windows}} windows, one for each hour of the seven days, and {{out:naive-baselines-held-out-week:missing forecasts}} missing forecasts across the three rules. That count is the one the [working rule in the first section](#why-a-baseline-comes-first) asks for, and zero is the only value that lets the three be compared on the same hours.

Read each error below against the lead client's mean demand of {{out:naive-baselines-held-out-week:mean demand}} kWh. The mean absolute error, MAE, is the mean of the absolute differences between forecast and reading, in kWh. The three baselines score:

- naive: {{out:naive-baselines-held-out-week:naive MAE}} kWh
- seasonal naive: {{out:naive-baselines-held-out-week:seasonal naive MAE}} kWh
- average of weeks: {{out:naive-baselines-held-out-week:average of weeks MAE}} kWh

The simplest rule has the lowest error on this week, and average of weeks, which averages four weeks, is the weakest. Nothing here explains that order, and this is one client and one week, so it is a result for this week only. Chapter 3 is where the book backtests seasonal naive across many cutoffs.

The totals hide where the errors sit. The week begins on Dec 25, 2014, a Portuguese national holiday, and its first window is where the rules part company. Naive has the lowest error on it, at {{out:naive-baselines-held-out-week:naive MAE Dec 25 window}} kWh, against {{out:naive-baselines-held-out-week:seasonal naive MAE Dec 25 window}} kWh for seasonal naive. Average of weeks is worse still, and its largest error of the week falls on that window, at {{out:naive-baselines-held-out-week:average of weeks worst window MAE}} kWh, because its four lags are ordinary days and the holiday has no match among them.

The day after the holiday is where copying costs seasonal naive. Its largest error of the week falls on {{out:naive-baselines-held-out-week:seasonal naive worst window}}, at {{out:naive-baselines-held-out-week:seasonal naive worst window MAE}} kWh, because that window copies the holiday's pattern. On that day naive is still the lowest, at {{out:naive-baselines-held-out-week:naive MAE Dec 26 window}} kWh.

We recommend reporting all three baselines every time. Use seasonal naive as the main reference for MASE, because it carries the daily cycle that makes hourly demand predictable, and keep naive beside it, because on some weeks the flat rule is the stronger one.

**Working rule:** Start from naive and seasonal naive, both from the library, and add average of weeks when the series has a weekly rhythm worth testing. Before you trust a baseline, check one forecast by hand against its formula: pick one cutoff, compute one hour from the readings in its history, and compare the result with the forecast.

## Scale errors with MASE {#scale-errors-with-mase}

A kilowatt-hour error means different things for different clients. A miss of {{out:naive-baselines-held-out-week:seasonal naive MAE}} kWh an hour is small for a client that uses {{out:naive-baselines-held-out-week:mean demand}} kWh an hour, and the same miss would be large for a hypothetical client that uses 4 kWh an hour. A scaled error removes that difference. It divides each absolute error by a scale, the typical error of a simple method on the same client's own history, and MASE, the mean absolute scaled error, is the mean of those scaled errors. [Hyndman and Koehler](https://robjhyndman.com/papers/mase.pdf) introduced it in a 2006 paper in the *International Journal of Forecasting*. The link is the authors' preprint, dated 2005. They noted that the scaled error does not depend on the units of the data, so the same number can be read across clients of very different size. This chapter scores one client, so that comparison across clients is the reason for the scale, not a result shown here.

The 2006 paper scales by the one-step naive error: the mean absolute difference between each reading and the one before it. That is the natural scale for non-seasonal series. For hourly demand, this book uses the seasonal form that [Hyndman and Athanasopoulos](https://otexts.com/fpp3/toolbox.html#sec-accuracy) give for seasonal data, because the comparison a reader wants is with seasonal naive: under the seasonal scale, the denominator is seasonal naive's error on the client's own history. A MASE below 1 then says the forecast's error on the scored hours is smaller than seasonal naive's error on that history. A one-step scale would compare it with a method this book does not use for hourly demand.

The two formulas are these:

- `scale = mean of |y(t) - y(t - 24)|` over every reading t before the first test day, with the pairs that touch a masked clock-change hour left out
- `MASE = mean of |actual(t) - forecast(t)| / scale` over the scored hours

The scale reads every reading before the first test day, {{out:mase-scale-leak-check:scale rows from history}} hours, about two years. That is longer than each window's eight-week history, and a longer history gives a steadier estimate of the client's usual swing. One scale for the week also means every window is read against the same number. The function `seasonal_scale` in `examples/lib/baselines.py` computes it for one client or for many, and returns one value per client. To see whether the cut matters, score the three baselines on the held-out week twice: with the scale from history, and with the scale from the whole series, both with the clock-change readings masked.

[Example: mase-scale-leak-check](../examples/mase-scale-leak-check.py)

The scale from history is {{out:mase-scale-leak-check:scale from history clock changes masked}} kWh, the average gap between each hour and the same hour the day before, and it reads against the mean demand the way an MAE does. A scale that is a large share of the mean says the client's own day-to-day swing is large, and every error for that client is read against that swing. Dividing seasonal naive's MAE of {{out:naive-baselines-held-out-week:seasonal naive MAE}} kWh by the scale gives its MASE, {{out:mase-scale-leak-check:seasonal naive MASE from history scale}}. utilsforecast's `mase`, reached through the helper `mase_scores`, computes this MASE by dividing the same MAE by the same seasonal scale (utilsforecast {{version:utilsforecast}}, `utilsforecast/losses.py`, function `mase`), so the division by hand gives the same figure.

utilsforecast ({{version:utilsforecast}}) computes MASE with its `mase` function in `utilsforecast/losses.py`. The function takes its scale from the `train_df` it is given, and it shifts that frame by rows, not by time, so a missing hour moves every later lag and nothing reports it. It also needs sorted input. Its time filter runs only when the scored frame has a `cutoff` column (function `_create_train_with_cutoffs`). So the library does not decide which readings the scale may read. The caller does, and the next section is about that choice.

**Working rule:** Report MASE beside MAE whenever the clients you compare differ in size, and say in the same sentence which history the scale comes from. Read MASE as a ratio of two mean errors, one from the scored hours and one from the history. It is not a percentage gain over seasonal naive, even when the MASE is well below 1.

## Take the MASE scale from history {#take-the-mase-scale-from-history}

A scale describes the history before the first test day, not the series as a whole. If the scale reads the whole series, it also reads the hours the forecast is scored on, so the test week enters the denominator of its own error. That is a leak, and it raises no error. The MASE it produces looks like any other. The rule in this book is that a scale reads only the readings before the first test day of the week it scores, which for the held-out week is Dec 25, 2014.

A whole-series scale looks like the safer choice, because it uses more readings. It uses more readings from the wrong period. The extra readings are the ones the forecast is judged on, so the scale is no longer a measure of what was known at the cutoff. More data does not fix that. Only the cut does.

The script in [the previous section](#scale-errors-with-mase) prints the rows each scale reads, and those lines are the check. The history scale reads {{out:mase-scale-leak-check:scale rows from history}} readings, and its last reading is {{out:mase-scale-leak-check:last scale reading from history}}, the hour before the first test day. A later date there would mean test hours reached the scale.

The whole-series scale reads {{out:mase-scale-leak-check:scale rows from whole series}} readings, through {{out:mase-scale-leak-check:last reading of whole series}}, which covers the whole held-out week. Both scales have their clock-change readings masked, so the test week is the only difference between them.

The whole-series scale is the larger one, so every MASE computed with it is smaller. At two decimals both scales give seasonal naive {{out:mase-scale-leak-check:seasonal naive MASE from whole-series scale}}, so the leak does not show. At four decimals it does, with seasonal naive's MASE at {{out:mase-scale-leak-check:seasonal naive MASE from history scale at four decimals}} against {{out:mase-scale-leak-check:seasonal naive MASE from whole-series scale at four decimals}}, and the factor below is the lesson, not the rounding.

The scales are {{out:mase-scale-leak-check:scale from history clock changes masked}} kWh from history and {{out:mase-scale-leak-check:scale from whole series clock changes masked}} kWh from the whole series, and the same factor, {{out:mase-scale-leak-check:seasonal naive MASE ratio whole-series over history}}, shrinks each of the three rules.

The leak is small on this week because the test week is a small share of the series. It makes up {{out:mase-scale-leak-check:test week share of whole-series rows in percent}} percent of the whole-series readings, so the whole-series scale is roughly the history scale moved toward the test week's own swing by that share. The test week's own mean day-to-day difference is {{out:mase-scale-leak-check:test week own mean day-to-day difference}} kWh, against {{out:mase-scale-leak-check:scale from history clock changes masked}} kWh for the history, so the week swings more than the history does. The share times that gap is the leak, and the ratio above is its size. The leak grows in two ways. It grows when more of the scale's rows are test hours, as in a test period much longer than one week, such as the year of cutoffs in Chapter 3. It also grows when the test week's swing sits further from the history's. A quieter week would pull the whole-series scale down and make every MASE larger, so the same leak can flatter a model in one week and penalise it in another.

The clock changes are a second, smaller point, and this chapter handles them as Chapter 3 does. The history holds {{out:mase-scale-leak-check:clock-change rows in scale history}} clock-change readings: the two gap hours and the two repeat hours of the March and October changes in 2013 and 2014. The gap hours read zero, and the repeat hours hold two hours of energy under one label, as the [data card](../examples/data/README.md) describes. Kept as recorded, those pairs give a scale of {{out:mase-scale-leak-check:scale from history as recorded}} kWh. Masked, the scale is {{out:mase-scale-leak-check:scale from history clock changes masked}} kWh, because the pairs that touch a masked reading drop out of the mean. A masked reading is neither zero nor a copy; it is missing. The chapter's scale masks them, and [Chapter 3 gives the reason](#meet-the-clock-changes), which sets them to missing in every training set and every score.

**Working rule:** Pass a scale only the readings before the first test day, mask its clock-change readings, and print the last reading it uses. Check that this reading is the hour before the first test day. A small difference between the two scales does not show the procedure is right. The row check does.

## Read a baseline's score {#read-a-baselines-score}

MASE compares a forecast with seasonal naive's error on the client's own history. Below 1, the forecast's error on the scored hours is smaller than the error seasonal naive made on that history; above 1, it is larger. [Hyndman and Koehler](https://robjhyndman.com/papers/mase.pdf) state this reading in their paper for the one-step scale, and the arithmetic is the same for the seasonal scale. The book gives no threshold for a good score. What counts as good depends on what a miss costs the supplier, which the supplier knows and this book does not.

The table puts the three baselines side by side, one row each. The MAE is in kWh, and the MASE uses the history scale with the clock changes masked. The leak from [the previous section](#take-the-mase-scale-from-history) does not show at two decimals, so the table has one MASE column.

| Baseline | MAE (kWh) | MASE |
| --- | --- | --- |
| Naive | {{out:naive-baselines-held-out-week:naive MAE}} | {{out:mase-scale-leak-check:naive MASE from history scale}} |
| Seasonal naive | {{out:naive-baselines-held-out-week:seasonal naive MAE}} | {{out:mase-scale-leak-check:seasonal naive MASE from history scale}} |
| Average of weeks | {{out:naive-baselines-held-out-week:average of weeks MAE}} | {{out:mase-scale-leak-check:average of weeks MASE from history scale}} |

All three MASE values are above 1, so every baseline missed by more than seasonal naive's typical error on the history. For seasonal naive itself, the week's day-to-day swing was larger than the history's. Its MAE equals the test week's own day-to-day difference, {{out:mase-scale-leak-check:test week own mean day-to-day difference}} kWh, against {{out:mase-scale-leak-check:scale from history clock changes masked}} kWh for the history. For naive and average of weeks, the scale compares them with seasonal naive, not with their own past, so this table does not say how those two rules did on the history. The week opens on a national holiday, and the [earlier section](#three-baselines-to-beat) shows the holiday window as the largest error for average of weeks. The book does not test the remaining days here.

The order is the same on both measures, naive first, seasonal naive second and average of weeks third. That agreement is not a finding. MASE divides every MAE by the same scale, so the order cannot differ between MAE and MASE for one client.

The table also raises the supplier's question: is the simplest rule good enough to buy on? That depends on what an hour of error costs, and the MASE does not price it. What the MASE gives the supplier is a comparison it can repeat. A later model can be scored with the same helpers, so its MASE sits beside these rows without a change of units.

Take the table in this order. First, confirm from the [earlier section](#three-baselines-to-beat) that every row scores the same hours, with no missing forecasts. Second, read the MASE column against 1. Third, compare each row with the others, not with a number from a textbook. A fitted model that is only as good as the best row has not shown much, and one that is better by less than the printed rounding has not shown anything at all.

**Working rule:** Read each baseline's MASE against 1 and against the other baselines, and do not name a winner on one week. A model counts as better only when it beats the best baseline on more than one set of windows, and when the margin is larger than the rounding of the printed scores.

## What goes wrong in production {#what-goes-wrong-ch2}

A baseline that is not like-for-like misleads quietly. A team may score its model on a year of cutoffs and its baseline on one week, or give the model a longer history than the baseline. The comparison then measures the setup, and the printed numbers look like evidence. The check is the one this chapter uses. Every method takes its cutoffs from `held_out_cutoffs` and its history from `history_before`, and the count of missing forecasts is printed for each run.

[The leak in the scale](#take-the-mase-scale-from-history) raises no error, and its MASE looks like any other. In this chapter it multiplies every seasonal naive MASE by {{out:mase-scale-leak-check:seasonal naive MASE ratio whole-series over history}}, a change that two decimals hide. A larger leak in another week would look the same on the page. The check is the printed row count and the last reading the scale uses, both on one line of output that a reviewer can read without rerunning anything.

A gap in the seasonal lag gives a wrong scale without any message. With a missing hour, a lag of 24 rows reaches 25 hours back, so each pair for the next day compares the wrong hours. The library's `mase` has no gap check of its own. The helper `seasonal_scale` checks that each client's hours are unbroken, and `mase_scores` runs the same check on the scale's rows before the library reads them. The check raises an error rather than using an assertion, because Python removes assertions under `-O`. The scored hours are checked only for a reading and for duplicates. A production run should count the hours in the feed before it computes any scale, because a check inside the helper reports the problem after it has reached the scale.

The seasonal lag can also be chosen for the wrong cycle. A lag of 24 suits hourly demand with a daily cycle. Daily data has a weekly cycle, and hourly data with a weekly rhythm may need a lag of 168 instead. A scale computed with the wrong season is still a number, so nothing in the output shows the mistake. The check is to name the season beside the scale in the prose and to confirm it against the data's cycle before the scale is used.

Picking a setting on the held-out week spoils the comparison. Average of weeks has a choice in it: how many weeks to average. If that number is picked because it gives the best score on the held-out week, the baseline has been fitted to the week it is judged on, and the comparison with the models is no longer fair. This book fixed four weeks before any score was printed, and this chapter keeps it for every window. A change to it belongs in the decision list at the end of the chapter, made before the scores are seen.

**Working rule:** Before you trust a baseline or a scale, check its inputs in this order: the same windows for every method, a scale that reads only earlier readings with its last reading printed, an unbroken hourly history for the seasonal lag, and a season that matches the data's cycle. Each check is one line of output or one count, and a failed check stops the comparison until it is fixed.

## Decisions this chapter made {#decisions-ch2}

This chapter made seven choices, and the hand-off to Chapter 3 is the last of them. Most are in `examples/lib/baselines.py`, so a later edition that changes one changes it everywhere it is used. The history length and the horizon are the settings in `examples/lib/settings.py`, the same ones Chapter 1 used, so no baseline here reads a different history from the forecasts beside it.

- The MASE scale is seasonal, with a lag of 24 hours. The one-step scale of the 2006 paper is not used.
- The scale is one number per client for the held-out week, from every reading before Dec 25, 2014. It is longer than each window's history, as set out above. utilsforecast can take a scale per cutoff when the frame carries a cutoff column, but this chapter does not use that option. If you add one, know that its filter keeps `ds <= cutoff`, so the cutoff hour itself is included. A cutoff in this book is the first hour a forecast covers, and its reading is not history. A cutoff column built from this book's cutoffs must therefore be shifted back one hour first.
- Average of weeks uses the four weeks before each forecast day, is computed in pandas, and reads only its own eight-week history.
- Naive and seasonal naive are the statsforecast classes. Scores are pooled over every scored hour of the week and printed to two decimals.
- The scale masks its {{out:mase-scale-leak-check:clock-change rows in scale history}} clock-change readings, following Chapter 3's rule. The printed history scale is then {{out:mase-scale-leak-check:scale from history clock changes masked}} kWh.
- The baselines draw no random numbers, so the examples set no seed.
- Chapter 3 keeps seasonal naive and these helpers and runs them over a year of Sunday cutoffs, with one scale per client taken from every reading before the first cutoff, the rule this chapter sets out. One week cannot give that test.
