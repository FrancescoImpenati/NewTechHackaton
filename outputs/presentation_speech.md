# EWS Presentation Speech (≈ 9 min, ~1,450 words)

> Read with one short pause between slides. We means the project team.

---

## Slide 1 — Cover (≈ 25 sec)

Good morning, and thank you for being here. The work we'll walk you through this
morning is an **Early Warning System for Risk-Off Detection** — a quantitative
strategy that runs 1.5× equity in calm markets and routes into a
context-appropriate safe haven the moment financial stress fires. We built it
on the weekly Bloomberg dataset you provided, from January 2000 to April 2021,
about eleven hundred observations. Let me show you what we found.

## Slide 2 — The problem (≈ 50 sec)

Markets crash. That's the unromantic premise of the whole project. And the
crashes are not evenly spread — they cluster: the dot-com unwind, the global
financial crisis, the euro sovereign crisis, COVID. Buy-and-hold pays for every
single one of those drawdowns in full. So **the value of skipping just a handful
of weeks is enormous**: a couple of avoided crashes lift Sharpe and Calmar by
multiples, not percentages. We took that as our business case. Our framing is
deliberately one of **nowcasting, not forecasting**. We are not trying to predict
that a crisis will happen six months ahead. We're trying to recognize that
financial stress has *just been born* and to react fast enough that the next two
or three weeks of rotation matter. That distinction governs every modeling
choice that follows.

## Slide 3 — Dataset & target (≈ 40 sec)

The data is what you gave us: weekly Bloomberg, 43 raw features across equities,
FX, gold, oil, sovereign yields and credit / MBS total-return indices, plus a
binary target Y where one means risk-off. The label is **imbalanced — about
twenty-one percent positive overall** — and the positives are heavily clustered
in 2000–2002, 2008–2009, 2010–2012 and 2020. Crucially, **2013, 2017 and 2019
have zero positives**, and 2014–2018 are sparse. That single fact will shape our
cross-validation design.

## Slide 4 — Pipeline (≈ 35 sec)

Our pipeline is a sequence of clean stages, all reusable as modules under
`src/`. Raw data goes through stationarity transforms applied by family, then
through engineered macro spreads, then through a collinearity cleanup. Splits
are produced once with strict purging. Four novelty detectors are fit on
**normal-only** weeks, combined into a percentile ensemble. Three macro domain
sub-scores feed a routing engine. Everything ends in a backtest against five
benchmarks with a full risk-adjusted metric suite. The motto: **preprocessing is
part of the model**.

## Slide 5 — Feature engineering (≈ 55 sec)

We treated stationarity seriously. Prices — equities, FX, gold, oil, bond total
returns — become **log-returns**. Yields and rates become **first differences**.
Things that are already stationary like VIX or the Economic Surprise Index stay
as levels. We add eight macro spreads — term structure, BTP-Bund, US-DE 10Y,
HY-IG, EM — each as level *and* four-week change, plus four standalone signals:
the variance risk premium, the gold/oil ratio, an equity-bond rotation signal
and JPY strength. One feature we want to flag: **the 13-week equity-bond
correlation**. It discriminates **liquidity-driven** crises, where stocks and
bonds fall together and the correlation flips positive, from **inflation-driven**
ones — a regime axis the price-and-yield features alone miss. After dropping
collinear twins we land on 56 features going into the models.

## Slide 6 — Walk-forward CV (≈ 60 sec)

This is the slide we want to dwell on, because we believe **it is the single
thing that separates a credible backtest from an illusory one**. We use a purged
expanding walk-forward with five folds. Train always starts at the dataset
start. Between each train and its validation window we insert a **four-week
embargo gap**, exactly the lookback horizon of our engineered features. That
embargo kills the most dangerous form of leakage. The test set — 2019 through
April 2021, including COVID — is **sealed**: untouched until the very end. Our
scalers are fit **per fold, on that fold's train only**, never on the full
dataset. The folds have wildly different positive counts — 53, 64, 2, 10, 6 —
so we aggregate cross-fold metrics **weighted by the number of positives**.
Folds one and two carry the signal; folds three through five are noisy and we
report them honestly.

## Slide 7 — The four models (≈ 45 sec)

We trained four anomaly detectors on the Y-equals-zero normals only. **MVG with
Ledoit-Wolf shrinkage** is our interpretable parametric baseline — a single
Mahalanobis distance, but with a covariance that is actually invertible despite
56 features and only a few hundred normals. **One-Class SVM** with an RBF
kernel gives a flexible non-linear boundary. The **Autoencoder** — 56 to 24 to
12 to 6, then mirrored back — captures non-linear feature interactions, with
dropout for regularization and internal early stopping on a temporal sub-split
of the train, never on the validation fold. **Isolation Forest** is our robust,
scale-invariant counterpart. All four pass their sanity bars. Test AUC-ROC
ranges from 0.81 to 0.88; AE has the best AUC-PR, MVG the best F1.

## Slide 8 — Ensemble (≈ 40 sec)

We then combined them. Raw scores live on incomparable scales, so we map each
model's score to its empirical percentile against the train-normals
distribution. Three voting modes — hard majority, soft mean, soft median —
tuned on walk-forward. The **soft median wins** test F1, beating the best single
model, 0.647 to 0.635. Small win. We then ran an **error-correlation analysis**
on the rich folds and found mean pairwise correlation around 0.51. The honest
finding: because all four models live in the *same feature space*, they tend to
fire on the same weeks, so the ensemble adds **stability and robustness more
than new information**. We list that explicitly as a limitation.

## Slide 9 — Routing (≈ 60 sec)

This is the macroeconomic heart of the project, and we think the most
interesting idea in the deck. **Detecting a crisis is not enough.** You also
have to know *where to hide*, and the right hiding place depends on the
*nature* of the crisis. In a funding-driven dollar squeeze, the world wants USD
cash. In an inflation- or real-yield-driven shock, gold rallies. In a moderate
stress with a positively-sloped curve and shallow drawdown, MBS pays carry. So
we built three domain sub-scores — USD, Gold, MBS — each as a calibrated mean
of signed z-scores from a tailored set of triggers, all calibrated **only on
the development set**. The MBS score is binary and rule-based, designed to
switch on in mild stress and switch off in acute crises. Notice that
`dxy_chg4w` deliberately enters both USD and Gold with *opposite* signs — same
variable, opposite economic reading.

## Slide 10 — Decision matrix (≈ 30 sec)

The router itself is a few lines of code. If the ensemble says risk-on, we run
1.5× equity. If it says risk-off, we pick the haven: USD cash if the USD score
exceeds its threshold and the dollar is appreciating; otherwise gold if the
gold score exceeds its threshold; otherwise MBS if the rule says active. If
nothing clears, we fall back to USD cash as the safe default.

## Slide 11 — Threshold tuning (≈ 30 sec)

The two thresholds — USD and gold — are tuned by **Calmar grid search** on
walk-forward fold validations, aggregated by **duration-weighted median**. The
winner is `usd = 1.5, oro = 0.5`. We deliberately frame the threshold choice as
a *business decision*, not a statistical artifact — exactly as you suggested in
the coaching session.

## Slide 12 — Backtest (≈ 35 sec)

We then ran the EWS strategy against five benchmarks on the sealed 2019–2021
test holdout: MSCI buy-and-hold, static 60/40, a Butterfly Balanced portfolio,
Permanent Portfolio, and Risk Parity with a 52-week inverse-vol scheme. The
equity curve tells the story: **EWS in black ends well above all five**,
finishing around 1.85× starting capital. The visible wedge during the COVID
crash is the routing engine rotating into gold and then back to leveraged
equity for the recovery.

## Slide 13 — Metrics & crisis behavior (≈ 55 sec)

On the headline numbers, **EWS wins Sharpe (1.45), Sortino (2.22), Calmar
(1.39) and CAGR (31 %) against every benchmark**. Max drawdown 22 % — better
than MSCI's 28 % and 60/40's 20 %. During the COVID crash, it draws down 21 %
against MSCI's 27 % and recovers in roughly 22 weeks against MSCI's 35. One
honest nuance: the always-defensive blends — Butterfly, Permanent, Risk Parity
— have **smaller absolute drawdowns**, around 10 to 12 percent, simply because
they're never fully invested in equity. We do not beat them on raw drawdown.
But we win on every risk-adjusted measure and on CAGR by a wide margin. We
want to be precise: **what we built is a Quant Strategy, optimized for
risk-adjusted return — it is not a Risk Management overlay whose only goal is
to minimize drawdown**.

## Slide 14 — Limitations (≈ 50 sec)

We want to state the limitations before you ask. Labels are sparse after 2013;
n_pos weighting mitigates this but doesn't erase the noise. Training is
dominated by liquidity and credit crises — the 2022-style inflation shock is
**not in the data**, so that regime is genuinely untested. The four models
share a feature space, so the ensemble boost is modest. The MSCI World series
is not in the dataset; we used an equal-weight DM composite as a proxy. The
credit spreads we built are log-ratios of total-return indices, not OAS in
basis points. Transaction costs are static estimates. And we deliberately
imposed a one-week execution lag, which is realistic but costs us in the very
first week of a fast crash.

## Slide 15 — Future work & close (≈ 35 sec)

Where this goes next. An HMM with two or three explicit regimes. Ensembles
built over **disjoint** feature sets — rates-only, equity-only, FX-only — to
genuinely cut error correlation. Daily signals to anticipate fast crises by a
few days. Multi-asset extensions to REITs, EM bonds and commodity baskets. And
of course post-2021 stress tests on the inflation shock and geopolitical
ruptures the data doesn't yet contain.

We'll close with one line that guided us throughout. Feynman wrote, *"The first
principle is that you must not fool yourself — and you are the easiest person
to fool."* That is exactly what walk-forward, embargo, leakage-free scalers,
sealed holdouts, n_pos-weighted aggregation and honest limitation slides are
for: not to convince *you* of our results, but to make sure *we* haven't
convinced ourselves of something that isn't there.

Thank you. We'd love to take your questions.
