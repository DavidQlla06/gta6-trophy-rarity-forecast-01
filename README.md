# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/DavidQlla06/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/DavidQlla06/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A forecaster for make.fun's GTA VI trophy rarity competition: given the
house's recorded reads of Exophase's GTA VI PS5 trophy page, it predicts
each trophy's earn rate at the case's target as five quantiles, scored by
pinball loss on the logit scale.

## How it works

- Collapses the read history to distinct page states (Exophase refreshes
  discretely; identical consecutive reads carry no new information).
- Forecasts each trophy's median earn rate with an ensemble of trend
  models on the logit scale against log-time since launch: linear fits
  over the last 12/24/48 hours, a capped quadratic fit (cohort
  progression is convex while a trophy ramps), and persistence.
- Weights the ensemble by rolling-origin backtests inside the case's own
  history, emphasizing long horizons, and calibrates each trophy's logit
  spread from those backtest errors, scaled to the forecast horizon.
- Enforces structural facts: story-completion trophies are ordered (you
  cannot complete Chapter 2 before Chapter 1), and the platinum rate
  cannot exceed any other trophy's rate.

Validated against a cohort-progression launch simulator, where it beats
the house persistence and curve baselines.

## Run it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only, no network.

---

<sub>Entered on [make.fun](https://make.fun/?ref=56285tn4&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
