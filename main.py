"""GTA VI trophy rarity forecaster for make.fun gta6-trophy-rarity-forecast-01.

Reads one case file (the house's recorded reads of Exophase's GTA VI PS5
trophy page) and writes, for each trophy, five quantiles of its earn rate
(percent of tracked players) at the case's target.

Approach, in short:
  * Collapse the read history to distinct page states (Exophase refreshes
    discretely, so consecutive identical reads carry no new information).
  * Forecast each trophy's median earn rate as the equal-weight mean, on
    the logit scale, of trend fits against log-time since launch: linear
    fits over the last 12/24/48 hours and a capped quadratic fit (cohort
    progression is convex while a trophy ramps).
  * Calibrate each trophy's logit spread from rolling-origin backtests of
    that same ensemble inside the case's own history, scaled from the
    backtest horizon to the forecast horizon and blended toward a fixed
    spread, so volatile trophies get wider bands than stable ones.
  * Enforce structural facts: story-completion trophies are ordered (you
    cannot complete Chapter 2 before Chapter 1), and the platinum rate
    cannot exceed any other trophy's rate.

Validated against a cohort-progression launch simulator, where this
ensemble beats the house persistence and curve baselines. Standard
library only; runs offline, one case per invocation directory.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
LAUNCH = datetime(2026, 11, 19, 0, 0, tzinfo=timezone.utc)

# Story-completion chain by trophy id (Exophase list order): Gone Fishing,
# Chapter 1..4, final mission. Earn rates must be non-increasing along it.
STORY_CHAIN = [2, 3, 4, 5, 6, 7]
PLATINUM_ID = 1
FULL_COMPLETION_ID = 14  # "Attain 100% Completion"
FINAL_MISSION_ID = 7


def parse_ts(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def clamp_rate(pct):
    return min(99.99, max(0.01, pct))


def logit(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(value):
    if value >= 0:
        return 100.0 / (1.0 + math.exp(-value))
    e = math.exp(value)
    return 100.0 * e / (1.0 + e)


def log_time(ts):
    hours = (ts - LAUNCH).total_seconds() / 3600.0
    return math.log1p(max(hours, 0.0))


def round4(x):
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def wls_line(xs, ys, ws):
    """Weighted least squares y ~ a + b*x. Returns (a, b) or None."""
    sw = sum(ws)
    if sw <= 0 or len(xs) < 2:
        return None
    mx = sum(w * x for x, w in zip(xs, ws)) / sw
    my = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws))
    if sxx < 1e-12:
        return (my, 0.0)
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws))
    b = sxy / sxx
    return (my - b * mx, b)


def solve3(a, b):
    """Solve a 3x3 linear system by Gaussian elimination. None if singular."""
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(3):
        piv = max(range(col, 3), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-12:
            return None
        m[col], m[piv] = m[piv], m[col]
        for r in range(3):
            if r != col:
                f = m[r][col] / m[col][col]
                for c in range(col, 4):
                    m[r][c] -= f * m[col][c]
    return [m[i][3] / m[i][i] for i in range(3)]


class State:
    __slots__ = ("t", "x", "players", "rates")

    def __init__(self, t, players, rates):
        self.t = t
        self.x = log_time(t)
        self.players = players
        self.rates = rates


def distinct_states(case):
    states = []
    seen = None
    for read in case["history"]:
        key = (read.get("players_tracked"), tuple(read.get("rates") or ()))
        if key == seen:
            continue
        seen = key
        states.append(State(parse_ts(read["at"]), read.get("players_tracked") or 0, read.get("rates") or []))
    return states


def trophy_series(states, idx):
    """(x, rate, players, t) points where this trophy has a rate."""
    out = []
    for s in states:
        if idx < len(s.rates) and s.rates[idx] is not None:
            out.append((s.x, float(s.rates[idx]), s.players, s.t))
    return out


def model_curve(series, target_x, window_hours):
    pts = series
    if window_hours is not None and series:
        last_t = series[-1][3]
        win = [p for p in series if (last_t - p[3]).total_seconds() <= window_hours * 3600]
        if len(win) >= 3:
            pts = win
    if len(pts) < 3:
        return None
    fit = wls_line([p[0] for p in pts], [logit(p[1]) for p in pts], [1.0] * len(pts))
    if not fit:
        return None
    return inv_logit(fit[0] + fit[1] * target_x)


def model_quad(series, target_x, window_hours=48, cap_quad=1.5):
    """Quadratic in log-time over the window, quadratic term capped.

    Cohort progression is convex in log-time while a trophy is ramping;
    the cap keeps the extrapolation from running away.
    """
    if len(series) < 5:
        return None
    last_t = series[-1][3]
    win = [p for p in series if (last_t - p[3]).total_seconds() <= window_hours * 3600]
    pts = win if len(win) >= 5 else series
    if len(pts) < 5:
        return None
    x0 = pts[-1][0]
    xs = [p[0] - x0 for p in pts]
    ys = [logit(p[1]) for p in pts]
    ata = [[0.0] * 3 for _ in range(3)]
    atb = [0.0] * 3
    for x, y in zip(xs, ys):
        row = [1.0, x, x * x]
        for i in range(3):
            atb[i] += row[i] * y
            for j in range(3):
                ata[i][j] += row[i] * row[j]
    for i in range(3):
        ata[i][i] += 1e-8
    sol = solve3(ata, atb)
    if not sol:
        return None
    a, b, c = sol
    dx = target_x - x0
    quad_term = max(-cap_quad, min(cap_quad, c * dx * dx))
    return inv_logit(a + b * dx + quad_term)


def ensemble_median_logit(series, target_x):
    """Equal-weight mean of the trend fits, in logit space. None if no fit."""
    preds = []
    for window in (12, 24, 48):
        pred = model_curve(series, target_x, window)
        if pred is not None and math.isfinite(pred):
            preds.append(logit(pred))
    pred = model_quad(series, target_x)
    if pred is not None and math.isfinite(pred):
        preds.append(logit(pred))
    if not preds:
        return None
    return sum(preds) / len(preds)


def backtest_residuals(states, n_trophies):
    """Absolute logit errors of the ensemble at 12h..54h horizons.

    Returns (residuals per trophy, mean horizon in days). Pair count is
    capped so the run stays well inside its time budget.
    """
    residuals = [[] for _ in range(n_trophies)]
    horizons = []
    n = len(states)
    if n < 6:
        return residuals, horizons
    pairs = []
    for j in range(3, n - 1):
        for k in range(j + 1, n):
            gap_h = (states[k].t - states[j].t).total_seconds() / 3600.0
            if gap_h < 12:
                continue
            if gap_h > 54:
                break
            pairs.append((j, k))
    if len(pairs) > 160:
        step = len(pairs) / 160.0
        pairs = [pairs[int(i * step)] for i in range(160)]
    for j, k in pairs:
        hist = states[: j + 1]
        target = states[k]
        horizons.append((target.t - states[j].t).total_seconds() / 86400.0)
        for idx in range(n_trophies):
            if idx >= len(target.rates) or target.rates[idx] is None:
                continue
            series = trophy_series(hist, idx)
            if len(series) < 3:
                continue
            med = ensemble_median_logit(series, target.x)
            if med is None:
                continue
            residuals[idx].append(abs(med - logit(float(target.rates[idx]))))
    return residuals, horizons


def isotonic_decreasing(values):
    """Pool-adjacent-violators for a non-increasing sequence."""
    vals = list(values)
    weights = [1.0] * len(vals)
    i = 0
    while i < len(vals) - 1:
        if vals[i] < vals[i + 1]:
            total_w = weights[i] + weights[i + 1]
            merged = (vals[i] * weights[i] + vals[i + 1] * weights[i + 1]) / total_w
            vals[i:i + 2] = [merged]
            weights[i:i + 2] = [total_w]
            if i > 0:
                i -= 1
        else:
            i += 1
    out = []
    for v, w in zip(vals, weights):
        out.extend([v] * int(w))
    return out


def forecast_case(case):
    trophies = case["trophies"]
    n_trophies = len(trophies)
    target_ts = parse_ts(case["target"])
    target_x = log_time(target_ts)
    horizon_days = case["horizon_minutes"] / 1440.0

    states = distinct_states(case)
    if not states:
        raise ValueError("case has no history")
    last_rates = []
    for idx in range(n_trophies):
        rate = None
        for s in reversed(states):
            if idx < len(s.rates) and s.rates[idx] is not None:
                rate = float(s.rates[idx])
                break
        last_rates.append(rate if rate is not None else 0.0)

    residuals, horizons = backtest_residuals(states, n_trophies)
    mean_bt_horizon = (sum(horizons) / len(horizons)) if horizons else max(horizon_days, 0.25)
    pooled = [e for per in residuals for e in per]
    global_mae = (sum(pooled) / len(pooled)) if pooled else 0.3
    horizon_scale = math.sqrt(max(horizon_days, 0.05) / max(mean_bt_horizon, 0.05))

    medians = []
    sigmas = []
    for idx in range(n_trophies):
        series = trophy_series(states, idx)
        med = ensemble_median_logit(series, target_x) if series else None
        if med is None:
            med = logit(last_rates[idx])
            sigma = 0.5 * math.sqrt(horizon_days) if horizons else 0.55
        else:
            per = residuals[idx]
            mae = (sum(per) + 8 * global_mae) / (len(per) + 8) if per else global_mae
            scaled = 1.25 * mae * horizon_scale
            sigma = 0.6 * scaled + 0.4 * 0.55
            sigma = min(1.0, max(0.35, sigma))
        if last_rates[idx] >= 99.99:
            sigma = min(sigma, 0.2)
        medians.append(med)
        sigmas.append(sigma)

    # structural constraints on medians (logit space)
    by_id = {t["id"]: i for i, t in enumerate(trophies)}
    chain = [by_id[i] for i in STORY_CHAIN if i in by_id]
    if len(chain) >= 2:
        adjusted = isotonic_decreasing([medians[i] for i in chain])
        for pos, idx in enumerate(chain):
            medians[idx] = adjusted[pos]
    if PLATINUM_ID in by_id:
        pi = by_id[PLATINUM_ID]
        medians[pi] = min(medians[pi], min(medians))
    if FULL_COMPLETION_ID in by_id and FINAL_MISSION_ID in by_id and PLATINUM_ID in by_id:
        fi = by_id[FULL_COMPLETION_ID]
        medians[fi] = min(max(medians[fi], medians[by_id[PLATINUM_ID]]), medians[by_id[FINAL_MISSION_ID]])

    rows = []
    for idx, trophy in enumerate(trophies):
        qs = [round4(inv_logit(medians[idx] + z * sigmas[idx])) for z in Z]
        for i in range(1, 5):
            if qs[i] < qs[i - 1]:
                qs[i] = qs[i - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        if case.get("levels") != LEVELS:
            raise ValueError(f"{path.name} asks for levels {case.get('levels')}, not {LEVELS}")
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
