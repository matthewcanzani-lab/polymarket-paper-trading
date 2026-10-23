"""Numbers behind the dashboard. Pure functions over the ledger, the latest
scorer snapshot, and saved model versions — no web code here.

Two kinds of results are kept strictly apart:
  * FORWARD — from the paper ledger: predictions logged before markets resolved.
  * BACKTEST — from models/<version>/meta.json: a historical simulation.
"""
import json

import numpy as np

from pmodel import config
from pmodel.trading import ledger as ledger_mod

TARGET_SETTLED_BETS = 150
CALIBRATION_TOLERANCE_PTS = 5.0
MIN_BETS_PER_CAL_BAND = 10
MIN_BETS_PER_CATEGORY = 10


def load_scores(path=None):
    path = path or config.LATEST_SCORES_PATH
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, ValueError):
        return None


def load_weather_scores(path=None):
    return load_scores(path or config.LATEST_WEATHER_SCORES_PATH)


def age_hours(iso, now=None):
    """Hours since an ISO-8601 UTC timestamp (None if missing)."""
    if not iso:
        return None
    from datetime import datetime, timezone
    t = datetime.strptime(iso, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    return ((now or datetime.now(timezone.utc)) - t).total_seconds() / 3600


def eastern(iso):
    """'Oct 23, 2026, 8:55 AM EDT' for an ISO-8601 UTC timestamp (US Eastern time)."""
    if not iso:
        return None
    from datetime import datetime, timezone
    from zoneinfo import ZoneInfo
    t = datetime.strptime(iso, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    et = t.astimezone(ZoneInfo("America/New_York"))
    hour = et.strftime("%I").lstrip("0")
    return f"{et:%b} {et.day}, {et.year}, {hour}:{et:%M %p} {et.tzname()}"


def latest_runs(conn):
    """Most recent run of each strategy, newest first."""
    seen, out = set(), []
    for r in conn.execute("SELECT * FROM runs ORDER BY seq DESC"):
        name = strategy_of(r["model_version"])
        if name not in seen:
            seen.add(name)
            out.append(dict(r, strategy=name))
    return out


def _mid(m):
    if m.get("best_bid") is not None and m.get("best_ask") is not None:
        return (m["best_bid"] + m["best_ask"]) / 2
    return m["last_price"]


def live_markets(scores, limit=300):
    rows = [dict(m, mid=_mid(m), edge=m["model_prob_yes"] - _mid(m))
            for m in (scores or {}).get("markets", [])]
    rows.sort(key=lambda r: -r["volume_usd"])
    return rows[:limit]


def edges(scores, limit=100):
    rows = [dict(m, mid=_mid(m), edge=m["model_prob_yes"] - _mid(m))
            for m in (scores or {}).get("markets", [])]
    rows.sort(key=lambda r: -abs(r["edge"]))
    return rows[:limit]


def strategy_of(model_version: str) -> str:
    return "Weather (forecast blend)" if (model_version or "").startswith("weather") else \
        "General (price/volume model)"


def open_by_strategy(conn):
    out = {}
    for b in ledger_mod.open_bets(conn):
        name = strategy_of(b["model_version"])
        o = out.setdefault(name, {"strategy": name, "bets": 0, "stake": 0.0})
        o["bets"] += 1
        o["stake"] += b["stake"]
    return sorted(out.values(), key=lambda o: -o["bets"])


def bets(conn):
    """Every bet with its settlement (if any), newest first."""
    return [dict(r) for r in conn.execute(
        """SELECT b.*, s.settled_at, s.yes_payout, s.payout, s.pnl
           FROM bets b LEFT JOIN settlements s USING (bet_id) ORDER BY b.seq DESC""")]


def pnl_curve(conn):
    """[(settled_at, cumulative P&L)] in settlement order."""
    total, out = 0.0, []
    for r in conn.execute("SELECT settled_at, pnl FROM settlements ORDER BY seq"):
        total += r["pnl"]
        out.append((r["settled_at"], total))
    return out


def forward_stats(conn):
    settled = [b for b in bets(conn) if b["settled_at"] is not None]
    n = len(settled)
    stats = {"settled": n, "open": len(ledger_mod.open_bets(conn)),
             "money": ledger_mod.bankroll(conn)}
    if not n:
        return dict(stats, brier=None, win_rate=None, return_on_stake=None, total_pnl=0.0,
                    calibration=[], max_cal_gap=None, by_category=[], by_strategy=[])
    # Probability the model gave to the side we bought, and whether that side won.
    p_side = np.array([b["model_prob_yes"] if b["side"] == "YES" else 1 - b["model_prob_yes"]
                       for b in settled])
    side_paid = np.array([b["yes_payout"] if b["side"] == "YES" else 1 - b["yes_payout"]
                          for b in settled])
    p_yes = np.array([b["model_prob_yes"] for b in settled])
    yes = np.array([b["yes_payout"] for b in settled])
    stake = sum(b["stake"] for b in settled)
    pnl = sum(b["pnl"] for b in settled)

    cal = []
    edges_ = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(p_side, edges_) - 1, 0, 9)
    for k in range(10):
        m = idx == k
        if m.any():
            cal.append({"band": f"{edges_[k]*100:.0f}-{edges_[k+1]*100:.0f}%",
                        "forecast": float(p_side[m].mean()), "actual": float(side_paid[m].mean()),
                        "n": int(m.sum())})
    big = [c for c in cal if c["n"] >= MIN_BETS_PER_CAL_BAND]
    max_gap = max((abs(c["forecast"] - c["actual"]) * 100 for c in big), default=None)

    cats = {}
    for b in settled:
        c = cats.setdefault(b["category"] or "unknown", {"category": b["category"] or "unknown",
                                                          "bets": 0, "stake": 0.0, "pnl": 0.0})
        c["bets"] += 1
        c["stake"] += b["stake"]
        c["pnl"] += b["pnl"]
    by_cat = sorted(cats.values(), key=lambda c: -c["bets"])
    strat = {}
    for b in settled:
        name = strategy_of(b["model_version"])
        c = strat.setdefault(name, {"strategy": name, "bets": 0, "stake": 0.0, "pnl": 0.0})
        c["bets"] += 1
        c["stake"] += b["stake"]
        c["pnl"] += b["pnl"]
    return dict(stats,
                brier=float(np.mean((p_yes - yes) ** 2)),
                win_rate=float(np.mean([b["pnl"] > 0 for b in settled])),
                return_on_stake=pnl / stake if stake else None,
                total_pnl=pnl, calibration=cal, max_cal_gap=max_gap, by_category=by_cat,
                by_strategy=sorted(strat.values(), key=lambda c: -c["bets"]))


def success_criteria(fs):
    """Progress toward the goals set at the start of the project."""
    n = fs["settled"]
    profitable_cats = [c for c in fs["by_category"]
                       if c["bets"] >= MIN_BETS_PER_CATEGORY and c["pnl"] > 0]
    gap = fs["max_cal_gap"]
    return [
        {"name": f"{TARGET_SETTLED_BETS}+ settled bets",
         "value": f"{n} / {TARGET_SETTLED_BETS}", "progress": min(n / TARGET_SETTLED_BETS, 1.0),
         "met": n >= TARGET_SETTLED_BETS},
        {"name": "Profitable after slippage",
         "value": f"${fs['total_pnl']:+,.2f}" if n else "no settled bets yet",
         "progress": None, "met": n > 0 and fs["total_pnl"] > 0},
        {"name": f"Calibration within ±{CALIBRATION_TOLERANCE_PTS:.0f} points",
         "value": (f"worst band off by {gap:.1f} pts" if gap is not None
                   else f"needs {MIN_BETS_PER_CAL_BAND}+ settled bets in a probability band"),
         "progress": None, "met": gap is not None and gap <= CALIBRATION_TOLERANCE_PTS},
        {"name": "Profitable in more than one category",
         "value": (f"{len(profitable_cats)} profitable categor"
                   f"{'y' if len(profitable_cats) == 1 else 'ies'} "
                   f"(with {MIN_BETS_PER_CATEGORY}+ bets)"),
         "progress": None, "met": len(profitable_cats) >= 2},
    ]


def backtests(models_dir):
    """Saved model versions, newest first, with their BACKTEST metrics."""
    out = []
    if not models_dir.exists():
        return out
    from pmodel.model.registry import versions
    for path in reversed(versions(models_dir)):
        try:
            meta = json.loads((path / "meta.json").read_text())
        except (FileNotFoundError, ValueError):
            continue
        out.append({"version": meta["version"], "path": path, "meta": meta,
                    "superseded": meta.get("superseded")})
    return out


EARLY_SAMPLE = 50   # below this many settled markets, results are flagged as early
MEANINGFUL_GAIN = 0.02  # model must beat the market's Brier score by 2%+ to claim it is better


def results_summary(conn):
    """Headline numbers for the Results page. FORWARD results only: every figure
    comes from bets logged before their market resolved. Nothing is rounded up
    and nothing is filtered: all settled bets count."""
    logged = conn.execute("SELECT COUNT(*) FROM bets").fetchone()[0]
    rows = conn.execute("""SELECT b.model_prob_yes, b.market_mid, b.stake, s.yes_payout, s.pnl
                           FROM bets b JOIN settlements s USING (bet_id)""").fetchall()
    settled = len(rows)
    out = {"logged": logged, "settled": settled, "early": settled < EARLY_SAMPLE,
           "early_threshold": EARLY_SAMPLE, "brier_model": None, "brier_market": None,
           "paper_return": None, "staked": 0.0, "pnl": 0.0, "verdict": None}
    if settled:
        y = np.array([r["yes_payout"] for r in rows])
        out["brier_model"] = float(np.mean((np.array([r["model_prob_yes"] for r in rows]) - y) ** 2))
        out["brier_market"] = float(np.mean((np.array([r["market_mid"] for r in rows]) - y) ** 2))
        out["staked"] = float(sum(r["stake"] for r in rows))
        out["pnl"] = float(sum(r["pnl"] for r in rows))
        out["paper_return"] = out["pnl"] / out["staked"] if out["staked"] else None
        bm, bk = out["brier_model"], out["brier_market"]
        if bm >= bk:
            out["verdict"] = ("So far my model is not beating the market: the market's own prices "
                              "have been at least as accurate as mine.")
        elif bm > bk * (1 - MEANINGFUL_GAIN):
            out["verdict"] = ("So far my model has been roughly as accurate as the market's own "
                              "prices. The difference is too small to call an edge.")
        else:
            out["verdict"] = ("So far my model's probabilities have been more accurate than the "
                              "market's own prices on the markets I bet on.")
    return out
