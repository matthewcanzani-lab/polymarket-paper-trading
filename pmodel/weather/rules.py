"""Per-version betting rules for the weather strategy, and the safety checks
that go with them.

Each weather model version file can carry a "rules" block. Versions saved
before 2026-11-05 have none and used OLD_RULES. Keeping the rules inside the
version file means every bet in the ledger can be traced to the exact rules
that produced it (bets record their model_version).

Also here:
  * calibration — pull the model's probability toward the market price by a
    factor fitted on past paper bets (k=1: trust the model, k=0: trust the market);
  * city-day exposure — total stake already on one city and date, across
    high/low events and across earlier runs;
  * stop-loss — halt a version's new bets once its settled losses reach a set
    share of the bankroll it started with, until a review is recorded.
"""
import json

import numpy as np
from scipy.optimize import minimize_scalar

from pmodel import config
from pmodel.sources import polymarket_weather as pw

# What versions without a "rules" block did.
OLD_RULES = {"edge_threshold": 0.05, "max_spread": None, "fill_at_best_price_only": False,
             "city_day_cap_fraction": config.WEATHER_MAX_EVENT_FRACTION,
             "stop_loss_fraction": None}

REVIEWS_PATH = config.PROJECT_ROOT / "models" / "weather" / "stop_loss_reviews.json"


def rules_for(params: dict) -> dict:
    return {**OLD_RULES, **params.get("rules", {})}


# ---------------------------------------------------------------- calibration
def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def shrink_toward_market(p, mid, k):
    """Keep a share k of the model's disagreement with the market (in log-odds)."""
    x = _logit(mid) + k * (_logit(p) - _logit(mid))
    return 1 / (1 + np.exp(-x))


def calibrate(params: dict, p: float, mid: float) -> float:
    cal = params.get("calibration")
    if not cal:
        return p
    return float(shrink_toward_market(p, mid, cal["k"]))


def fit_calibration(conn, before: str) -> dict:
    """Fit k on weather paper bets settled before `before` (YYYY-MM-DD).

    k is limited to [0, 1]: below 0 would mean betting AGAINST the model's view,
    which a small noisy sample can suggest but nothing supports.
    """
    rows = conn.execute(
        """SELECT b.model_prob_yes, b.market_mid, s.yes_payout FROM bets b
           JOIN settlements s USING (bet_id)
           WHERE b.model_version LIKE 'weather%' AND s.settled_at < ? ORDER BY b.seq""",
        (before,)).fetchall()
    p = np.array([r[0] for r in rows])
    mid = np.array([r[1] for r in rows])
    y = np.array([r[2] for r in rows], dtype=float)

    def loss(k):
        f = np.clip(shrink_toward_market(p, mid, k), 1e-6, 1 - 1e-6)
        return -np.mean(y * np.log(f) + (1 - y) * np.log(1 - f))

    k = float(minimize_scalar(loss, bounds=(0.0, 1.0), method="bounded").x)
    k = 0.0 if k < 1e-3 else round(k, 3)
    brier = lambda f: float(np.mean((f - y) ** 2))
    return {"method": "shrink_toward_market", "k": k, "fit_on_bets": len(y),
            "fit_on": f"weather paper bets settled before {before}",
            "brier_before": brier(p), "brier_after": brier(shrink_toward_market(p, mid, k)),
            "brier_market": brier(mid)}


# ---------------------------------------------------------------- city-day exposure
def city_day(question: str, end_date: str):
    parsed = pw.parse_question(question or "", int((end_date or "0000")[:4]) or 2026)
    return (parsed["city"], parsed["day"]) if parsed else None


def city_day_stakes(conn) -> dict:
    """{(city, day): total stake} over every weather bet ever logged (any version)."""
    out = {}
    for r in conn.execute("SELECT question, end_date, stake FROM bets WHERE category = 'Weather'"):
        key = city_day(r["question"], r["end_date"])
        if key:
            out[key] = out.get(key, 0.0) + r["stake"]
    return out


# ---------------------------------------------------------------- stop-loss
def _last_review(version: str, path) -> str | None:
    try:
        reviews = json.loads(path.read_text())
    except (FileNotFoundError, ValueError):
        return None
    times = [r["reviewed_at"] for r in reviews if r.get("version") == version]
    return max(times) if times else None


def stop_loss_status(conn, params: dict, reviews_path=None) -> dict:
    """Is this version halted? Counts settled P&L of its own bets placed after
    its last recorded review (if any)."""
    frac = rules_for(params)["stop_loss_fraction"]
    version = params["version"]
    if not frac:
        return {"enabled": False, "halted": False, "version": version}
    start = params["starting_bankroll"]
    since = _last_review(version, reviews_path or REVIEWS_PATH)
    pnl = conn.execute(
        """SELECT COALESCE(SUM(s.pnl), 0) FROM bets b JOIN settlements s USING (bet_id)
           WHERE b.model_version = ? AND b.placed_at >= ?""", (version, since or "")).fetchone()[0]
    limit = -frac * start
    return {"enabled": True, "version": version, "pnl": pnl, "limit": limit,
            "fraction": frac, "starting_bankroll": start, "reviewed_at": since,
            "halted": pnl <= limit + 1e-9}



def calibration_chart(conn, before: str, k: float, path):
    """Before/after reliability chart of the calibration, on the bets it was fitted on."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = conn.execute(
        """SELECT b.side, b.model_prob_yes, b.market_mid, s.yes_payout FROM bets b
           JOIN settlements s USING (bet_id)
           WHERE b.model_version LIKE 'weather%' AND s.settled_at < ?""", (before,)).fetchall()
    side = lambda p, r: p if r["side"] == "YES" else 1 - p
    won = np.array([side(r["yes_payout"], r) for r in rows])
    before_p = np.array([side(r["model_prob_yes"], r) for r in rows])
    after_p = np.array([side(float(shrink_toward_market(r["model_prob_yes"], r["market_mid"], k)), r)
                        for r in rows])
    edges = np.linspace(0, 1, 6)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
    for ax, p, title in ((axes[0], before_p, "Before: model as it was"),
                         (axes[1], after_p, f"After: calibrated toward market (k = {k:g})")):
        idx = np.clip(np.digitize(p, edges) - 1, 0, len(edges) - 2)
        xs, ys, ns = [], [], []
        for b in range(len(edges) - 1):
            m = idx == b
            if m.sum():
                xs.append(p[m].mean()); ys.append(won[m].mean()); ns.append(int(m.sum()))
        ax.plot([0, 1], [0, 1], ls="--", color="#888", lw=1)
        ax.scatter(xs, ys, s=[30 + 6 * n for n in ns], color="#1a5fb4", zorder=3)
        for x, y_, n in zip(xs, ys, ns):
            ax.annotate(f"n={n}", (x, y_), textcoords="offset points", xytext=(8, -12), fontsize=9)
        brier = np.mean((p - won) ** 2)
        ax.set_title(f"{title}\nBrier {brier:.3f}", fontsize=11)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("probability given to the side bought")
        ax.grid(alpha=.3)
    axes[0].set_ylabel("share of those bets that won")
    fig.suptitle(f"Weather paper bets settled before {before} (n = {len(rows)}); "
                 "dots on the dashed line = honest confidence", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
