"""Volume distribution report — used to choose MIN_VOLUME_USD.

Trade-off the report is meant to expose:
  * a LOW cutoff keeps more markets (more training data, more bets) but includes
    thin markets whose prices are noisy and whose spreads would eat any edge;
  * a HIGH cutoff gives cleaner prices and cheaper fills but fewer markets, so it
    takes longer to reach 150 settled bets.
"""
import math
import statistics
from datetime import datetime, timedelta, timezone

from pmodel import config

CUTOFFS = [0, 1_000, 5_000, 10_000, 25_000, 50_000, 100_000, 250_000, 1_000_000]
BUCKETS = [(0, 100), (100, 1_000), (1_000, 10_000), (10_000, 100_000),
           (100_000, 1_000_000), (1_000_000, math.inf)]


def _money(x):
    if x == math.inf:
        return "inf"
    for unit, size in (("M", 1e6), ("k", 1e3)):
        if x >= size:
            return f"${x / size:g}{unit}"
    return f"${x:g}"


def _pct(values, q):
    if not values:
        return float("nan")
    values = sorted(values)
    k = (len(values) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def _is_short(row, hours=24):
    return row["hours"] is not None and row["hours"] < hours


def load(conn, platform="polymarket", now_s=None):
    resolved = conn.execute(
        """SELECT m.volume_usd, o.resolved_yes, m.category, m.is_yes_no, m.market_id,
                  (julianday(m.end_date) - julianday(m.created_at)) * 24 AS hours
           FROM markets m JOIN outcomes o USING (platform, market_id)
           WHERE m.platform=? AND o.status='resolved'""", (platform,)).fetchall()
    ambiguous = conn.execute(
        "SELECT COUNT(*) FROM outcomes WHERE platform=? AND status='ambiguous'",
        (platform,)).fetchone()[0]
    # Betting pool: open markets whose scheduled end hasn't passed. Markets past
    # their end date are just waiting for the oracle and can't be fairly bet.
    now_s = now_s or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    active = conn.execute(
        """SELECT volume_usd, best_bid, best_ask, end_date, category,
                  (julianday(end_date) - julianday(created_at)) * 24 AS hours
           FROM markets WHERE platform=? AND closed=0 AND active=1
             AND (end_date IS NULL OR end_date >= ?)""", (platform, now_s)).fetchall()
    return resolved, ambiguous, active


def build_report(conn, platform="polymarket", now=None) -> str:
    now = now or datetime.now(timezone.utc)
    now_s = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    resolved, ambiguous, active = load(conn, platform, now_s)
    past_end = conn.execute(
        """SELECT COUNT(*) FROM markets WHERE platform=? AND closed=0 AND active=1
           AND end_date < ?""", (platform, now_s)).fetchone()[0]
    rv = [r["volume_usd"] for r in resolved]
    av = [a["volume_usd"] for a in active]
    soon = (now + timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%SZ")

    out = []
    w = out.append
    w("VOLUME DISTRIBUTION REPORT")
    w("=" * 78)
    w(f"Resolved markets (clean YES/NO outcome): {len(resolved):,}")
    w(f"Closed but ambiguous/void (excluded):    {ambiguous:,}")
    w(f"Active markets still before end date:    {len(active):,}"
      f"   (zero volume: {sum(v == 0 for v in av):,})")
    w(f"Active but past end date (excluded):     {past_end:,}")
    w(f"Current MIN_VOLUME_USD setting:          {_money(config.MIN_VOLUME_USD)}")
    w("")
    w("Lifetime volume percentiles")
    w(f"  {'':10}{'10%':>10}{'25%':>10}{'50%':>10}{'75%':>10}{'90%':>10}{'99%':>10}")
    for name, vals in (("resolved", rv), ("active", av)):
        cells = "".join(f"{_money(round(_pct(vals, q))) if vals else '-':>10}"
                        for q in (.1, .25, .5, .75, .9, .99))
        w(f"  {name:10}{cells}")
    w("")
    w("How many markets fall in each volume bucket")
    w(f"  {'bucket':22}{'resolved':>10}{'active':>10}")
    for lo, hi in BUCKETS:
        label = f"{_money(lo)} - {_money(hi)}"
        w(f"  {label:22}{sum(lo <= v < hi for v in rv):>10,}{sum(lo <= v < hi for v in av):>10,}")
    w("")
    w("What each candidate cutoff would leave us with")
    w("  resolved = training data | YES% = how often YES won | active = markets we could bet")
    w("  ending<30d = active markets resolving in the next 30 days (how fast bets settle)")
    w("  spread = median bid-ask gap on active markets, in cents (a cost on every fill)")
    w("  cats>=30 = categories with at least 30 resolved markets")
    w("  short = share of kept resolved markets scheduled to last under 24 hours")
    w(f"  {'cutoff':>8}{'resolved':>10}{'kept':>7}{'YES%':>7}{'active':>8}{'ending<30d':>11}"
      f"{'spread':>8}{'cats>=30':>9}{'short':>7}")
    for c in CUTOFFS:
        r = [x for x in resolved if x["volume_usd"] >= c]
        a = [x for x in active if x["volume_usd"] >= c]
        yes = sum(x["resolved_yes"] for x in r) / len(r) * 100 if r else float("nan")
        ending = sum(1 for x in a if x["end_date"] and now_s <= x["end_date"] <= soon)
        spreads = [(x["best_ask"] - x["best_bid"]) * 100 for x in a
                   if x["best_bid"] is not None and x["best_ask"] is not None
                   and x["best_ask"] >= x["best_bid"]]
        spread = f"{statistics.median(spreads):.1f}c" if spreads else "-"
        cats = {}
        for x in r:
            cats[x["category"] or "unknown"] = cats.get(x["category"] or "unknown", 0) + 1
        n_cats = sum(1 for k, v in cats.items() if v >= 30 and k != "unknown")
        kept = f"{len(r) / len(resolved) * 100:.0f}%" if resolved else "-"
        short = f"{sum(_is_short(x) for x in r) / len(r) * 100:.0f}%" if r else "-"
        w(f"  {_money(c):>8}{len(r):>10,}{kept:>7}{yes:>6.0f}%{len(a):>8,}{ending:>11,}"
          f"{spread:>8}{n_cats:>9}{short:>7}")
    w("")
    w("Resolved markets by category, at a few cutoffs (all lengths / excluding <24h markets)")
    cats = sorted({x["category"] or "unknown" for x in resolved},
                  key=lambda k: -sum(1 for x in resolved if (x["category"] or "unknown") == k))
    picks = [1_000, 10_000, 50_000]
    w(f"  {'category':16}" + "".join(f"{_money(c):>18}" for c in picks))
    for cat in cats:
        cells = ""
        for c in picks:
            rows = [x for x in resolved if (x["category"] or "unknown") == cat and x["volume_usd"] >= c]
            long_ = sum(not _is_short(x) for x in rows)
            cells += f"{len(rows):>10,} /{long_:>6,}"
        w(f"  {cat:16}{cells}")
    unknown = sum(1 for x in resolved if not x["category"])
    if resolved:
        w("")
        w(f"Note: {unknown / len(resolved) * 100:.0f}% of resolved markets have no category from the API.")
    return "\n".join(out)


def save_chart(conn, path, platform="polymarket", note=None):
    """Histogram of log10(volume) for resolved vs active markets, with cutoff lines."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker

    resolved, _, active = load(conn, platform)
    panels = [("Resolved markets (training pool)", [r["volume_usd"] for r in resolved]),
              ("Active markets (betting pool)", [a["volume_usd"] for a in active])]
    ink, muted, grid, bar = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"
    edges = [10 ** (x / 4) for x in range(0, 33)]  # $1 .. $100M, 4 bins per decade

    fig, axes = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True, facecolor="#fcfcfb")
    for ax, (title, vals) in zip(axes, panels):
        ax.set_facecolor("#fcfcfb")
        traded = [v for v in vals if v >= 1]
        ax.hist(traded, bins=edges, color=bar, edgecolor="#fcfcfb", linewidth=1.5)
        ax.set_xscale("log")
        ax.set_title(f"{title} — {len(traded):,} markets with volume "
                     f"(+{len(vals) - len(traded):,} with ~$0 not shown)",
                     loc="left", color=ink, fontsize=11)
        ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
        ax.set_ylabel("markets", color=muted)
        ax.grid(axis="y", color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(grid)
        ax.tick_params(colors=muted)
        for c in (1_000, 10_000, 100_000):
            ax.axvline(c, color=muted, linestyle="--", linewidth=1)
            ax.text(c * 1.08, ax.get_ylim()[1] * 0.92, _money(c), color=muted, fontsize=8,
                    bbox=dict(facecolor="#fcfcfb", edgecolor="none", pad=1))
    axes[-1].xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: _money(x)))
    axes[-1].set_xlabel("lifetime volume (USD, log scale)", color=muted)
    if note:
        axes[0].text(0.99, 0.97, note, transform=axes[0].transAxes, ha="right", va="top",
                     color=muted, fontsize=8, bbox=dict(facecolor="#fcfcfb", edgecolor="none"))
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
