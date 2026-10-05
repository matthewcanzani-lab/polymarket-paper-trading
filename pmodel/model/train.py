"""Train the XGBoost model with a time-based split, evaluate it, save a version.

Design choices, in plain terms:

* The model STARTS FROM THE MARKET PRICE. XGBoost's "base margin" is set to the
  log-odds of the current price, so with no trees at all the model just repeats
  the market. Each tree can only learn a correction to it. This keeps the model
  honest: it has to find a reason to disagree with the market.

* TIME-BASED SPLIT, NOT RANDOM. Snapshots are ordered by prediction time t.
    - test  = the most recent ~20% of snapshots (t >= test_start)
    - valid = the ~15% before that, used only to decide when to stop training
    - train = everything earlier, AND only markets that had already RESOLVED
              before the next period started. A model deployed on test_start
              could not have known the outcome of a market still open then.
  So nothing the model learns from comes from after the moment it's tested on.

* Every saved model is a new folder under models/ (never overwritten), with
  its metrics and settings, so versions can be compared later (Phase 5).
"""
import json
import subprocess
from datetime import datetime, timezone

import numpy as np
import xgboost as xgb

from pmodel import config
from pmodel.model import evaluate
from pmodel.model.features import FEATURE_NAMES, logit

MODELS_DIR = config.PROJECT_ROOT / "models"

PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "eta": 0.03,
    "max_depth": 4,
    "min_child_weight": 50,   # each leaf needs ~50 markets' worth of weight
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "lambda": 5.0,
    "tree_method": "hist",
    "seed": 7,
}


def time_split(df, test_frac=0.20, valid_frac=0.15):
    test_start = int(np.quantile(df["t"], 1 - test_frac))
    test = df[df["t"] >= test_start]
    before = df[(df["t"] < test_start) & (df["closed_ts"] < test_start)]
    valid_start = int(np.quantile(before["t"], 1 - valid_frac))
    valid = before[before["t"] >= valid_start]
    train = before[(before["t"] < valid_start) & (before["closed_ts"] < valid_start)]
    return train.reset_index(drop=True), valid.reset_index(drop=True), \
        test.reset_index(drop=True), valid_start, test_start


def _dmatrix(df):
    margin = np.array([logit(p) for p in df["price"]])
    return xgb.DMatrix(df[FEATURE_NAMES], label=df["label"], weight=df["weight"],
                       base_margin=margin, feature_names=FEATURE_NAMES)


def fit(train, valid, rounds=3000):
    dtrain, dvalid = _dmatrix(train), _dmatrix(valid)
    booster = xgb.train(PARAMS, dtrain, num_boost_round=rounds,
                        evals=[(dtrain, "train"), (dvalid, "valid")],
                        early_stopping_rounds=100, verbose_eval=False)
    return booster


def predict(booster, df):
    return booster.predict(_dmatrix(df), iteration_range=(0, booster.best_iteration + 1))


def _iso(ts):
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=config.PROJECT_ROOT, text=True).strip()
    except Exception:
        return None


def run(df, log=print):
    train, valid, test, valid_start, test_start = time_split(df)
    log(f"train: {len(train):,} snapshots / {train['market_id'].nunique():,} markets "
        f"(predictions before {_iso(valid_start)}, all resolved by then)")
    log(f"valid: {len(valid):,} snapshots / {valid['market_id'].nunique():,} markets")
    log(f"test:  {len(test):,} snapshots / {test['market_id'].nunique():,} markets "
        f"(predictions from {_iso(test_start)})")

    booster = fit(train, valid)
    p_test = predict(booster, test)
    summ, cal_model, cal_market = evaluate.summary(test, p_test)
    hz = test.assign(horizon=_horizon_bucket(test["hours_to_end"]))
    by_cat = evaluate.breakdown(test, p_test, "category")
    by_hz = evaluate.breakdown(hz, p_test, "horizon")
    disagree = evaluate.disagreement(test, p_test)

    version = datetime.now(timezone.utc).strftime("v%Y%m%d_%H%M%S")
    out = MODELS_DIR / version
    out.mkdir(parents=True)
    booster.save_model(out / "model.json")
    importance = booster.get_score(importance_type="gain")
    meta = {
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "result_type": "BACKTEST (historical, out-of-sample by time) - not forward paper results",
        "features": FEATURE_NAMES,
        "params": PARAMS,
        "best_iteration": booster.best_iteration,
        "universe": {"min_volume_usd": config.MIN_VOLUME_USD,
                     "exclude_recurrences": list(config.EXCLUDE_RECURRENCES),
                     "exclude_question_patterns": list(config.EXCLUDE_QUESTION_PATTERNS)},
        "split": {"valid_start": _iso(valid_start), "test_start": _iso(test_start),
                  "train_snapshots": len(train), "valid_snapshots": len(valid),
                  "test_snapshots": len(test),
                  "train_markets": int(train["market_id"].nunique()),
                  "test_markets": int(test["market_id"].nunique())},
        "test_metrics": summ,
        "test_disagreement_over_5pts": disagree,
        "feature_importance_gain": dict(sorted(importance.items(), key=lambda kv: -kv[1])),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    report = _report(meta, cal_model, cal_market, by_cat, by_hz)
    (out / "backtest_report.txt").write_text(report)
    _calibration_chart(cal_model, cal_market, out / "calibration.png", summ)
    log(report)
    log(f"\nSaved model version {version} -> {out}")
    return out


def _horizon_bucket(hours):
    import pandas as pd
    return pd.cut(hours, [-np.inf, 24, 72, 168, 720, np.inf],
                  labels=["<1 day", "1-3 days", "3-7 days", "7-30 days", ">30 days"])


def _report(meta, cal_model, cal_market, by_cat, by_hz):
    m = meta["test_metrics"]
    d = meta["test_disagreement_over_5pts"]
    lines = [
        f"MODEL {meta['version']} — BACKTEST RESULTS (historical, not forward paper trading)",
        "=" * 78,
        f"Test period: predictions made from {meta['split']['test_start']}",
        f"Test set: {m['markets']:,} markets / {m['snapshots']:,} snapshots "
        "(metrics weight each market equally)",
        "",
        f"{'':28}{'model':>10}{'market price':>15}",
        f"{'Brier score (lower=better)':28}{m['brier_model']:>10.4f}{m['brier_market']:>15.4f}",
        f"{'Log loss (lower=better)':28}{m['logloss_model']:>10.4f}{m['logloss_market']:>15.4f}",
        f"{'Worst calibration gap':28}{m['max_cal_gap_model_pts']:>9.1f}p"
        f"{m['max_cal_gap_market_pts']:>14.1f}p",
        f"{'Win rate (>=50% calls)':28}{m['win_rate_model']*100:>9.1f}%",
        "",
        "Calibration — when the forecast says X%, how often did YES happen?",
        f"  {'band':>9}{'model says':>12}{'happened':>10}{'markets':>9}   |"
        f"{'market says':>12}{'happened':>10}",
    ]
    cm = cal_market.set_index("band")
    for _, r in cal_model.iterrows():
        k = cm.loc[r["band"]] if r["band"] in cm.index else None
        tail = (f"{k['forecast']*100:>11.1f}%{k['actual']*100:>9.1f}%" if k is not None else "")
        lines.append(f"  {r['band']:>9}{r['forecast']*100:>11.1f}%{r['actual']*100:>9.1f}%"
                     f"{r['markets']:>9.0f}   |{tail}")
    lines += ["", "Brier by category (model vs market)"]
    for _, r in by_cat.sort_values("markets", ascending=False).iterrows():
        better = "model" if r["brier_model"] < r["brier_market"] else "market"
        lines.append(f"  {r['category']:16}{r['markets']:>7,} mkts  "
                     f"{r['brier_model']:.4f} vs {r['brier_market']:.4f}  ({better} better)")
    lines += ["", "Brier by time left until scheduled end"]
    for _, r in by_hz.iterrows():
        lines.append(f"  {str(r['horizon']):16}{r['markets']:>7,} mkts  "
                     f"{r['brier_model']:.4f} vs {r['brier_market']:.4f}")
    lines += ["", "Where model and market disagree by more than 5 points (Phase 3 preview)"]
    if d.get("snapshots"):
        lines += [f"  {d['snapshots']:,} snapshots in {d['markets']:,} markets",
                  f"  model closer to the outcome: {d['model_closer_share']*100:.1f}% of the time",
                  f"  avg return per $1 if bet at market price, BEFORE slippage: "
                  f"{d['avg_return_per_dollar_before_slippage']*100:+.1f}%"]
    else:
        lines.append("  none")
    lines += ["", "Top features (gain):"]
    for k, v in list(meta["feature_importance_gain"].items())[:10]:
        lines.append(f"  {k:20}{v:10.2f}")
    return "\n".join(lines)


def _calibration_chart(cal_model, cal_market, path, summ):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid, surface = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    blue, orange = "#2a78d6", "#eb6834"
    fig, ax = plt.subplots(figsize=(6.5, 6), facecolor=surface)
    ax.set_facecolor(surface)
    ax.plot([0, 100], [0, 100], color=muted, linewidth=1, linestyle="--")
    ax.text(88, 82, "perfect", color=muted, fontsize=8, rotation=45)
    for tbl, color, name in ((cal_market, orange, "market price"), (cal_model, blue, "model")):
        ax.plot(tbl["forecast"] * 100, tbl["actual"] * 100, color=color, linewidth=2,
                marker="o", markersize=6, markeredgecolor=surface, markeredgewidth=1.5, label=name)
    ax.set_xlim(0, 100), ax.set_ylim(0, 100)
    ax.set_xlabel("forecast probability of YES (%)", color=muted)
    ax.set_ylabel("how often YES actually happened (%)", color=muted)
    ax.set_title(f"Calibration on the test period (BACKTEST)\n"
                 f"Brier: model {summ['brier_model']:.4f} · market {summ['brier_market']:.4f}",
                 loc="left", color=ink, fontsize=11)
    ax.grid(color=grid, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(grid)
    ax.tick_params(colors=muted)
    ax.legend(frameon=False, loc="upper left", labelcolor=ink)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
