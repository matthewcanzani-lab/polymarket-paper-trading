"""Scoring a probability forecast: Brier score, log loss, calibration.

Every metric is computed for the MODEL and for the MARKET PRICE (treating the
price as a forecast). The model is only useful if it beats the market.
"""
import numpy as np
import pandas as pd

BINS = np.linspace(0, 1, 11)  # 0-10%, 10-20%, ... 90-100%


def brier(y, p, w):
    """Mean squared error of the probability. 0 is perfect; always saying 50% scores 0.25."""
    return float(np.average((p - y) ** 2, weights=w))


def log_loss(y, p, w):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(np.average(-(y * np.log(p) + (1 - y) * np.log(1 - p)), weights=w))


def calibration_table(y, p, w):
    """For each 10-point band of forecasts: average forecast vs how often YES happened."""
    idx = np.clip(np.digitize(p, BINS) - 1, 0, len(BINS) - 2)
    rows = []
    for b in range(len(BINS) - 1):
        m = idx == b
        if not m.any():
            continue
        rows.append({
            "band": f"{BINS[b]*100:.0f}-{BINS[b+1]*100:.0f}%",
            "forecast": float(np.average(p[m], weights=w[m])),
            "actual": float(np.average(y[m], weights=w[m])),
            "markets": float(w[m].sum()),  # market-weighted count
        })
    return pd.DataFrame(rows)


def max_calibration_gap(table, min_markets=50):
    """Largest |forecast - actual| in percentage points, over bands with enough data."""
    t = table[table["markets"] >= min_markets]
    return float((t["forecast"] - t["actual"]).abs().max() * 100) if len(t) else float("nan")


def summary(df, p_model):
    y, w, p_mkt = df["label"].to_numpy(), df["weight"].to_numpy(), df["price"].to_numpy()
    cal_m, cal_k = calibration_table(y, p_model, w), calibration_table(y, p_mkt, w)
    return {
        "snapshots": int(len(df)),
        "markets": int(df["market_id"].nunique()),
        "brier_model": brier(y, p_model, w),
        "brier_market": brier(y, p_mkt, w),
        "logloss_model": log_loss(y, p_model, w),
        "logloss_market": log_loss(y, p_mkt, w),
        "win_rate_model": float(np.average((p_model >= 0.5) == (y == 1), weights=w)),
        "max_cal_gap_model_pts": max_calibration_gap(cal_m),
        "max_cal_gap_market_pts": max_calibration_gap(cal_k),
    }, cal_m, cal_k


def breakdown(df, p_model, by):
    out = []
    for key, g in df.groupby(by, observed=True):
        idx = g.index.to_numpy()
        y, w = g["label"].to_numpy(), g["weight"].to_numpy()
        out.append({by: key, "markets": int(g["market_id"].nunique()),
                    "brier_model": brier(y, p_model[idx], w),
                    "brier_market": brier(y, g["price"].to_numpy(), w)})
    return pd.DataFrame(out)


def disagreement(df, p_model, threshold=0.05):
    """Preview of Phase 3: snapshots where model and market differ by > threshold.

    'avg_return' = average profit per $1 staked if you'd bought the side the model
    favoured AT THE MARKET PRICE, before any slippage or fees. A backtest only.
    """
    p_mkt = df["price"].to_numpy()
    y = df["label"].to_numpy()
    w = df["weight"].to_numpy()
    edge = p_model - p_mkt
    m = np.abs(edge) > threshold
    if not m.any():
        return {"snapshots": 0}
    buy_yes = edge[m] > 0
    cost = np.where(buy_yes, p_mkt[m], 1 - p_mkt[m])
    won = np.where(buy_yes, y[m] == 1, y[m] == 0)
    ret = np.where(won, (1 - cost) / cost, -1.0)
    return {
        "snapshots": int(m.sum()),
        "markets": int(df.loc[m, "market_id"].nunique()),
        "model_closer_share": float(np.average(
            np.abs(p_model[m] - y[m]) < np.abs(p_mkt[m] - y[m]), weights=w[m])),
        "avg_return_per_dollar_before_slippage": float(np.average(ret, weights=w[m])),
    }
