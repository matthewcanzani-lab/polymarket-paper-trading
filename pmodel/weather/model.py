"""Forecast-based probabilities for daily temperature markets.

  P(range) = P(lo - 0.5 <= T < hi + 0.5),  T ~ Student-t(mu, sigma, df)
  mu    = as-issued forecast + the station's typical bias (learned from the past)
  sigma = the station's typical forecast error (learned from the past) x SCALE

Stations report whole degrees, hence the +/- 0.5. Bias and error width are
learned ONLY from days before the prediction period, and shrunk toward the
all-station average when a station has little history.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

OPEN = 1e8          # |lo| or |hi| beyond this means an open-ended range
SHRINK_N = 30       # pseudo-observations pulling a station toward the global average


def to_unit(c, unit):
    return c * 9 / 5 + 32 if unit == "F" else c


def range_prob(mu, sigma, lo, hi, df=6.0):
    a = -np.inf if lo <= -OPEN else lo - 0.5
    b = np.inf if hi >= OPEN else hi + 0.5
    dist = stats.t(df, loc=mu, scale=sigma)
    return float(dist.cdf(b) - dist.cdf(a))


@dataclass
class ErrorModel:
    """Bias and spread of forecast errors (°C) per (station, kind, lead)."""
    by_key: dict = field(default_factory=dict)
    global_: dict = field(default_factory=dict)
    scale: float = 1.0
    df: float = 6.0

    @classmethod
    def fit(cls, errors, scale=1.0, df=6.0):
        """errors: DataFrame with station, kind, lead, err_c (observed - forecast, °C)."""
        m = cls(scale=scale, df=df)
        for (kind, lead), g in errors.groupby(["kind", "lead"]):
            m.global_[(kind, lead)] = (g.err_c.mean(), g.err_c.std())
        for (st, kind, lead), g in errors.groupby(["station", "kind", "lead"]):
            gm, gs = m.global_[(kind, lead)]
            n = len(g)
            w = n / (n + SHRINK_N)
            bias = w * g.err_c.mean() + (1 - w) * gm
            var = (w * (g.err_c.var(ddof=1) if n > 1 else gs ** 2) + (1 - w) * gs ** 2)
            m.by_key[(st, kind, lead)] = (bias, float(np.sqrt(var)))
        return m

    def params(self, station, kind, lead):
        bias, sd = self.by_key.get((station, kind, lead), self.global_.get((kind, lead), (0.0, 2.0)))
        return bias, max(sd, 0.5) * self.scale

    def prob(self, forecast_c, station, kind, lead, unit, lo, hi):
        bias, sd_c = self.params(station, kind, lead)
        mu = to_unit(forecast_c + bias, unit)
        sd = sd_c * 9 / 5 if unit == "F" else sd_c
        return range_prob(mu, sd, lo, hi, self.df)

    def prob_batch(self, forecast_c, station, kind, lead, unit, lo, hi):
        """Vectorised prob() over arrays (same maths, one call)."""
        params = [self.params(s, k, l) for s, k, l in zip(station, kind, lead)]
        bias = np.array([b for b, _ in params])
        sd_c = np.array([s for _, s in params])
        is_f = np.asarray(unit) == "F"
        mu_c = np.asarray(forecast_c, dtype=float) + bias
        mu = np.where(is_f, mu_c * 9 / 5 + 32, mu_c)
        sd = np.where(is_f, sd_c * 9 / 5, sd_c)
        lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
        a = np.where(lo <= -OPEN, -np.inf, lo - 0.5)
        b = np.where(hi >= OPEN, np.inf, hi + 0.5)
        return stats.t.cdf((b - mu) / sd, self.df) - stats.t.cdf((a - mu) / sd, self.df)
