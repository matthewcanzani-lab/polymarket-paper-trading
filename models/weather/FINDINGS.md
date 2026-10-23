# Weather model — backtest findings (BACKTEST, not forward results)

Run 2026-10-23. Daily high/low temperature markets, 49 cities (Hong Kong
excluded: resolves on the Observatory, not an airport). Forecasts are Open-Meteo
archived model runs, used only after `issued_by_utc` (valid hour - N days + 6h).

Protocol: forecast bias/spread learned from Jan–Jul; spread scale + blend
weights chosen on August; tested once on Sep 1 – Oct 18 (123,475 market
snapshots, 4,571 city-days). All markets included regardless of volume.

## What held up
* Forecasts carry information the recorded prices don't: a blend of
  market price + forecast probability beat the recorded price on Brier score
  (0.0672 vs 0.0798; 95% range of the difference -0.0132 to -0.0120).
* The gain is 1–2 days ahead. On the morning of the day there is none
  (0.0552 vs 0.0544) — the market knows more by then.

## What did NOT hold up
* Simulated betting showed +25% per $ after 2c costs. Not believable:
  - the gain is concentrated in markets with < $1k final volume (+33%), and
    is NEGATIVE in markets with $5k+ volume (-8%);
  - recorded prices that hadn't moved in 12h were wildly off (priced ~37%,
    happened ~11%) — they are placeholder midpoints of wide spreads nobody
    traded at;
  - live check of open weather markets: median spread 4c for ranges priced
    10–90%, 21% wider than 10c; median $936 resting liquidity.
* Historical data records only a midpoint, not the price you could buy at,
  so the backtest cannot say whether the edge was tradeable.
* Note: the price-activity breakdown was done AFTER seeing test results.
  Any rule based on it must be validated on new data, i.e. forward.

## Conclusion
Promising signal, unproven profit. The honest next step is a forward paper
test that prices every fake fill from the LIVE order book (which the scorer
already does), so wide spreads and thin books are charged for real.
