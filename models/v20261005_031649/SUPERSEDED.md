# Superseded — do not trust this version's backtest

This model's training and test sets were selected by each market's FINAL
lifetime volume (>= $10k). Final volume leaks the outcome: upsets attract
heavy late trading, so the sample over-represents upsets. In the test set,
the share of 10–25% underdogs that won rose from ~20% (final volume $10–25k)
to ~27% (> $1M).

That made the market price look miscalibrated toward underdogs, and the
model learned to exploit it. Its "+24% return on disagreements" is not
achievable live, where you only know volume SO FAR.

Fixed in the next version: snapshots are kept only if the market had already
reached $10k of volume at the moment of the prediction, and trading activity
up to that moment is a feature.
