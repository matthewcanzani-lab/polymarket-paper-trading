"""Pure math for fake bets: fill prices from an order book, and Kelly sizing.

A binary share costs `c` and pays $1 if it wins. With win probability q:
    full Kelly fraction of bankroll = (q - c) / (1 - c)      (0 if negative)
We stake KELLY_MULTIPLIER (quarter) of that, capped.
"""


def kelly_fraction(q: float, cost: float) -> float:
    if not 0 < cost < 1:
        return 0.0
    return max((q - cost) / (1 - cost), 0.0)


def walk_book(levels, stake_usd: float, slippage: float):
    """Spend `stake_usd` buying from `levels` [(price, shares)], best first.

    Returns (average price paid per share INCLUDING slippage, shares bought),
    or (None, 0) if the book can't absorb the stake.
    """
    remaining, shares, spent = stake_usd, 0.0, 0.0
    for price, size in levels:
        px = min(price + slippage, 0.999)
        if px <= 0:
            continue
        take = min(size, remaining / px)
        shares += take
        spent += take * px
        remaining -= take * px
        if remaining <= 1e-9:
            return spent / shares, shares
    return None, 0.0


def no_side_levels(bids):
    """Buying NO = selling YES into the YES bids: NO costs (1 - bid)."""
    return [(1 - price, size) for price, size in bids]


def size_bet(q: float, levels, bankroll: float, cash: float, cfg) -> dict | None:
    """Decide the fake stake for buying a side with win probability q.

    Two passes: size from the best price, then re-check Kelly at the actual
    average fill (bigger bets walk deeper into the book and pay more).
    """
    if not levels:
        return None
    cap = min(cfg.MAX_STAKE_FRACTION * bankroll, cash)
    first_cost = min(levels[0][0] + cfg.SLIPPAGE_PER_SHARE, 0.999)
    stake = min(cfg.KELLY_MULTIPLIER * kelly_fraction(q, first_cost) * bankroll, cap)
    if stake < cfg.MIN_STAKE_USD:
        return None
    price, shares = walk_book(levels, stake, cfg.SLIPPAGE_PER_SHARE)
    if price is None:
        return None
    kelly = kelly_fraction(q, price)
    stake = min(stake, cfg.KELLY_MULTIPLIER * kelly * bankroll)
    if stake < cfg.MIN_STAKE_USD:
        return None
    price, shares = walk_book(levels, stake, cfg.SLIPPAGE_PER_SHARE)
    if price is None or q <= price:
        return None
    return {"stake": round(stake, 2), "fill_price": price, "shares": shares,
            "kelly_full": kelly, "expected_value": shares * q - stake}
