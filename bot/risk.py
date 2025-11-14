"""
Risk management utilities.

This module calculates position sizing and exit levels based on the user's risk
profile.  It converts the abstract risk percentage into a concrete quantity
that the exchange can accept.  The formulas here assume isolated margin on
Binance Futures and do not account for fees or slippage.

Note: This code is simplified for educational purposes and should be
thoroughly tested before being used in live trading.
"""

from __future__ import annotations

import math
from typing import Tuple


def calculate_position_size(
    equity_usd: float, risk_percentage: float, stop_distance: float, entry_price: float
) -> float:
    """Calculate the quantity of the base asset to trade.

    The risk amount is computed as `equity_usd * risk_percentage`.  This is the
    maximum dollar amount the trader is willing to lose on the trade.  Given a
    stop‑loss placed `stop_distance` away from the entry price, the notional
    position size (in USD) is `risk_amount / stop_distance`.  To convert this
    notional into units of the base asset, divide by the entry price.

    Args:
        equity_usd: Current account equity in USD.
        risk_percentage: Fraction of equity to risk (0.0–1.0).
        stop_distance: Distance between entry and stop price (in price units).
        entry_price: Price at which the position will be opened.

    Returns:
        Quantity of the base asset to trade.
    """
    risk_amount = equity_usd * risk_percentage
    if risk_amount <= 0:
        return 0.0
    # Avoid division by zero or invalid inputs
    if (
        stop_distance <= 0
        or entry_price <= 0
        or math.isnan(stop_distance)
        or math.isnan(entry_price)
    ):
        return 0.0
    position_notional = risk_amount / stop_distance
    quantity = position_notional / entry_price
    return quantity


def compute_exit_prices(
    entry_price: float,
    stop_distance: float,
    take_distance: float,
    side: str,
) -> Tuple[float, float]:
    """Compute stop‑loss and take‑profit prices.

    Args:
        entry_price: Price at which the position will be opened.
        stop_distance: Absolute distance between entry and stop price.
        take_distance: Absolute distance between entry and take‑profit price.
        side: 'LONG' or 'SHORT'.

    Returns:
        Tuple (stop_price, take_price).  For long positions the stop price
        is entry_price - stop_distance and the take price is entry_price +
        take_distance.  For short positions the directions are reversed.
    """
    if side.upper() == "LONG":
        stop_price = entry_price - stop_distance
        take_price = entry_price + take_distance
    elif side.upper() == "SHORT":
        stop_price = entry_price + stop_distance
        take_price = entry_price - take_distance
    else:
        raise ValueError(f"Unknown side {side}")
    return stop_price, take_price
