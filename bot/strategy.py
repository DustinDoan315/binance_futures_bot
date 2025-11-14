"""
Trading strategy implementation.

This module implements a simple trend‑following strategy based on exponential
moving averages (EMAs) and the average true range (ATR).  The idea is to
determine whether the market is trending up or down and to enter trades on
pull‑backs to the fast EMA.

The `generate_signal` function is pure – it depends only on the provided
historical data and the current position state.  This makes it easy to
backtest and unit test without connecting to an exchange.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd


def calculate_indicators(df: pd.DataFrame, ema_fast: int, ema_slow: int, atr_period: int) -> pd.DataFrame:
    """Compute EMA and ATR indicators.

    Args:
        df: DataFrame with columns 'open', 'high', 'low', 'close'.  Index should be datetime.
        ema_fast: Period for the fast EMA.
        ema_slow: Period for the slow EMA.
        atr_period: Period for the ATR.

    Returns:
        The input DataFrame with additional columns 'ema_fast', 'ema_slow' and 'atr'.
    """
    result = df.copy().astype(float)
    # Fast and slow EMAs
    result["ema_fast"] = result["close"].ewm(span=ema_fast, adjust=False).mean()
    result["ema_slow"] = result["close"].ewm(span=ema_slow, adjust=False).mean()

    # True range calculation
    high_low = result["high"] - result["low"]
    high_close_prev = (result["high"] - result["close"].shift(1)).abs()
    low_close_prev = (result["low"] - result["close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    # ATR
    result["atr"] = true_range.rolling(window=atr_period).mean()
    return result


def generate_signal(
    df: pd.DataFrame,
    config: dict,
    current_position: Optional[str] = None,
) -> dict:
    """Generate a trading signal based on EMA and ATR.

    Args:
        df: DataFrame containing at least the most recent two rows with indicators.
        config: Scenario configuration dictionary with keys: ema_fast, ema_slow,
            atr_period, risk_reward_ratio.
        current_position: 'LONG', 'SHORT' or None indicating existing position.

    Returns:
        A dictionary with keys:
            action: 'OPEN_LONG', 'OPEN_SHORT' or 'HOLD'
            entry_price: the close price of the latest candle
            stop_distance: recommended stop‑loss distance (ATR)
            take_distance: recommended take‑profit distance (ATR * risk_reward_ratio)
    """
    if len(df) < 3:
        return {"action": "HOLD"}

    # Compute indicators
    ema_fast = config["ema_fast"]
    ema_slow = config["ema_slow"]
    atr_period = config.get("atr_period", 14)
    enriched = calculate_indicators(df, ema_fast, ema_slow, atr_period)

    # Use the last two bars to detect pull‑back completion
    prev = enriched.iloc[-2]
    cur = enriched.iloc[-1]

    # Determine trend direction
    trend_long = cur["ema_fast"] > cur["ema_slow"] and cur["close"] > cur["ema_slow"]
    trend_short = cur["ema_fast"] < cur["ema_slow"] and cur["close"] < cur["ema_slow"]

    # Determine pull‑back completion conditions
    entry_long = (
        current_position is None
        and trend_long
        and prev["close"] < prev["ema_fast"]
        and cur["close"] > cur["ema_fast"]
    )
    entry_short = (
        current_position is None
        and trend_short
        and prev["close"] > prev["ema_fast"]
        and cur["close"] < cur["ema_fast"]
    )

    if entry_long:
        stop_distance = cur["atr"]
        take_distance = stop_distance * config["risk_reward_ratio"]
        return {
            "action": "OPEN_LONG",
            "entry_price": cur["close"],
            "stop_distance": stop_distance,
            "take_distance": take_distance,
        }
    if entry_short:
        stop_distance = cur["atr"]
        take_distance = stop_distance * config["risk_reward_ratio"]
        return {
            "action": "OPEN_SHORT",
            "entry_price": cur["close"],
            "stop_distance": stop_distance,
            "take_distance": take_distance,
        }
    # Otherwise hold
    return {"action": "HOLD"}
