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


def _calculate_rsi(series: pd.Series, period: int) -> pd.Series:
    if period <= 0:
        return pd.Series(index=series.index, dtype=float)
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def _calculate_macd(series: pd.Series, fast: int, slow: int, signal: int) -> pd.DataFrame:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame(
        {"macd": macd_line, "macd_signal": signal_line, "macd_hist": histogram},
        index=series.index,
    )


def calculate_indicators(
    df: pd.DataFrame,
    ema_fast: int,
    ema_slow: int,
    atr_period: int,
    rsi_period: int,
    macd_fast: int,
    macd_slow: int,
    macd_signal: int,
) -> pd.DataFrame:
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
    result["rsi"] = _calculate_rsi(result["close"], rsi_period)
    macd_df = _calculate_macd(result["close"], macd_fast, macd_slow, macd_signal)
    result = pd.concat([result, macd_df], axis=1)
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
    rsi_period = config.get("rsi_period", 14)
    macd_fast = config.get("macd_fast", 12)
    macd_slow = config.get("macd_slow", 26)
    macd_signal = config.get("macd_signal", 9)
    enriched = calculate_indicators(
        df,
        ema_fast,
        ema_slow,
        atr_period,
        rsi_period,
        macd_fast,
        macd_slow,
        macd_signal,
    )

    # Use the last two bars to detect pull‑back completion
    prev = enriched.iloc[-2]
    cur = enriched.iloc[-1]

    if pd.isna(cur["atr"]) or cur["atr"] <= 0:
        return {"action": "HOLD"}

    # Determine trend direction
    trend_long = cur["ema_fast"] > cur["ema_slow"] and cur["close"] > cur["ema_slow"]
    trend_short = cur["ema_fast"] < cur["ema_slow"] and cur["close"] < cur["ema_slow"]

    rsi_long_threshold = config.get("rsi_long_threshold", 55)
    rsi_short_threshold = config.get("rsi_short_threshold", 45)
    rsi_value = cur.get("rsi")
    rsi_long_ok = pd.notna(rsi_value) and rsi_value >= rsi_long_threshold
    rsi_short_ok = pd.notna(rsi_value) and rsi_value <= rsi_short_threshold

    macd_value = cur.get("macd")
    macd_signal_value = cur.get("macd_signal")
    macd_long_ok = pd.notna(macd_value) and pd.notna(macd_signal_value) and macd_value > macd_signal_value
    macd_short_ok = pd.notna(macd_value) and pd.notna(macd_signal_value) and macd_value < macd_signal_value

    # Determine pull‑back completion conditions
    entry_long = (
        current_position is None
        and trend_long
        and prev["close"] < prev["ema_fast"]
        and cur["close"] > cur["ema_fast"]
        and rsi_long_ok
        and macd_long_ok
    )
    entry_short = (
        current_position is None
        and trend_short
        and prev["close"] > prev["ema_fast"]
        and cur["close"] < cur["ema_fast"]
        and rsi_short_ok
        and macd_short_ok
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
