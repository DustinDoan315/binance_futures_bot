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

import numpy as np
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

def _calculate_stochastic(df: pd.DataFrame, k_period: int, d_period: int) -> pd.DataFrame:
    lowest_low = df["low"].rolling(window=k_period).min()
    highest_high = df["high"].rolling(window=k_period).max()
    k = ((df["close"] - lowest_low) / (highest_high - lowest_low).replace(0, pd.NA)) * 100
    k = k.fillna(50)
    d = k.rolling(window=d_period).mean().fillna(50)
    return pd.DataFrame({"stoch_k": k, "stoch_d": d}, index=df.index)


def _evaluate_rsi(cur: pd.Series, config: dict) -> tuple[bool, bool]:
    rsi_long_threshold = config.get("rsi_long_threshold", 55)
    rsi_short_threshold = config.get("rsi_short_threshold", 45)
    rsi_value = cur.get("rsi")
    long_ok = pd.notna(rsi_value) and rsi_value >= rsi_long_threshold
    short_ok = pd.notna(rsi_value) and rsi_value <= rsi_short_threshold
    return long_ok, short_ok


def _evaluate_stochastic(cur: pd.Series) -> tuple[bool, bool]:
    stoch_k = cur.get("stoch_k")
    stoch_d = cur.get("stoch_d")
    long_ok = pd.notna(stoch_k) and pd.notna(stoch_d) and stoch_k > stoch_d and stoch_k < 80
    short_ok = pd.notna(stoch_k) and pd.notna(stoch_d) and stoch_k < stoch_d and stoch_k > 20
    return long_ok, short_ok


def _evaluate_confirmation(
    enriched: pd.DataFrame,
    cur: pd.Series,
    config: dict,
) -> tuple[bool, bool, bool, bool]:
    window = max(1, config.get("confirmation_window", 5))
    recent = enriched.iloc[-window:]
    ema_fast_slope = recent["ema_fast"].diff().mean()
    ema_slow_slope = recent["ema_slow"].diff().mean()
    momentum_long_ok = ema_fast_slope > 0 and ema_slow_slope > 0
    momentum_short_ok = ema_fast_slope < 0 and ema_slow_slope < 0
    breakout_long_ok = cur["close"] >= recent["high"].max()
    breakout_short_ok = cur["close"] <= recent["low"].min()
    return momentum_long_ok, momentum_short_ok, breakout_long_ok, breakout_short_ok


def _evaluate_volume_flow(cur: pd.Series) -> tuple[bool, bool]:
    obv = cur.get("obv")
    obv_signal = cur.get("obv_signal")
    if pd.isna(obv) or pd.isna(obv_signal):
        return False, False
    long_ok = obv > obv_signal
    short_ok = obv < obv_signal
    return long_ok, short_ok


def _count_true(flags: list[bool]) -> int:
    return sum(1 for flag in flags if flag)

def calculate_indicators(
    df: pd.DataFrame,
    ema_fast: int,
    ema_slow: int,
    atr_period: int,
    rsi_period: int,
    stoch_k_period: int,
    stoch_d_period: int,
    macd_fast: int,
    macd_slow: int,
    macd_signal: int,
    obv_smoothing: int,
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
    stoch_df = _calculate_stochastic(result, stoch_k_period, stoch_d_period)
    ema_macd_fast = result["close"].ewm(span=macd_fast, adjust=False).mean()
    ema_macd_slow = result["close"].ewm(span=macd_slow, adjust=False).mean()
    macd_line = ema_macd_fast - ema_macd_slow
    macd_signal_line = macd_line.ewm(span=macd_signal, adjust=False).mean()
    macd_hist = macd_line - macd_signal_line

    price_diff = result["close"].diff().fillna(0.0)
    direction = np.sign(price_diff)
    obv = (direction * result["volume"]).cumsum()
    obv_signal = obv.ewm(span=max(1, obv_smoothing), adjust=False).mean()

    result = pd.concat(
        [
            result,
            stoch_df,
            pd.DataFrame(
                {
                    "macd_line": macd_line,
                    "macd_signal": macd_signal_line,
                    "macd_hist": macd_hist,
                    "obv": obv,
                    "obv_signal": obv_signal,
                },
                index=result.index,
            ),
        ],
        axis=1,
    )
    return result


def generate_signal(
    df: pd.DataFrame,
    config: dict,
    _current_position: Optional[str] = None,
) -> dict:
    """Generate trading signals based on EMA, ATR, and confirmation indicators.

    Args:
        df: DataFrame containing at least the most recent two rows with indicators.
        config: Scenario configuration dictionary with keys: ema_fast, ema_slow,
            atr_period, risk_reward_ratio, etc.
        current_position: Kept for backward compatibility (ignored).

    Returns:
        A dictionary with keys:
            actions: list of zero, one, or two entries describing orders to place.
            action: legacy single-action string for compatibility ('HOLD' when no entries).
    """
    if len(df) < 3:
        return {"action": "HOLD", "actions": []}

    # Compute indicators
    ema_fast = config["ema_fast"]
    ema_slow = config["ema_slow"]
    atr_period = config.get("atr_period", 14)
    rsi_period = config.get("rsi_period", 14)
    stoch_k_period = config.get("stoch_k_period", 14)
    stoch_d_period = config.get("stoch_d_period", 3)
    macd_fast = config.get("macd_fast", 12)
    macd_slow = config.get("macd_slow", 26)
    macd_signal = config.get("macd_signal", 9)
    obv_smoothing = config.get("obv_smoothing", 10)
    allow_long = config.get("allow_long", True)
    allow_short = config.get("allow_short", True)
    enriched = calculate_indicators(
        df,
        ema_fast,
        ema_slow,
        atr_period,
        rsi_period,
        stoch_k_period,
        stoch_d_period,
        macd_fast,
        macd_slow,
        macd_signal,
        obv_smoothing,
    )

    # Use the last two bars to detect pull‑back completion
    prev = enriched.iloc[-2]
    cur = enriched.iloc[-1]

    if pd.isna(cur["atr"]) or cur["atr"] <= 0:
        return {"action": "HOLD", "actions": []}

    # Determine trend direction
    trend_long = cur["ema_fast"] > cur["ema_slow"] and cur["close"] > cur["ema_slow"]
    trend_short = cur["ema_fast"] < cur["ema_slow"] and cur["close"] < cur["ema_slow"]

    rsi_long_ok, rsi_short_ok = _evaluate_rsi(cur, config)
    stoch_long_ok, stoch_short_ok = _evaluate_stochastic(cur)
    momentum_long_ok, momentum_short_ok, breakout_long_ok, breakout_short_ok = _evaluate_confirmation(
        enriched, cur, config
    )
    volume_long_ok, volume_short_ok = _evaluate_volume_flow(cur)
    macd_hist = cur.get("macd_hist")
    macd_long_ok = pd.notna(macd_hist) and macd_hist > 0
    macd_short_ok = pd.notna(macd_hist) and macd_hist < 0

    pullback_tolerance = max(0.0, config.get("pullback_tolerance", 0.001))

    def _within_band(value: float, target: float, tolerance: float) -> bool:
        return target * (1 - tolerance) <= value <= target * (1 + tolerance)

    pullback_long_ok = (
        prev["close"] <= prev["ema_fast"] * (1 + pullback_tolerance)
        and _within_band(cur["close"], cur["ema_fast"], pullback_tolerance)
    )
    pullback_short_ok = (
        prev["close"] >= prev["ema_fast"] * (1 - pullback_tolerance)
        and _within_band(cur["close"], cur["ema_fast"], pullback_tolerance)
    )

    long_score = _count_true(
        [
            trend_long,
            pullback_long_ok,
            rsi_long_ok,
            stoch_long_ok,
            momentum_long_ok,
            breakout_long_ok,
            macd_long_ok,
            volume_long_ok,
        ]
    )
    short_score = _count_true(
        [
            trend_short,
            pullback_short_ok,
            rsi_short_ok,
            stoch_short_ok,
            momentum_short_ok,
            breakout_short_ok,
            macd_short_ok,
            volume_short_ok,
        ]
    )
    long_threshold = max(3, config.get("long_score_threshold", 5))
    short_threshold = max(3, config.get("short_score_threshold", 5))

    # Determine pull‑back completion conditions
    entry_long = (
        allow_long
        and trend_long
        and pullback_long_ok
        and long_score >= long_threshold
    )
    entry_short = (
        allow_short
        and trend_short
        and pullback_short_ok
        and short_score >= short_threshold
    )

    actions: list[dict] = []
    if entry_long:
        stop_distance = cur["atr"]
        take_distance = stop_distance * config["risk_reward_ratio"]
        actions.append(
            {
                "action": "OPEN_LONG",
                "entry_price": cur["close"],
                "stop_distance": stop_distance,
                "take_distance": take_distance,
            }
        )
    if entry_short:
        stop_distance = cur["atr"]
        take_distance = stop_distance * config["risk_reward_ratio"]
        actions.append(
            {
                "action": "OPEN_SHORT",
                "entry_price": cur["close"],
                "stop_distance": stop_distance,
                "take_distance": take_distance,
            }
        )
    primary_action = actions[0]["action"] if actions else "HOLD"
    return {"action": primary_action, "actions": actions}
