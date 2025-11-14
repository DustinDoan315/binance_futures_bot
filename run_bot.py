#!/usr/bin/env python3
"""
Entry point for the Binance futures trading bot.

This script demonstrates how the various components of the bot fit together.  It
loads a risk scenario, obtains market data, generates a signal, computes order
parameters and (optionally) submits orders to Binance.  By default it runs in
simulation mode and prints the proposed order details.

Usage:
    python run_bot.py --scenario safe    # safe, neutral or risky
    python run_bot.py --scenario neutral --symbol ETHUSDT
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd

from config.scenarios import get_scenario
from bot.exchange import BinanceExchange
from bot.data_feed import fetch_klines
from bot.strategy import generate_signal
from bot.executor import open_position


def simulate_price_series(n: int = 300) -> pd.DataFrame:
    """Generate a synthetic OHLCV DataFrame for demonstration.

    This helper creates a random walk with slight drift to illustrate the
    strategy when real API data is unavailable (e.g. network errors).  The
    resulting DataFrame has columns: open, high, low, close, volume and an
    index of timestamps at one‑minute intervals.
    """
    rng = np.random.default_rng()
    # Start price around 10000
    prices = [10_000.0]
    for _ in range(n - 1):
        change = rng.normal(loc=0.0, scale=50.0)
        prices.append(prices[-1] + change)
    prices = np.array(prices)
    # Create OHLCV by adding small random ranges
    opens = prices[:-1]
    closes = prices[1:]
    highs = np.maximum(opens, closes) + rng.uniform(0, 20, size=n - 1)
    lows = np.minimum(opens, closes) - rng.uniform(0, 20, size=n - 1)
    volumes = rng.uniform(1, 100, size=n - 1)
    ts = pd.date_range(end=pd.Timestamp.utcnow(), periods=n - 1, freq="T")
    df = pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        },
        index=ts,
    )
    return df


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Binance futures bot")
    parser.add_argument("--scenario", type=str, default="safe", help="Risk scenario to use: safe, neutral or risky")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading pair, e.g. BTCUSDT")
    parser.add_argument("--interval", type=str, default="15m", help="Kline interval (e.g. 1m, 15m)")
    parser.add_argument("--live", action="store_true", help="Send orders to Binance instead of simulating")
    args = parser.parse_args(argv)

    # Load scenario configuration
    try:
        scenario = get_scenario(args.scenario)
    except KeyError as e:
        print(e)
        return 1

    # Instantiate the exchange client
    # The exchange will read your API keys from environment variables
    try:
        exchange = BinanceExchange()
    except RuntimeError as e:
        # If keys are missing and live trading is requested, abort
        if args.live:
            print(f"Fatal: {e}")
            return 1
        # Otherwise, warn and continue in simulation mode
        print(f"Warning: {e}. Continuing in simulation mode.")
        exchange = None

    # Fetch market data
    df: pd.DataFrame
    if exchange is not None and not args.live:
        try:
            df = fetch_klines(exchange, args.symbol, args.interval, limit=300)
        except Exception as e:
            print(f"Error fetching klines: {e}. Falling back to simulated data.")
            df = simulate_price_series(300)
    elif exchange is not None and args.live:
        try:
            df = fetch_klines(exchange, args.symbol, args.interval, limit=300)
        except Exception as e:
            print(f"Fatal: could not fetch klines for live trading: {e}")
            return 1
    else:
        # Exchange is None, so we must simulate
        df = simulate_price_series(300)

    # Generate trading signal
    signal = generate_signal(df, scenario, current_position=None)
    print(f"Strategy signal: {signal['action']}")
    if signal["action"] == "HOLD":
        print("No entry signal generated.")
        return 0

    # In this example we assume a fixed equity of $10,000.
    equity_usd = 10_000.0
    print(f"Using equity: ${equity_usd:.2f}, scenario risk: {scenario['risk_percentage']*100:.0f}%")

    # In live mode we need a real exchange instance
    if args.live and exchange is None:
        print("Cannot place live orders without API keys.")
        return 1

    # Place (or simulate) orders
    order_result = open_position(
        exchange=exchange if exchange is not None else BinanceExchange(base_url=""),
        symbol=args.symbol,
        equity_usd=equity_usd,
        signal=signal,
        config=scenario,
        live=args.live,
    )
    # Print order details
    print("Entry order:")
    print(order_result.entry_order)
    print("Stop‑loss order:")
    print(order_result.stop_order)
    print("Take‑profit order:")
    print(order_result.take_order)
    return 0


if __name__ == "__main__":
    sys.exit(main())
