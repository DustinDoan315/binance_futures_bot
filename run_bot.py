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
from bot.monitor import plot_monitoring_dashboard, simulate_trading_session
from bot.live_monitor import run_live_chart


def simulate_price_series(length: int = 300, seed: Optional[int] = None) -> pd.DataFrame:
    """Generate a synthetic OHLCV DataFrame for demonstration.

    This helper creates a random walk with slight drift to illustrate the
    strategy when real API data is unavailable (e.g. network errors).  The
    resulting DataFrame has columns: open, high, low, close, volume and an
    index of timestamps at one‑minute intervals.
    """
    if length < 2:
        raise ValueError("length must be at least 2 to form candles")
    rng = np.random.default_rng(seed)
    # Start price around 10000
    prices = [1_000.0]
    for _ in range(length):
        change = rng.normal(loc=0.0, scale=50.0)
        prices.append(prices[-1] + change)
    prices = np.array(prices)
    # Create OHLCV by adding small random ranges
    opens = prices[:-1]
    closes = prices[1:]
    highs = np.maximum(opens, closes) + rng.uniform(0, 15, size=length)
    lows = np.minimum(opens, closes) - rng.uniform(0, 10, size=length)
    volumes = rng.uniform(1, 100, size=length)
    ts = pd.date_range(end=pd.Timestamp.utcnow(), periods=length, freq="1min")
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


def interval_to_minutes(interval: str) -> int:
    """Convert Binance-style intervals (e.g. '15m', '1h') to minutes."""

    if not interval:
        raise ValueError("Interval string is empty")
    unit = interval[-1]
    value = interval[:-1]
    try:
        magnitude = int(value)
    except ValueError as exc:
        raise ValueError(f"Invalid interval value '{interval}'") from exc
    multipliers = {
        "m": 1,
        "h": 60,
        "d": 60 * 24,
        "w": 60 * 24 * 7,
    }
    if unit not in multipliers:
        raise ValueError(f"Unsupported interval unit '{unit}' in '{interval}'")
    return magnitude * multipliers[unit]


def candles_from_days(days: float, interval: str) -> int:
    minutes = interval_to_minutes(interval)
    total_minutes = max(days, 0) * 24 * 60
    candles = int(total_minutes / minutes) if minutes else 0
    return max(candles, 1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Binance futures bot")
    parser.add_argument("--scenario", type=str, default="safe", help="Risk scenario to use: safe, neutral or risky")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading pair, e.g. BTCUSDT")
    parser.add_argument("--interval", type=str, default="15m", help="Kline interval (e.g. 1m, 15m)")
    parser.add_argument("--live", action="store_true", help="Send orders to Binance instead of simulating")
    parser.add_argument(
        "--equity",
        type=float,
        default=1_000.0,
        help="Account equity in USD used for position sizing (default: 1,000)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=300,
        help="Number of candles to download or simulate (auto-adjusted if too low)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Force offline mode and use synthetic candles even if the exchange is available",
    )
    parser.add_argument(
        "--days",
        type=float,
        default=None,
        help="Number of days of candles to fetch (overrides --limit when larger)",
    )
    monitor_group = parser.add_mutually_exclusive_group()
    monitor_group.add_argument(
        "--monitor",
        action="store_true",
        help="Run a multi-trade simulation with balance tracking and chart output",
    )
    monitor_group.add_argument(
        "--monitor-live",
        action="store_true",
        help="Animate the price and equity curves in real time for testing",
    )
    parser.add_argument(
        "--chart-file",
        type=str,
        default="monitor_report.png",
        help="Where to save the monitoring chart (used with --monitor)",
    )
    parser.add_argument(
        "--chart-delay",
        type=float,
        default=0.5,
        help="Seconds between live chart frames when using --monitor-live",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for synthetic candles (offline mode)",
    )
    return parser


def init_exchange(live: bool) -> Optional[BinanceExchange]:
    try:
        return BinanceExchange(require_api_keys=live)
    except RuntimeError as e:
        if live:
            print(f"Fatal: {e}")
            return None
        print(f"Warning: {e}. Continuing in simulation mode.")
        return None


def determine_candle_limit(requested: int, scenario: dict) -> int:
    min_candles = max(scenario["ema_slow"] + 5, scenario.get("atr_period", 14) + 5, 50)
    candle_limit = max(requested, min_candles)
    if candle_limit != requested:
        print(f"Adjusted candle limit to {candle_limit} to satisfy indicator warm-up.")
    return candle_limit


def fetch_market_data(
    *,
    args,
    scenario: dict,
    exchange: Optional[BinanceExchange],
) -> pd.DataFrame:
    requested = args.limit
    if args.days is not None:
        requested = max(requested, candles_from_days(args.days, args.interval))
    candle_limit = determine_candle_limit(requested, scenario)
    df: Optional[pd.DataFrame] = None
    if not args.offline:
        try:
            if exchange is None:
                raise RuntimeError("Exchange client unavailable; cannot download klines.")
            df = fetch_klines(exchange, args.symbol, args.interval, limit=candle_limit)
        except Exception as e:
            if args.live:
                raise RuntimeError(f"Fatal: could not fetch klines for live trading: {e}") from e
            print(f"Error fetching klines: {e}. Falling back to simulated data.")
    if df is None:
        df = simulate_price_series(candle_limit, seed=args.seed)
    return df


def run_monitor_mode(df: pd.DataFrame, scenario: dict, args) -> int:
    trades, equity_curve = simulate_trading_session(df, scenario, args.equity)
    if trades:
        print("Trade log:")
        for trade in trades:
            print(
                f"{trade.entry_time} -> {trade.exit_time} | {trade.side:<5} "
                f"{trade.outcome:<11} | PnL: {trade.pnl:>8.2f} | Balance: {trade.balance_after:>8.2f}"
            )
        final_balance = trades[-1].balance_after
    else:
        print("No trades were generated in the monitoring session.")
        final_balance = args.equity
    chart_path = plot_monitoring_dashboard(df, trades, equity_curve, args.chart_file)
    print(f"Monitoring chart saved to {chart_path}")
    print(f"Starting balance: ${args.equity:.2f}")
    print(f"Ending balance:   ${final_balance:.2f}")
    print(f"Net PnL:          ${final_balance - args.equity:.2f}")
    return 0
def run_live_chart_mode(df: pd.DataFrame, scenario: dict, args) -> int:
    print("Starting interactive live chart (close the window to stop)...")
    run_live_chart(df, scenario, args.equity, delay=args.chart_delay)
    return 0


def run_single_cycle(
    *,
    df: pd.DataFrame,
    scenario: dict,
    args,
    exchange: Optional[BinanceExchange],
) -> int:
    signal = generate_signal(df, scenario, current_position=None)
    print(f"Strategy signal: {signal['action']}")
    if signal["action"] == "HOLD":
        print("No entry signal generated.")
        return 0

    equity_usd = args.equity
    slot_fraction = scenario.get("slot_fraction", 1 / 50)
    slot_equity = equity_usd * slot_fraction
    print(
        f"Total equity: ${equity_usd:.2f} | Slot equity ({slot_fraction*100:.2f}%): ${slot_equity:.2f} "
        f"| Risk per slot: {scenario['risk_percentage']*100:.0f}%"
    )
    if slot_equity <= 0:
        print("Slot equity calculated as zero; aborting.")
        return 1

    if args.live and exchange is None:
        print("Cannot place live orders without API keys.")
        return 1

    order_result = open_position(
        exchange=exchange,
        symbol=args.symbol,
        equity_usd=slot_equity,
        signal=signal,
        config=scenario,
        live=args.live,
    )
    print("Entry order:")
    print(order_result.entry_order)
    print("Stop-loss order:")
    print(order_result.stop_order)
    print("Take-profit order:")
    print(order_result.take_order)
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        scenario = get_scenario(args.scenario)
    except KeyError as e:
        print(e)
        return 1

    exchange = init_exchange(args.live)
    if args.live and exchange is None:
        return 1

    try:
        df = fetch_market_data(args=args, scenario=scenario, exchange=exchange)
    except RuntimeError as e:
        print(e)
        return 1

    if args.monitor_live:
        return run_live_chart_mode(df, scenario, args)
    if args.monitor:
        return run_monitor_mode(df, scenario, args)

    return run_single_cycle(df=df, scenario=scenario, args=args, exchange=exchange)


if __name__ == "__main__":
    sys.exit(main())
