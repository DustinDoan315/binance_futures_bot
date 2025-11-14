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
from bot.monitor import (
    plot_monitoring_dashboard,
    plot_multi_symbol_dashboard,
    simulate_trading_session,
)
from bot.live_monitor import run_live_chart, run_live_chart_multi


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
    parser.add_argument(
        "--symbols",
        nargs="+",
        help="Trade multiple symbols simultaneously (overrides --symbol)",
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="15m",
        choices=["3m", "5m", "15m", "30m", "1h", "2h", "4h"],
        help="Kline interval for all symbols (choices: 3m,5m,15m,30m,1h,2h,4h)",
    )
    parser.add_argument(
        "--direction",
        type=str,
        default="both",
        choices=["long", "short", "both"],
        help="Force long-only, short-only, or both directions (default: both)",
    )
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
    symbol: str,
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
            df = fetch_klines(exchange, symbol, args.interval, limit=candle_limit)
        except Exception as e:
            if args.live:
                raise RuntimeError(f"Fatal: could not fetch klines for live trading: {e}") from e
            print(f"Error fetching klines: {e}. Falling back to simulated data.")
    if df is None:
        df = simulate_price_series(candle_limit, seed=args.seed)
    return df


def run_monitor_mode(
    data_map: dict[str, pd.DataFrame],
    scenario: dict,
    args,
    symbols: list[str],
) -> int:
    per_symbol_equity = args.equity / max(1, len(symbols))
    trades_map: dict[str, list] = {}
    equity_map: dict[str, list] = {}
    final_balances: dict[str, float] = {}

    for symbol in symbols:
        df = data_map[symbol]
        trades, equity_curve = simulate_trading_session(df, scenario, per_symbol_equity)
        trades_map[symbol] = trades
        equity_map[symbol] = equity_curve
        final_balance = equity_curve[-1] if equity_curve else per_symbol_equity
        final_balances[symbol] = final_balance

        print(f"\nSymbol: {symbol}")
        if trades:
            print("Trade log:")
            for trade in trades:
                print(
                    f"{trade.entry_time} -> {trade.exit_time} | {trade.side:<5} "
                    f"{trade.outcome:<11} | PnL: {trade.pnl:>8.2f} | Balance: {trade.balance_after:>8.2f}"
                )
        else:
            print("No trades were generated for this symbol.")

    if len(symbols) == 1:
        symbol = symbols[0]
        chart_path = plot_monitoring_dashboard(
            data_map[symbol], trades_map[symbol], equity_map[symbol], args.chart_file
        )
        final_equity = final_balances[symbol]
    else:
        ordered_data = {symbol: data_map[symbol] for symbol in symbols}
        ordered_trades = {symbol: trades_map[symbol] for symbol in symbols}
        ordered_equity = {symbol: equity_map[symbol] for symbol in symbols}
        chart_path = plot_multi_symbol_dashboard(ordered_data, ordered_trades, ordered_equity, args.chart_file)
        final_equity = sum(final_balances.values())

    print(f"\nMonitoring chart saved to {chart_path}")
    print(f"Starting balance: ${args.equity:.2f}")
    print(f"Ending balance:   ${final_equity:.2f}")
    print(f"Net PnL:          ${final_equity - args.equity:.2f}")
    return 0


def run_live_chart_mode(
    data_map: dict[str, pd.DataFrame],
    scenario: dict,
    args,
    symbols: list[str],
) -> int:
    if len(symbols) == 1:
        primary_symbol = symbols[0]
        symbol_equity = args.equity
        df = data_map[primary_symbol]
        print(f"Starting interactive live chart for {primary_symbol} (close the window to stop)...")
        run_live_chart(df, scenario, symbol_equity, delay=args.chart_delay)
    else:
        print(
            f"Starting multi-symbol live chart for {', '.join(symbols)} "
            "(close the window to stop)..."
        )
        run_live_chart_multi(data_map, scenario, args.equity, delay=args.chart_delay)
    return 0


def run_single_cycle(
    *,
    data_map: dict[str, pd.DataFrame],
    scenario: dict,
    args,
    exchange: Optional[BinanceExchange],
    symbols: list[str],
) -> int:
    slot_fraction = scenario.get("slot_fraction", 1 / 30)
    per_symbol_equity = args.equity / max(1, len(symbols))
    exit_code = 0

    for symbol in symbols:
        print(f"\n=== {symbol} ===")
        df = data_map[symbol]
        signal_bundle = generate_signal(df, scenario)
        actions = signal_bundle.get("actions", [])
        if not actions:
            print("Strategy signal: HOLD")
            print("No entry signal generated.")
            continue
        print(
            "Strategy actions: "
            + ", ".join(action["action"] for action in actions)
        )

        slot_equity = per_symbol_equity * slot_fraction
        print(
            f"Symbol equity share: ${per_symbol_equity:.2f} | Slot equity ({slot_fraction*100:.2f}%): ${slot_equity:.2f} "
            f"| Risk per slot: {scenario['risk_percentage']*100:.0f}%"
        )
        if slot_equity <= 0:
            print("Slot equity calculated as zero; skipping symbol.")
            exit_code = 1
            continue

        if args.live and exchange is None:
            print("Cannot place live orders without API keys.")
            return 1

        for action in actions:
            order_result = open_position(
                exchange=exchange,
                symbol=symbol,
                equity_usd=slot_equity,
                signal=action,
                config=scenario,
                live=args.live,
            )
            print(f"Action {action['action']}:")
            print("  Entry order:")
            print(f"    {order_result.entry_order}")
            print("  Stop-loss order:")
            print(f"    {order_result.stop_order}")
            print("  Take-profit order:")
            print(f"    {order_result.take_order}")
    return exit_code


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    symbols = [sym.upper() for sym in (args.symbols if args.symbols else [args.symbol])]

    try:
        scenario = get_scenario(args.scenario)
    except KeyError as e:
        print(e)
        return 1

    direction = args.direction.lower()
    if direction == "long":
        scenario["allow_long"] = True
        scenario["allow_short"] = False
    elif direction == "short":
        scenario["allow_long"] = False
        scenario["allow_short"] = True
    else:
        scenario["allow_long"] = True
        scenario["allow_short"] = True

    exchange = init_exchange(args.live)
    if args.live and exchange is None:
        return 1

    try:
        data_map = {
            symbol: fetch_market_data(args=args, scenario=scenario, exchange=exchange, symbol=symbol)
            for symbol in symbols
        }
    except RuntimeError as e:
        print(e)
        return 1

    if args.monitor_live:
        return run_live_chart_mode(data_map, scenario, args, symbols)
    if args.monitor:
        return run_monitor_mode(data_map, scenario, args, symbols)

    return run_single_cycle(data_map=data_map, scenario=scenario, args=args, exchange=exchange, symbols=symbols)


if __name__ == "__main__":
    sys.exit(main())
