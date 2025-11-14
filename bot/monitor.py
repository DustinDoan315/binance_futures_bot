"""
Monitoring and simulation utilities for the trading bot.

This module provides helpers to backtest the strategy on a batch of historical
candles, track an in-memory account balance, and visualize both the executed
trades and the resulting equity curve.  The simulation is intentionally simple:
positions are entered with market orders and exit immediately when either the
stop-loss or take-profit level is reached.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd

from .risk import compute_exit_prices
from .strategy import generate_signal


ANNOTATION_COORD_UNITS = "offset points"
LEGEND_LOCATION = "upper left"
SYMBOL_COLORS = ["#2ca02c", "#d62728", "#ff7f0e", "#1f77b4", "#9467bd", "#8c564b"]


@dataclass
class TradeRecord:
    """Represents a simulated trade outcome."""

    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    side: str
    outcome: str
    entry_price: float
    exit_price: float
    pnl: float
    balance_after: float


def check_exit_conditions(
    row: pd.Series,
    side: str,
    stop_price: float,
    take_price: float,
) -> Tuple[Optional[str], Optional[float]]:
    """Determine whether stop-loss or take-profit has been hit on this bar."""

    if side == "LONG":
        if row["low"] <= stop_price:
            return "STOP_LOSS", stop_price
        if row["high"] >= take_price:
            return "TAKE_PROFIT", take_price
    else:
        if row["high"] >= stop_price:
            return "STOP_LOSS", stop_price
        if row["low"] <= take_price:
            return "TAKE_PROFIT", take_price
    return None, None


def _create_position_from_signal(
    *,
    signal: dict,
    config: dict,
    balance: float,
    entry_time: pd.Timestamp,
) -> Optional[dict]:
    side = "LONG" if signal["action"] == "OPEN_LONG" else "SHORT"
    slot_fraction = config.get("slot_fraction", 1 / 30)
    slot_equity = balance * slot_fraction
    if slot_equity <= 0:
        return None
    risk_amount = slot_equity * config["risk_percentage"]
    if risk_amount <= 0:
        return None
    stop_price, take_price = compute_exit_prices(
        entry_price=signal["entry_price"],
        stop_distance=signal["stop_distance"],
        take_distance=signal["take_distance"],
        side=side,
    )
    return {
        "entry_time": entry_time,
        "entry_price": signal["entry_price"],
        "side": side,
        "stop_price": stop_price,
        "take_price": take_price,
        "stop_distance": abs(signal["entry_price"] - stop_price),
        "take_distance": abs(take_price - signal["entry_price"]),
        "risk_amount": risk_amount,
    }


def _exit_open_positions(
    *,
    positions: dict[str, Optional[dict]],
    row: pd.Series,
    config: dict,
    balance: float,
) -> tuple[float, List[TradeRecord], bool]:
    trades: List[TradeRecord] = []
    updated = False
    for side in ("LONG", "SHORT"):
        position = positions.get(side)
        if position is None:
            continue
        positions[side], trade, balance = exit_position(
            position=position,
            row=row,
            config=config,
            balance=balance,
        )
        if trade:
            trades.append(trade)
            updated = True
    return balance, trades, updated


def enter_positions(
    window: pd.DataFrame,
    config: dict,
    balance: float,
    occupied_sides: set[str],
) -> List[dict]:
    bundle = generate_signal(window, config)
    actions = bundle.get("actions", [])
    if not actions:
        return []
    new_positions: List[dict] = []
    entry_time = window.index[-1]
    for action in actions:
        side = "LONG" if action["action"] == "OPEN_LONG" else "SHORT"
        if side in occupied_sides:
            continue
        position = _create_position_from_signal(
            signal=action,
            config=config,
            balance=balance,
            entry_time=entry_time,
        )
        if position is not None:
            new_positions.append(position)
            occupied_sides.add(side)
    return new_positions


def exit_position(
    *,
    position: dict,
    row: pd.Series,
    config: dict,
    balance: float,
) -> Tuple[Optional[dict], Optional[TradeRecord], float]:
    outcome, exit_price = check_exit_conditions(
        row=row,
        side=position["side"],
        stop_price=position["stop_price"],
        take_price=position["take_price"],
    )
    if not outcome:
        return position, None, balance
    pnl = (
        -position["risk_amount"]
        if outcome == "STOP_LOSS"
        else position["risk_amount"] * config["risk_reward_ratio"]
    )
    balance += pnl
    trade = TradeRecord(
        entry_time=position["entry_time"],
        exit_time=row.name,
        side=position["side"],
        outcome=outcome,
        entry_price=position["entry_price"],
        exit_price=exit_price or row["close"],
        pnl=pnl,
        balance_after=balance,
    )
    return None, trade, balance


def simulate_trading_session(
    df: pd.DataFrame,
    config: dict,
    initial_equity: float,
) -> Tuple[List[TradeRecord], List[float]]:
    """Run a simple trade-by-trade simulation over the provided candles."""

    balance = initial_equity
    trades: List[TradeRecord] = []
    equity_curve: List[float] = []
    positions: dict[str, Optional[dict]] = {"LONG": None, "SHORT": None}

    for idx in range(len(df)):
        row = df.iloc[idx]
        equity_curve.append(balance)

        balance, closed_trades, updated = _exit_open_positions(
            positions=positions,
            row=row,
            config=config,
            balance=balance,
        )
        if closed_trades:
            trades.extend(closed_trades)
            equity_curve[-1] = balance
        if updated:
            continue

        if idx >= 2:
            window = df.iloc[: idx + 1]
            occupied = {side for side, pos in positions.items() if pos is not None}
            for new_position in enter_positions(window, config, balance, occupied):
                positions[new_position["side"]] = new_position

    return trades, equity_curve


def plot_monitoring_dashboard(
    df: pd.DataFrame,
    trades: List[TradeRecord],
    equity_curve: List[float],
    output_path: str,
) -> str:
    """Render a two-panel chart (price + equity curve) and save it to disk."""

    import matplotlib.pyplot as plt

    times = df.index
    closes = df["close"]

    fig, (ax_price, ax_equity) = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )

    ax_price.plot(times, closes, label="Close", color=SYMBOL_COLORS[0])
    for trade in trades:
        color = "#2ca02c" if trade.outcome == "TAKE_PROFIT" else "#d62728"
        marker = "^" if trade.side == "LONG" else "v"
        ax_price.scatter(
            trade.entry_time,
            trade.entry_price,
            color=color,
            marker=marker,
            s=60,
            label=f"{trade.side} {trade.outcome}",
        )
        ax_price.annotate(
            f"{trade.side} entry\n{trade.entry_price:.2f}",
            xy=(trade.entry_time, trade.entry_price),
            xytext=(0, 10),
            textcoords=ANNOTATION_COORD_UNITS,
            fontsize=8,
            ha="center",
            color=color,
        )
        ax_price.annotate(
            f"{trade.side} exit\n{trade.exit_price:.2f}",
            xy=(trade.exit_time, trade.exit_price),
            xytext=(0, -12),
            textcoords=ANNOTATION_COORD_UNITS,
            fontsize=8,
            ha="center",
            color=color,
        )
    ax_price.set_ylabel("Price")
    ax_price.legend(loc=LEGEND_LOCATION, fontsize="small")
    ax_price.grid(True, alpha=0.2)

    ax_equity.plot(times[: len(equity_curve)], equity_curve, color="#9467bd")
    ax_equity.set_ylabel("Equity ($)")
    ax_equity.set_xlabel("Time")
    ax_equity.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _aggregate_equity_series(equity_series_map: dict[str, pd.Series]) -> pd.Series:
    combined_index: Optional[pd.Index] = None
    for series in equity_series_map.values():
        combined_index = series.index if combined_index is None else combined_index.union(series.index)
    if combined_index is None or combined_index.empty:
        raise ValueError("No equity series available for aggregation")
    aggregate = pd.Series(0.0, index=combined_index, dtype=float)
    for series in equity_series_map.values():
        expanded = series.reindex(combined_index, method="ffill")
        expanded = expanded.bfill()
        expanded = expanded.ffill()
        aggregate = aggregate.add(expanded, fill_value=0.0)
    return aggregate


def plot_multi_symbol_dashboard(
    data_map: dict[str, pd.DataFrame],
    trades_map: dict[str, List[TradeRecord]],
    equity_map: dict[str, List[float]],
    output_path: str,
) -> str:
    """Render stacked charts for multiple symbols plus an aggregate equity curve."""

    import matplotlib.pyplot as plt

    if not data_map:
        raise ValueError("No data provided for multi-symbol dashboard")
    symbols = list(data_map.keys())
    rows = len(symbols) + 1
    height_ratios = [2] * len(symbols) + [1]
    fig, axes = plt.subplots(
        rows,
        1,
        figsize=(14, max(6, 3.5 * rows)),
        sharex=False,
        gridspec_kw={"height_ratios": height_ratios},
    )

    equity_series_map: dict[str, pd.Series] = {}
    for idx, symbol in enumerate(symbols):
        df = data_map[symbol]
        eq_values = equity_map.get(symbol, [])
        eq_index = df.index[: len(eq_values)]
        series = pd.Series(eq_values, index=eq_index, dtype=float)
        equity_series_map[symbol] = series
    aggregate_equity = _aggregate_equity_series(equity_series_map)

    for idx, symbol in enumerate(symbols):
        ax_price = axes[idx]
        df = data_map[symbol]
        trades = trades_map.get(symbol, [])
        color = SYMBOL_COLORS[idx % len(SYMBOL_COLORS)]
        ax_price.plot(df.index, df["close"], label=f"{symbol} Close", color=color)
        for trade in trades:
            color = "#2ca02c" if trade.outcome == "TAKE_PROFIT" else "#d62728"
            marker = "^" if trade.side == "LONG" else "v"
            ax_price.scatter(trade.entry_time, trade.entry_price, color=color, marker=marker, s=50)
            ax_price.annotate(
                f"{trade.side} entry\n{trade.entry_price:.2f}",
                xy=(trade.entry_time, trade.entry_price),
                xytext=(0, 8),
                textcoords=ANNOTATION_COORD_UNITS,
                fontsize=7,
                ha="center",
                color=color,
            )
            ax_price.annotate(
                f"{trade.side} exit\n{trade.exit_price:.2f}",
                xy=(trade.exit_time, trade.exit_price),
                xytext=(0, -10),
                textcoords=ANNOTATION_COORD_UNITS,
                fontsize=7,
                ha="center",
                color=color,
            )
        ax_price.set_ylabel(f"{symbol} Price")
        ax_price.grid(True, alpha=0.2)
        ax_price.legend(loc=LEGEND_LOCATION, fontsize="small")

    ax_equity = axes[-1]
    ax_equity.plot(aggregate_equity.index, aggregate_equity.values, color="#9467bd", label="Aggregate equity")
    ax_equity.set_ylabel("Equity ($)")
    ax_equity.set_xlabel("Time")
    ax_equity.grid(True, alpha=0.2)
    ax_equity.legend(loc=LEGEND_LOCATION)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path

