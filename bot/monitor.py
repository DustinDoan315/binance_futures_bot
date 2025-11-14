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


def enter_position(
    window: pd.DataFrame,
    config: dict,
    balance: float,
) -> Optional[dict]:
    signal = generate_signal(window, config, current_position=None)
    if signal.get("action") not in {"OPEN_LONG", "OPEN_SHORT"}:
        return None
    side = "LONG" if signal["action"] == "OPEN_LONG" else "SHORT"
    slot_fraction = config.get("slot_fraction", 1 / 50)
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
        "entry_time": window.index[-1],
        "entry_price": signal["entry_price"],
        "side": side,
        "stop_price": stop_price,
        "take_price": take_price,
        "stop_distance": abs(signal["entry_price"] - stop_price),
        "take_distance": abs(take_price - signal["entry_price"]),
        "risk_amount": risk_amount,
    }


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
    position: Optional[dict] = None

    for idx in range(len(df)):
        row = df.iloc[idx]
        equity_curve.append(balance)

        if position is not None:
            position, trade, balance = exit_position(
                position=position,
                row=row,
                config=config,
                balance=balance,
            )
            if trade:
                trades.append(trade)
                equity_curve[-1] = balance
                continue

        if position is None and idx >= 2:
            window = df.iloc[: idx + 1]
            position = enter_position(window, config, balance)

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

    ax_price.plot(times, closes, label="Close", color="#1f77b4")
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
            textcoords="offset points",
            fontsize=8,
            ha="center",
            color=color,
        )
        ax_price.annotate(
            f"{trade.side} exit\n{trade.exit_price:.2f}",
            xy=(trade.exit_time, trade.exit_price),
            xytext=(0, -12),
            textcoords="offset points",
            fontsize=8,
            ha="center",
            color=color,
        )
    ax_price.set_ylabel("Price")
    ax_price.legend(loc="upper left", fontsize="small")
    ax_price.grid(True, alpha=0.2)

    ax_equity.plot(times[: len(equity_curve)], equity_curve, color="#9467bd")
    ax_equity.set_ylabel("Equity ($)")
    ax_equity.set_xlabel("Time")
    ax_equity.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path

