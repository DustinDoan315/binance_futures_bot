"""
Interactive charting utilities for real-time monitoring (simulation mode).

This module animates price action and the simulated equity curve as the bot
walks forward through historical candles.  It is intended for educational
“test” runs where you want to visualize drawdowns and profits in real time
without connecting to the exchange.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .monitor import TradeRecord, enter_position, exit_position


def _to_offsets(points: List[Tuple[pd.Timestamp, float]]) -> np.ndarray:
    if not points:
        return np.empty((0, 2))
    xs = [mdates.date2num(t) for t, _ in points]
    ys = [price for _, price in points]
    return np.column_stack([xs, ys])


def _compute_open_pnl(position: dict, current_price: float, risk_reward: float) -> float:
    stop_dist = max(position.get("stop_distance", 0.0), 1e-9)
    take_dist = max(position.get("take_distance", 0.0), 1e-9)
    entry_price = position["entry_price"]
    risk_amount = position["risk_amount"]
    if position["side"] == "LONG":
        if current_price >= entry_price:
            progress = min(1.0, (current_price - entry_price) / take_dist)
            return risk_amount * risk_reward * progress
        loss_ratio = min(1.0, (entry_price - current_price) / stop_dist)
        return -risk_amount * loss_ratio
    if current_price <= entry_price:
        progress = min(1.0, (entry_price - current_price) / take_dist)
        return risk_amount * risk_reward * progress
    loss_ratio = min(1.0, (current_price - entry_price) / stop_dist)
    return -risk_amount * loss_ratio


class LiveChartAnimator:
    def __init__(
        self,
        df: pd.DataFrame,
        config: dict,
        initial_equity: float,
        delay: float = 0.5,
    ) -> None:
        self.df = df
        self.config = config
        self.delay = max(0.05, delay)
        self.balance = initial_equity
        self.price_times: List[pd.Timestamp] = []
        self.price_values: List[float] = []
        self.equity_values: List[float] = []
        self.tp_points: List[Tuple[pd.Timestamp, float]] = []
        self.sl_points: List[Tuple[pd.Timestamp, float]] = []
        self.position: Optional[dict] = None
        self.current_open_pnl: float = 0.0
        self.open_trades: List[TradeRecord] = []
        self.annotations: List[plt.Annotation] = []
        plt.ion()
        self.fig, (self.ax_price, self.ax_equity) = plt.subplots(
            2,
            1,
            figsize=(13, 8),
            sharex=True,
            gridspec_kw={"height_ratios": [2, 1]},
        )
        self.price_line, = self.ax_price.plot([], [], color="#1f77b4", label="Close")
        self.equity_line, = self.ax_equity.plot([], [], color="#9467bd", label="Equity")
        self.tp_scatter = self.ax_price.scatter([], [], color="#2ca02c", marker="^", s=60, label="Take-profit")
        self.sl_scatter = self.ax_price.scatter([], [], color="#d62728", marker="v", s=60, label="Stop-loss")
        self.ax_price.set_ylabel("Price")
        self.ax_equity.set_ylabel("Equity ($)")
        self.ax_equity.set_xlabel("Time")
        self.ax_price.legend(loc="upper left")
        self.ax_price.grid(True, alpha=0.2)
        self.ax_equity.grid(True, alpha=0.2)
        self.ax_price.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        self.stats_text = self.ax_equity.text(0.01, 0.95, "", transform=self.ax_equity.transAxes, va="top")
        self.position_text = self.ax_price.text(0.01, 0.95, "", transform=self.ax_price.transAxes, va="top")

    def _update_lines(self) -> None:
        x_vals = mdates.date2num(self.price_times)
        self.price_line.set_data(x_vals, self.price_values)
        self.equity_line.set_data(x_vals[: len(self.equity_values)], self.equity_values)
        self.tp_scatter.set_offsets(_to_offsets(self.tp_points))
        self.sl_scatter.set_offsets(_to_offsets(self.sl_points))
        self.ax_price.relim()
        self.ax_price.autoscale_view()
        self.ax_equity.relim()
        self.ax_equity.autoscale_view()

    def _update_text(self) -> None:
        total_value = self.balance + self.current_open_pnl
        self.stats_text.set_text(
            f"Balance: ${self.balance:,.2f}\n"
            f"Open PnL: ${self.current_open_pnl:,.2f}\n"
            f"Total equity: ${total_value:,.2f}"
        )
        if self.position is None:
            self.position_text.set_text("Position: Flat")
        else:
            self.position_text.set_text(
                f"Position: {self.position['side']} @ {self.position['entry_price']:.2f}\n"
                f"Stop: {self.position['stop_price']:.2f}  Take: {self.position['take_price']:.2f}"
            )

    def _annotate_price(self, timestamp: pd.Timestamp, price: float, text: str, color: str) -> None:
        annotation = self.ax_price.annotate(
            text,
            xy=(mdates.date2num(timestamp), price),
            xytext=(0, 10),
            textcoords="offset points",
            color=color,
            fontsize=8,
            ha="center",
        )
        self.annotations.append(annotation)

    def _handle_existing_position(self, row: pd.Series) -> None:
        if self.position is None:
            self.current_open_pnl = 0.0
            return
        self.position, trade, self.balance = exit_position(
            position=self.position,
            row=row,
            config=self.config,
            balance=self.balance,
        )
        if trade:
            target = self.tp_points if trade.outcome == "TAKE_PROFIT" else self.sl_points
            target.append((trade.exit_time, trade.exit_price))
            self.equity_values[-1] = self.balance
            self.current_open_pnl = 0.0
            color = "#2ca02c" if trade.outcome == "TAKE_PROFIT" else "#d62728"
            self._annotate_price(
                trade.exit_time,
                trade.exit_price,
                f"{trade.side} exit\n{trade.exit_price:.2f}",
                color,
            )
            return
        if self.position is not None:
            self.current_open_pnl = _compute_open_pnl(
                self.position,
                row["close"],
                self.config["risk_reward_ratio"],
            )
        else:
            self.current_open_pnl = 0.0

    def _try_open_position(self, row: pd.Series) -> None:
        if self.position is not None:
            return
        window = self.df.loc[: row.name]
        if len(window) < 3:
            self.current_open_pnl = 0.0
            return
        self.position = enter_position(window, self.config, self.balance)
        if self.position is not None:
            self._annotate_price(
                self.position["entry_time"],
                self.position["entry_price"],
                f"{self.position['side']} entry\n{self.position['entry_price']:.2f}",
                "#1f77b4",
            )

    def _process_position(self, row: pd.Series) -> None:
        self._handle_existing_position(row)
        self._try_open_position(row)

    def run(self) -> None:
        for _, row in self.df.iterrows():
            self.price_times.append(row.name)
            self.price_values.append(row["close"])
            self.equity_values.append(self.balance)
            self._process_position(row)
            self._update_lines()
            self._update_text()
            plt.pause(self.delay)
        self.current_open_pnl = 0.0
        self._update_text()
        plt.ioff()
        plt.show()


def run_live_chart(df: pd.DataFrame, config: dict, initial_equity: float, delay: float = 0.5) -> None:
    """Animate a live chart for the given candle set."""

    animator = LiveChartAnimator(df, config, initial_equity, delay=delay)
    animator.run()

