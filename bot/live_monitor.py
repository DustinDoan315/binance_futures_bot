"""
Interactive charting utilities for real-time monitoring (simulation mode).

This module animates price action and the simulated equity curve as the bot
walks forward through historical candles.  It is intended for educational
“test” runs where you want to visualize drawdowns and profits in real time
without connecting to the exchange.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .monitor import TradeRecord, enter_positions, exit_position

LEGEND_LOCATION = "upper left"
ANNOTATION_COORD_UNITS = "offset points"
POSITION_FLAT_TEXT = "Position: Flat"
SYMBOL_COLORS = ["#2ca02c", "#d62728", "#ff7f0e", "#1f77b4", "#9467bd", "#8c564b"]


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
        self.positions: Dict[str, Optional[dict]] = {"LONG": None, "SHORT": None}
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
        self.price_line, = self.ax_price.plot([], [], color=SYMBOL_COLORS[0], label="Close")
        self.equity_line, = self.ax_equity.plot([], [], color="#9467bd", label="Equity")
        self.tp_scatter = self.ax_price.scatter([], [], color="#2ca02c", marker="^", s=60, label="Take-profit")
        self.sl_scatter = self.ax_price.scatter([], [], color="#d62728", marker="v", s=60, label="Stop-loss")
        self.ax_price.set_ylabel("Price")
        self.ax_equity.set_ylabel("Equity ($)")
        self.ax_equity.set_xlabel("Time")
        self.ax_price.legend(loc=LEGEND_LOCATION)
        self.ax_price.grid(True, alpha=0.2)
        self.ax_equity.grid(True, alpha=0.2)
        self.ax_price.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        self.stats_text = self.fig.text(0.01, 0.98, "", ha="left", va="top")
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
            f"Balance: ${self.balance:,.2f} | Open PnL: ${self.current_open_pnl:,.2f} | "
            f"Total equity: ${total_value:,.2f}"
        )
        active = []
        for side, position in self.positions.items():
            if position is None:
                continue
            stop = position["stop_price"]
            take = position["take_price"]
            risk_amt = position.get("risk_amount", 0.0)
            rr = self.config.get("risk_reward_ratio", 1.0)
            active.append(
                f"{side} @ {position['entry_price']:.2f} | Stop {stop:.2f} | Target {take:.2f} | Risk ${risk_amt:,.2f} | RR {rr:.2f}"
            )
        self.position_text.set_text(" || ".join(active) if active else POSITION_FLAT_TEXT)

    def _annotate_price(self, timestamp: pd.Timestamp, price: float, text: str, color: str) -> None:
        annotation = self.ax_price.annotate(
            text,
            xy=(mdates.date2num(timestamp), price),
            xytext=(0, 10),
            textcoords=ANNOTATION_COORD_UNITS,
            color=color,
            fontsize=8,
            ha="center",
        )
        self.annotations.append(annotation)

    def _close_positions(self, row: pd.Series) -> bool:
        changed = False
        for side in ("LONG", "SHORT"):
            position = self.positions[side]
            if position is None:
                continue
            self.positions[side], trade, self.balance = exit_position(
                position=position,
                row=row,
                config=self.config,
                balance=self.balance,
            )
            if trade:
                target = self.tp_points if trade.outcome == "TAKE_PROFIT" else self.sl_points
                target.append((trade.exit_time, trade.exit_price))
                if self.equity_values:
                    self.equity_values[-1] = self.balance
                color = "#2ca02c" if trade.outcome == "TAKE_PROFIT" else "#d62728"
                self._annotate_price(
                    trade.exit_time,
                    trade.exit_price,
                    f"{trade.side} exit\n{trade.exit_price:.2f}",
                    color,
                )
                changed = True
        return changed

    def _compute_open_pnl(self, current_price: float) -> float:
        total = 0.0
        for position in self.positions.values():
            if position is None:
                continue
            total += _compute_open_pnl(
                position,
                current_price,
                self.config["risk_reward_ratio"],
            )
        return total

    def _handle_existing_positions(self, row: pd.Series) -> bool:
        changed = self._close_positions(row)
        self.current_open_pnl = self._compute_open_pnl(row["close"])
        return changed

    def _try_open_positions(self, row: pd.Series) -> None:
        window = self.df.loc[: row.name]
        if len(window) < 3:
            return
        occupied = {side for side, pos in self.positions.items() if pos is not None}
        new_positions = enter_positions(window, self.config, self.balance, occupied)
        for position in new_positions:
            self.positions[position["side"]] = position
            color = "#2ca02c" if position["side"] == "LONG" else "#d62728"
            self._annotate_price(
                position["entry_time"],
                position["entry_price"],
                f"{position['side']} entry\n{position['entry_price']:.2f}",
                color,
            )

    def _process_position(self, row: pd.Series) -> None:
        closed = self._handle_existing_positions(row)
        if not closed:
            self._try_open_positions(row)

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


class MultiSymbolLiveChartAnimator:
    """Animate stacked price panels plus aggregate equity for multiple symbols."""

    def __init__(
        self,
        data_map: Dict[str, pd.DataFrame],
        config: dict,
        initial_equity: float,
        delay: float = 0.5,
    ) -> None:
        if not data_map:
            raise ValueError("data_map must contain at least one symbol")
        self.symbols = list(data_map.keys())
        self.data_map = data_map
        self.config = config
        self.delay = max(0.05, delay)
        self.per_symbol_equity = initial_equity / max(1, len(self.symbols))

        rows = len(self.symbols) + 1
        height_ratios = [2] * len(self.symbols) + [1]
        plt.ion()
        self.fig, axes = plt.subplots(
            rows,
            1,
            figsize=(13, max(8, 3.2 * rows)),
            sharex=True,
            gridspec_kw={"height_ratios": height_ratios},
        )
        if isinstance(axes, np.ndarray):
            axes_list = axes.flatten().tolist()
        else:
            axes_list = [axes]
        if len(self.symbols) == 1 and len(axes_list) >= 2:
            axes_list = [axes_list[0], axes_list[1]]
        self.axes_price = axes_list[:-1]
        self.ax_equity = axes_list[-1]

        self.symbol_state: Dict[str, dict] = {}
        for idx, (symbol, ax) in enumerate(zip(self.symbols, self.axes_price)):
            state = {
                "df": data_map[symbol],
                "balance": self.per_symbol_equity,
                "positions": {"LONG": None, "SHORT": None},
                "price_times": [],
                "price_values": [],
                "equity_values": [],
                "tp_points": [],
                "sl_points": [],
                "current_open_pnl": 0.0,
            }
            line_color = SYMBOL_COLORS[idx % len(SYMBOL_COLORS)]
            line, = ax.plot([], [], color=line_color, label=f"{symbol} Close")
            tp_scatter = ax.scatter([], [], color="#2ca02c", marker="^", s=50, label="Take-profit")
            sl_scatter = ax.scatter([], [], color="#d62728", marker="v", s=50, label="Stop-loss")
            ax.set_ylabel(f"{symbol} Price")
            ax.grid(True, alpha=0.2)
            ax.legend(loc=LEGEND_LOCATION, fontsize="small")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            state["line"] = line
            state["line_color"] = line_color
            state["tp_scatter"] = tp_scatter
            state["sl_scatter"] = sl_scatter
            state["annotations"] = []
            state["status_text"] = ax.text(0.01, 0.95, POSITION_FLAT_TEXT, transform=ax.transAxes, va="top")
            self.symbol_state[symbol] = state

        self.ax_equity.set_ylabel("Aggregate Equity ($)")
        self.ax_equity.set_xlabel("Time")
        self.ax_equity.grid(True, alpha=0.2)
        self.ax_equity.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        (self.aggregate_line,) = self.ax_equity.plot([], [], color="#9467bd", label="Aggregate equity")
        self.ax_equity.legend(loc=LEGEND_LOCATION)
        self.stats_text = self.fig.text(0.01, 0.98, "", ha="left", va="top")
        self.aggregate_times: List[pd.Timestamp] = []
        self.aggregate_values: List[float] = []

    def _annotate(self, ax, timestamp: pd.Timestamp, price: float, text: str, color: str) -> None:
        annotation = ax.annotate(
            text,
            xy=(mdates.date2num(timestamp), price),
            xytext=(0, 8),
            textcoords=ANNOTATION_COORD_UNITS,
            color=color,
            fontsize=7,
            ha="center",
        )
        return annotation

    def _handle_existing_positions(self, symbol: str, row: pd.Series) -> bool:
        state = self.symbol_state[symbol]
        changed = False
        for side in ("LONG", "SHORT"):
            position = state["positions"][side]
            if position is None:
                continue
            state["positions"][side], trade, state["balance"] = exit_position(
                position=position,
                row=row,
                config=self.config,
                balance=state["balance"],
            )
            if trade:
                target_points = state["tp_points"] if trade.outcome == "TAKE_PROFIT" else state["sl_points"]
                target_points.append((trade.exit_time, trade.exit_price))
                color = "#2ca02c" if trade.outcome == "TAKE_PROFIT" else "#d62728"
                annotation = self._annotate(
                    ax=self._get_axis(symbol),
                    timestamp=trade.exit_time,
                    price=trade.exit_price,
                    text=f"{trade.side} exit\n{trade.exit_price:.2f}",
                    color=color,
                )
                state["annotations"].append(annotation)
                changed = True
        state["current_open_pnl"] = sum(
            _compute_open_pnl(position, row["close"], self.config["risk_reward_ratio"])
            for position in state["positions"].values()
            if position is not None
        )
        self._update_status_text(symbol)
        return changed

    def _get_axis(self, symbol: str):
        idx = self.symbols.index(symbol)
        return self.axes_price[idx]

    def _update_status_text(self, symbol: str) -> None:
        state = self.symbol_state[symbol]
        active = []
        rr = self.config.get("risk_reward_ratio", 1.0)
        for side, position in state["positions"].items():
            if position is None:
                continue
            stop = position["stop_price"]
            take = position["take_price"]
            risk_amt = position.get("risk_amount", 0.0)
            active.append(
                f"{side} @ {position['entry_price']:.2f} | Stop {stop:.2f} | Target {take:.2f} | Risk ${risk_amt:,.2f} | RR {rr:.2f}"
            )
        state["status_text"].set_text(" || ".join(active) if active else POSITION_FLAT_TEXT)

    def _try_open_positions(self, symbol: str, idx: int) -> None:
        state = self.symbol_state[symbol]
        if idx < 2:
            return
        window = state["df"].iloc[: idx + 1]
        occupied = {side for side, pos in state["positions"].items() if pos is not None}
        new_positions = enter_positions(window, self.config, state["balance"], occupied)
        for position in new_positions:
            state["positions"][position["side"]] = position
            annotation = self._annotate(
                ax=self._get_axis(symbol),
                timestamp=position["entry_time"],
                price=position["entry_price"],
                text=f"{position['side']} entry\n{position['entry_price']:.2f}",
                color=state["line_color"],
            )
            state["annotations"].append(annotation)
        self._update_status_text(symbol)

    def _update_symbol_lines(self, symbol: str) -> None:
        state = self.symbol_state[symbol]
        ax = self._get_axis(symbol)
        x_vals = mdates.date2num(state["price_times"])
        state["line"].set_data(x_vals, state["price_values"])
        tp_offsets = _to_offsets(state["tp_points"])
        sl_offsets = _to_offsets(state["sl_points"])
        state["tp_scatter"].set_offsets(tp_offsets)
        state["sl_scatter"].set_offsets(sl_offsets)
        ax.relim()
        ax.autoscale_view()

    def _update_stats_text(self, total_equity: float) -> None:
        parts = [
            f"{symbol}: ${(self.symbol_state[symbol]['balance'] + self.symbol_state[symbol]['current_open_pnl']):,.2f}"
            for symbol in self.symbols
        ]
        parts.append(f"Aggregate: ${total_equity:,.2f}")
        self.stats_text.set_text(" | ".join(parts))

    def _update_aggregate_line(self, timestamp: pd.Timestamp) -> None:
        total_equity = 0.0
        for state in self.symbol_state.values():
            total_equity += state["balance"] + state["current_open_pnl"]
        self.aggregate_times.append(timestamp)
        self.aggregate_values.append(total_equity)
        x_vals = mdates.date2num(self.aggregate_times)
        self.aggregate_line.set_data(x_vals, self.aggregate_values)
        self.ax_equity.relim()
        self.ax_equity.autoscale_view()
        self._update_stats_text(total_equity)

    def run(self) -> None:
        max_length = max(len(state["df"]) for state in self.symbol_state.values())
        for idx in range(max_length):
            latest_timestamp: Optional[pd.Timestamp] = None
            for symbol in self.symbols:
                state = self.symbol_state[symbol]
                if idx >= len(state["df"]):
                    continue
                row = state["df"].iloc[idx]
                latest_timestamp = row.name
                state["price_times"].append(row.name)
                state["price_values"].append(row["close"])
                state["equity_values"].append(state["balance"])
                closed = self._handle_existing_positions(symbol, row)
                if not closed:
                    self._try_open_positions(symbol, idx)
                self._update_symbol_lines(symbol)
            if latest_timestamp is not None:
                self._update_aggregate_line(latest_timestamp)
            plt.pause(self.delay)
        plt.ioff()
        plt.show()


def run_live_chart_multi(
    data_map: Dict[str, pd.DataFrame],
    config: dict,
    initial_equity: float,
    delay: float = 0.5,
) -> None:
    """Animate multiple symbols simultaneously with stacked panels."""

    animator = MultiSymbolLiveChartAnimator(data_map, config, initial_equity, delay=delay)
    animator.run()

