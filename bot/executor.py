"""
Order execution utilities.

This module coordinates the creation of orders on Binance Futures.  It takes
signals produced by the strategy and, using the risk management module,
calculates the appropriate position size and exit prices.  It then calls
methods on the provided `BinanceExchange` instance to place orders.

For the purposes of this educational example, the `open_position` function
supports simulation: if `live=False`, it will simply return the computed
order parameters rather than calling the exchange.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from .exchange import BinanceExchange
from .risk import calculate_position_size, compute_exit_prices


@dataclass
class OrderResult:
    """Container for the results of an order placement.

    Attributes:
        entry_order: The response or details of the entry order.
        stop_order: The response or details of the stop‑loss order.
        take_order: The response or details of the take‑profit order.
    """

    entry_order: Dict[str, object]
    stop_order: Dict[str, object]
    take_order: Dict[str, object]


def _place_live_orders(
    exchange: BinanceExchange,
    symbol: str,
    side: str,
    position_side: str,
    qty: float,
    stop_price: float,
    take_price: float,
) -> Tuple[Dict[str, object], Dict[str, object], Dict[str, object]]:
    entry_resp = exchange.create_order(
        symbol=symbol,
        side=side,
        position_side=position_side,
        order_type="MARKET",
        quantity=qty,
    )
    stop_resp = exchange.create_order(
        symbol=symbol,
        side="SELL" if side == "BUY" else "BUY",
        position_side=position_side,
        order_type="STOP_MARKET",
        stop_price=stop_price,
        quantity=qty,
        reduce_only=True,
        close_position=False,
    )
    take_resp = exchange.create_order(
        symbol=symbol,
        side="SELL" if side == "BUY" else "BUY",
        position_side=position_side,
        order_type="TAKE_PROFIT_MARKET",
        stop_price=take_price,
        quantity=qty,
        reduce_only=True,
        close_position=False,
    )
    return entry_resp, stop_resp, take_resp


def _simulate_orders(
    symbol: str,
    side: str,
    position_side: str,
    qty: float,
    entry_price: float,
    stop_price: float,
    take_price: float,
) -> Tuple[Dict[str, object], Dict[str, object], Dict[str, object]]:
    entry_resp = {
        "symbol": symbol,
        "side": side,
        "positionSide": position_side,
        "type": "MARKET",
        "quantity": qty,
        "price": entry_price,
    }
    stop_resp = {
        "symbol": symbol,
        "side": "SELL" if side == "BUY" else "BUY",
        "positionSide": position_side,
        "type": "STOP_MARKET",
        "stopPrice": stop_price,
        "quantity": qty,
    }
    take_resp = {
        "symbol": symbol,
        "side": "SELL" if side == "BUY" else "BUY",
        "positionSide": position_side,
        "type": "TAKE_PROFIT_MARKET",
        "stopPrice": take_price,
        "quantity": qty,
    }
    return entry_resp, stop_resp, take_resp


def open_position(
    exchange: Optional[BinanceExchange],
    symbol: str,
    equity_usd: float,
    signal: Dict[str, object],
    config: Dict[str, object],
    live: bool = False,
) -> OrderResult:
    """Open a position based on a signal and risk configuration.

    Args:
        exchange: The BinanceExchange instance.
        symbol: Symbol to trade (e.g. 'BTCUSDT').
        equity_usd: Current account equity in USD.
        signal: Dictionary returned from strategy.generate_signal with keys
            action, entry_price, stop_distance, take_distance.
        config: Scenario configuration dictionary with keys including risk_percentage,
            leverage and risk_reward_ratio.
        live: If True, send orders to Binance; if False, return order details.

    Returns:
        OrderResult containing either exchange responses or simulated parameters.
    """
    action = signal.get("action")
    if action not in {"OPEN_LONG", "OPEN_SHORT"}:
        raise ValueError(f"Unsupported action {action}")
    side = "BUY" if action == "OPEN_LONG" else "SELL"
    position_side = "LONG" if action == "OPEN_LONG" else "SHORT"

    entry_price: float = signal["entry_price"]
    stop_distance: float = signal["stop_distance"]
    take_distance: float = signal["take_distance"]

    # Compute quantity based on risk
    qty = calculate_position_size(
        equity_usd=equity_usd,
        risk_percentage=config["risk_percentage"],
        stop_distance=stop_distance,
        entry_price=entry_price,
    )
    if qty <= 0:
        raise ValueError("Calculated quantity is zero or negative; check inputs")

    # Compute stop and take prices
    stop_price, take_price = compute_exit_prices(
        entry_price=entry_price,
        stop_distance=stop_distance,
        take_distance=take_distance,
        side=position_side,
    )

    # Round quantity to acceptable precision (Binance often requires 3 decimal places or more)
    qty = float(f"{qty:.6f}")

    if live:
        if exchange is None:
            raise RuntimeError("Exchange client is None but live trading requested")
        entry_resp, stop_resp, take_resp = _place_live_orders(
            exchange=exchange,
            symbol=symbol,
            side=side,
            position_side=position_side,
            qty=qty,
            stop_price=stop_price,
            take_price=take_price,
        )
    else:
        entry_resp, stop_resp, take_resp = _simulate_orders(
            symbol=symbol,
            side=side,
            position_side=position_side,
            qty=qty,
            entry_price=entry_price,
            stop_price=stop_price,
            take_price=take_price,
        )
    return OrderResult(entry_order=entry_resp, stop_order=stop_resp, take_order=take_resp)
