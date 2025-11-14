"""
Market data feed utilities.

For a production trading bot you would typically use Binance's WebSocket
streams to receive real‑time market data.  In this educational example we
provide a simple synchronous helper that downloads a batch of candlesticks.

You can extend this module to include asynchronous WebSocket streams using
the `websockets` library and Binance's public endpoints (e.g., wss://fstream.binance.com/stream).
"""

from __future__ import annotations

from typing import List, Tuple

import pandas as pd

from .exchange import BinanceExchange


def fetch_klines(
    exchange: BinanceExchange, symbol: str, interval: str = "15m", limit: int = 200
) -> pd.DataFrame:
    """Retrieve recent candlesticks and return them as a pandas DataFrame.

    Each kline returned by Binance is a list: [
        openTime, open, high, low, close, volume, closeTime,
        quoteAssetVolume, numberOfTrades, takerBuyBaseVolume,
        takerBuyQuoteVolume, ignore
    ].

    Args:
        exchange: An instance of BinanceExchange.
        symbol: Trading symbol, e.g. 'BTCUSDT'.
        interval: Candlestick interval (default '15m').
        limit: Number of candles to fetch (default 200).

    Returns:
        A DataFrame with columns: openTime, open, high, low, close, volume.
        Prices are floats and timestamps are pandas.Timestamp objects.
    """
    raw = exchange.get_klines(symbol, interval=interval, limit=limit)
    cols = [
        "openTime",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "closeTime",
        "quoteAssetVolume",
        "numberOfTrades",
        "takerBuyBaseVolume",
        "takerBuyQuoteVolume",
        "ignore",
    ]
    df = pd.DataFrame(raw, columns=cols)
    # Convert numeric columns to floats
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    # Convert times from milliseconds to datetime
    df["openTime"] = pd.to_datetime(df["openTime"], unit="ms")
    df.set_index("openTime", inplace=True)
    return df[["open", "high", "low", "close", "volume"]]
