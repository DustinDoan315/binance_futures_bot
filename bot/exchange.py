"""
Exchange client abstraction.

This module provides a minimal wrapper around the Binance Futures REST API.
It hides the details of signing requests and selecting the correct base URL
(testnet vs mainnet).  Only the endpoints needed for this educational bot
are implemented.  If you wish to support more endpoints, extend this class.

Warning: This client is intentionally incomplete and only meant for
educational use.  For production use consider `python-binance` or
`binance-connector`, which handle rate limiting, error codes and edge cases.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import time
import urllib.parse
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests


TESTNET_URL = "https://testnet.binancefuture.com"
MAINNET_URL = "https://fapi.binance.com"


def _get_api_keys() -> tuple[str, str]:
    """Read API credentials from environment variables.

    Returns:
        A tuple (api_key, api_secret).

    Raises:
        RuntimeError: if either environment variable is missing.
    """
    key = os.getenv("BINANCE_API_KEY")
    secret = os.getenv("BINANCE_API_SECRET")
    if not key or not secret:
        raise RuntimeError(
            "API key and secret must be set via environment variables BINANCE_API_KEY and BINANCE_API_SECRET"
        )
    return key, secret


@dataclass
class BinanceExchange:
    """Simplified Binance Futures client.

    Attributes:
        base_url: The REST endpoint to call (testnet or mainnet).
        require_api_keys: If True, instantiation will fail when credentials are missing.
        api_key/api_secret: Optional explicit credentials.  When omitted, they are
            loaded from the environment.
        session: A persistent requests.Session for connection pooling.
    """

    base_url: str = TESTNET_URL
    require_api_keys: bool = False
    api_key: Optional[str] = None
    api_secret: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.api_key or not self.api_secret:
            try:
                self.api_key, self.api_secret = _get_api_keys()
            except RuntimeError:
                if self.require_api_keys:
                    raise
                self.api_key = None
                self.api_secret = None
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({"X-MBX-APIKEY": self.api_key})

    def _ensure_credentials(self) -> None:
        if not self.api_key or not self.api_secret:
            raise RuntimeError("Binance API credentials are required for this operation")

    def _sign_params(self, params: Dict[str, Any]) -> str:
        """Generate the HMAC SHA256 signature required by Binance.

        Args:
            params: Dictionary of query parameters (without the signature).

        Returns:
            The hexadecimal signature string.
        """
        self._ensure_credentials()
        query_string = urllib.parse.urlencode(params, quote_via=urllib.parse.quote)
        signature = hmac.new(
            self.api_secret.encode("utf-8"), query_string.encode("utf-8"), hashlib.sha256
        ).hexdigest()
        return signature

    def _request(self, method: str, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Send a signed request to the Binance Futures API.

        Args:
            method: HTTP method (GET, POST, DELETE).
            path: API path (e.g., '/fapi/v1/klines').
            params: Query parameters.

        Returns:
            The JSON response from Binance.

        Raises:
            requests.HTTPError: for HTTP errors or non‑200 responses.
        """
        if params is None:
            params = {}
        # Add timestamp and signature
        self._ensure_credentials()
        params["timestamp"] = int(time.time() * 1000)
        params["recvWindow"] = 5000
        signature = self._sign_params(params)
        params["signature"] = signature
        url = f"{self.base_url}{path}"
        resp = self.session.request(method, url, params=params)
        resp.raise_for_status()
        return resp.json()

    # -- Public and private API wrappers --

    def get_server_time(self) -> Any:
        """Ping the server and return its time.
        Useful for connectivity checks.
        """
        url = f"{self.base_url}/fapi/v1/time"
        resp = self.session.get(url)
        resp.raise_for_status()
        return resp.json()

    def get_klines(self, symbol: str, interval: str = "1m", limit: int = 200) -> Any:
        """Fetch recent candlestick data.

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT').
            interval: Candlestick interval (e.g., '1m', '5m').
            limit: Number of candles to retrieve (maximum 1500 for daily intervals).

        Returns:
            List of klines, where each kline is a list: [openTime, open, high, low, close, volume, ...].
        """
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        }
        url = f"{self.base_url}/fapi/v1/klines"
        resp = self.session.get(url, params=params)
        resp.raise_for_status()
        return resp.json()

    def get_account_info(self) -> Any:
        """Return account balances and positions.
        Requires a signed request.
        """
        self._ensure_credentials()
        return self._request("GET", "/fapi/v2/account", {})

    def create_order(
        self,
        symbol: str,
        side: str,
        position_side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: Optional[str] = None,
        reduce_only: bool = False,
        close_position: bool = False,
    ) -> Any:
        """Place an order on Binance Futures.

        Args:
            symbol: Trading pair.
            side: BUY or SELL.
            positionSide: LONG or SHORT.
            order_type: MARKET, LIMIT, STOP_MARKET, TAKE_PROFIT_MARKET, etc.
            quantity: Quantity in base asset.
            price: Price for limit orders.
            stopPrice: Trigger price for stop orders.
            timeInForce: FOK, GTC, IOC for limit orders.
            reduceOnly: If True, ensures the order only reduces a position.
            closePosition: If True, closes the entire position (used for STOP_MARKET and TAKE_PROFIT_MARKET).

        Returns:
            JSON response from Binance.
        """
        self._ensure_credentials()
        params = {
            "symbol": symbol,
            "side": side,
            "positionSide": position_side,
            "type": order_type,
            "quantity": quantity,
            "reduceOnly": reduce_only,
            "closePosition": close_position,
        }
        if price is not None:
            params["price"] = price
        if stop_price is not None:
            params["stopPrice"] = stop_price
        if time_in_force is not None:
            params["timeInForce"] = time_in_force
        return self._request("POST", "/fapi/v1/order", params)
