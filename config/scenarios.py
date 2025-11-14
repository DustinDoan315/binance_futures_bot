"""
Configuration presets for different risk scenarios.

Each scenario is a dictionary of parameters consumed by the bot.  Adjusting
`risk_percentage` changes the fraction of equity you are willing to risk per
trade.  Increasing leverage amplifies both gains and losses.  The
`risk_reward_ratio` controls the relationship between the stop‑loss and
take‑profit distances.

Note: values here are deliberately set high to illustrate how risk tuning
changes the bot's behaviour.  They are **not** recommended for live
trading.
"""

SCENARIOS = {
    "safe": {
        "risk_percentage": 0.20,  # 20% of equity per trade
        "leverage": 3,
        "risk_reward_ratio": 2.0,  # take‑profit twice the stop‑loss distance
        "max_positions": 1,
        "ema_fast": 50,
        "ema_slow": 200,
        "atr_period": 14,
        "rsi_period": 14,
        "rsi_long_threshold": 55,
        "rsi_short_threshold": 45,
        "slot_fraction": 1 / 50,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
    },
    "neutral": {
        "risk_percentage": 0.40,  # 40% of equity per trade
        "leverage": 5,
        "risk_reward_ratio": 1.5,
        "max_positions": 1,
        "ema_fast": 30,
        "ema_slow": 100,
        "atr_period": 14,
        "rsi_period": 14,
        "rsi_long_threshold": 55,
        "rsi_short_threshold": 45,
        "slot_fraction": 1 / 50,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
    },
    "risky": {
        "risk_percentage": 0.75,  # 75% of equity per trade
        "leverage": 10,
        "risk_reward_ratio": 1.0,
        "max_positions": 1,
        "ema_fast": 20,
        "ema_slow": 50,
        "atr_period": 14,
        "rsi_period": 14,
        "rsi_long_threshold": 60,
        "rsi_short_threshold": 40,
        "slot_fraction": 1 / 50,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
    },
}

def get_scenario(name: str) -> dict:
    """Return the configuration dictionary for the given scenario.

    Args:
        name: One of "safe", "neutral" or "risky".

    Returns:
        A dictionary of configuration values.

    Raises:
        KeyError: if the scenario name is unknown.
    """
    key = name.lower()
    if key not in SCENARIOS:
        raise KeyError(f"Unknown scenario '{name}'. Available options: {list(SCENARIOS.keys())}")
    return SCENARIOS[key].copy()
