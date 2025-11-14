# Binance Futures Trading Bot – Risk Scenarios

This repository contains a simple prototype of a Binance Futures trading bot written in Python.  
The goal is **education** and to demonstrate how different levels of risk can be encoded into a trading system.  
It **does not** place live trades out‑of‑the‑box, and you should never connect it to a real account without a deep understanding of the code and the risks involved.

## About the bot

The bot implements a basic **trend‑following strategy** with a pull‑back entry.  
Every time a new candle closes it:

1. Calculates short‑ and long‑term exponential moving averages (EMAs) and the average true range (ATR) over the most recent data.
2. Computes a 14‑period **RSI** plus a **Stochastic oscillator** (14/3) to confirm momentum and overbought/oversold states.
3. Uses these indicators to decide whether the market is trending up, trending down or ranging and generates a **signal** (open long, open short or stay flat) only when price resumes trending after a small pull‑back that aligns with the RSI and Stochastic filters.
4. Calculates position size based on your account equity and a **risk percentage**.
5. Splits the equity into **50 equal slots** and only risks one slot per order, so repeated signals never tie up the entire balance.
6. Creates paired stop‑loss and take‑profit orders to cap downside and lock in profits.

The bot architecture has been split into separate modules:

| Module                | Purpose                                                       |
| --------------------- | ------------------------------------------------------------- |
| `bot/exchange.py`     | Wraps the Binance Futures REST API (connects to testnet by default). |
| `bot/data_feed.py`    | Handles market data (klines) and user data streams.          |
| `bot/strategy.py`     | Contains the EMA/ATR‑based trend‑following strategy.         |
| `bot/risk.py`         | Computes position size, stop‑loss and take‑profit levels based on risk percentage. |
| `bot/executor.py`     | Sends orders to Binance and manages open positions.           |
| `config/scenarios.py` | Defines three risk profiles (safe, neutral and risky) and common settings. |
| `run_bot.py`          | Entrypoint script that wires everything together.            |

## Risk scenarios

This repository shows how a single strategy can be run with three different risk profiles.  
The **risk percentage** defines the fraction of your trading capital you are willing to lose **per trade**.  
Smaller values lead to smaller positions and lower potential drawdowns; larger values quickly amplify both gains and losses.  
Financial educators often recommend risking **1–2 %** of capital per trade【436900721831628†L80-L90】, so the `neutral` and `risky` scenarios below are for illustration only and would be considered extremely aggressive by most traders.

### Safe scenario — 20 % risk

* **Risk percentage:** 0.20 (20 % of account equity per trade).  
    Although still far above the 1–2 % recommended by experienced traders, this is the lowest risk of the three scenarios.  
* **Leverage:** capped at **3×**.  
    Binance’s own educational articles advise starting with low leverage (1×–3×) for futures bots【973641807447532†L60-L93】.  
* **Risk‑to‑reward ratio:** 1:2 (the take‑profit distance is twice the stop‑loss distance).  
* **Max open positions:** 1 per symbol.  
    New signals are ignored if a position is already open.

### Neutral scenario — 40 % risk

* **Risk percentage:** 0.40 (40 % of account equity per trade).  
    This doubles the position size relative to the safe scenario.  
* **Leverage:** capped at **5×**.  
    Still within the “low leverage” recommendations【973641807447532†L60-L93】, but high enough to materially increase the stakes.  
* **Risk‑to‑reward ratio:** 1:1.5 (slightly tighter take‑profit relative to stop‑loss).  
* **Max open positions:** 1 per symbol.  
    Signals are more frequent because risk tolerance is higher.

### Risky scenario — 75 % risk

* **Risk percentage:** 0.75 (75 % of account equity per trade).  
    This is a **very aggressive** setting; a few losing trades could wipe out the account.  
    It is documented here purely as a cautionary example.  
* **Leverage:** capped at **10×**.  
    Still well below the 125× maximum available on Binance Futures, but far beyond what novices should use【973641807447532†L60-L93】.  
* **Risk‑to‑reward ratio:** 1:1 (stop‑loss and take‑profit distances are equal).  
    This profile aims for quick, high‑risk trades with minimal margin for error.  
* **Max open positions:** 1 per symbol.  
    Entry signals are taken whenever the strategy indicates a trend continuation.

## Important warnings

* **Do not trade with money you can’t afford to lose.**  
  Even the safest scenario presented here (20 % risk) far exceeds professional risk management guidelines like the **1–2 % rule**【436900721831628†L80-L90】.
* **Always use the Binance testnet** when experimenting with automated strategies.  
  Live trading should only occur after extensive backtesting and paper trading on historical data【851120296267963†L339-L350】.
* **Set stop‑losses and take‑profits.**  
  A bot should never run without pre‑defined exit levels【436900721831628†L99-L114】.
* **Monitor and adjust your bot.**  
  Automated systems are not fire‑and‑forget; you must track their performance and adapt risk management rules as market conditions change【851120296267963†L339-L350】.
* **This repository is for educational purposes only** and does not constitute financial advice.

## Getting started

1. **Install dependencies:** The code uses `requests`, `websockets`, `pandas`, and `matplotlib` (for charts).  
   You can install them with pip:

   ```bash
   pip install requests websockets pandas matplotlib python-binance
   ```

   The `python-binance` package can be swapped for any other library; in `bot/exchange.py` we illustrate how to use the official `binance-connector` client but do not import it to avoid unnecessary dependencies.
2. **Configure your API keys:** Create a `.env` file or export environment variables `BINANCE_API_KEY` and `BINANCE_API_SECRET`.  
   Ensure the keys have **read** and **trade** permissions but **no withdrawal rights**.  
   > API keys are only required when you want to place live/signed requests.  Data downloads and simulations can run without them.
3. **Run in test mode:** Start the bot using one of the predefined scenarios:

   ```bash
   python run_bot.py --scenario safe
   ```

   Replace `safe` with `neutral` or `risky` to try the other risk profiles.  The script will print out simulated signals and order parameters rather than placing real trades.  Extra CLI flags:

   * `--equity 2500` – override the default $1k sizing capital.
   * `--limit 720` – request or simulate more candles for longer lookbacks.
   * `--offline` – skip all network calls and rely on synthetic candles.
   * `--monitor` – simulate multiple trades, keep an in-memory balance, and render a price/equity chart.
   * `--chart-file monitor.png` – choose where the monitoring chart is saved (defaults to `monitor_report.png`).
   * `--monitor-live` – stream the candles in “real time” with an interactive chart that shows running PnL.
   * `--chart-delay 0.25` – control how fast the live chart updates (defaults to 0.5 s between candles).
   * `--days 1` – automatically request enough candles to cover a full day at your chosen interval (e.g., 96 bars for 15 m).
   * Every order automatically uses 1/50 of the configured equity, so you can keep firing signals without over-allocating capital.

When `--monitor` is enabled the bot prints a trade-by-trade PnL log, shows the running “money flow” (simulated balance), and saves a PNG chart containing both the price action and the equity curve so you can visually track performance.

`--monitor-live` opens an interactive matplotlib window that animates the price and equity curves while also displaying the current balance, open PnL, stop/target levels, and inline labels for every entry/exit price.  Close the window (or press Ctrl+C in the terminal) to end the session.

To replay “one day after today” on the 15 m chart, run:

```bash
python run_bot.py --scenario safe --symbol BTCUSDT --interval 15m --days 1 --offline --monitor-live
```

## Extending the bot

The current implementation is intentionally simple.  Some ideas for future enhancements include:

* Adding a **backtesting engine** to evaluate strategies on historical data before risking capital.
* Incorporating a **dashboard** (e.g., using FastAPI and Plotly) to visualize equity curves, open positions and performance metrics in real time.
* Supporting **multiple symbols**, each with independent position sizing and risk controls.
* Integrating **machine learning models** for signal generation once you are comfortable with the basics.
