"""
Multi-Ticker Oscillator Alert Scanner
--------------------------------------
Standalone program (separate from the backtester). Scans a list of
50-100+ tickers every hour, detects confirmed RSI/TSI swing lows and
highs (same extrema logic as the backtester), and pushes a notification
to your phone via ntfy.sh whenever a new confirmed signal appears.

No brokerage account, no auto-trading. This just tells you when to look.

Setup:
  1. In Discord: right-click the channel you want alerts in ->
     Edit Channel -> Integrations -> Webhooks -> New Webhook -> Copy
     Webhook URL.
  2. Paste that URL into DISCORD_WEBHOOK_URL below.
  3. Edit TICKERS below with your watchlist.
  4. Run: python ticker_alert_scanner.py
     Leave it running (on your laptop, or a cheap always-on machine
     like a Raspberry Pi / free-tier cloud VM if you want 24/7 coverage).

pip install yfinance scipy pandas numpy requests
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from scipy.signal import find_peaks

# ----------------------------------------------------------------------
# CONFIG - edit this section
# ----------------------------------------------------------------------

DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "PASTE_YOUR_WEBHOOK_URL_HERE")

TICKERS = [
    "TSLA", "AAPL", "NVDA", "MSFT", "AMZN", "GOOGL", "META", "NFLX",
    "AMD", "AVGO", "CRM", "ADBE", "JPM", "BAC", "XOM", "CVX",
    "SPY", "QQQ", "COIN", "PLTR",
]

OSCILLATOR = "tsi"        # "tsi" or "rsi"
PROMINENCE = 15.0         # swing "depth" required to count as a real extremum
DISTANCE = 5              # min bars between detected extrema
CONFIRM_BARS = 2          # bars to wait after extremum before trusting it

INTERVAL = "1h"           # bar timeframe
LOOKBACK_PERIOD = "1mo"   # how much history to pull each scan (needs enough
                          # bars for the oscillator + peak detection to warm up)
SCAN_EVERY_SECONDS = 3600  # 1 hour

STATE_FILE = Path("alert_state.json")   # tracks which bars we've already alerted on
BATCH_SIZE = 20            # tickers per yfinance download call (keeps requests reasonable)


# ----------------------------------------------------------------------
# Indicators (same as the backtester)
# ----------------------------------------------------------------------

def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def compute_tsi(close: pd.Series, long: int = 25, short: int = 13):
    diff = close.diff()
    abs_diff = diff.abs()

    def double_smooth(series, r, s):
        first = series.ewm(span=r, adjust=False).mean()
        return first.ewm(span=s, adjust=False).mean()

    smoothed_diff = double_smooth(diff, long, short)
    smoothed_abs_diff = double_smooth(abs_diff, long, short)
    tsi = 100 * (smoothed_diff / smoothed_abs_diff.replace(0, np.nan))
    return tsi.fillna(0)


def detect_extrema(osc: pd.Series, prominence: float, distance: int, confirm_bars: int):
    values = osc.values
    troughs, _ = find_peaks(-values, prominence=prominence, distance=distance)
    peaks, _ = find_peaks(values, prominence=prominence, distance=distance)

    n = len(values)
    buy_signal = pd.Series(False, index=osc.index)
    sell_signal = pd.Series(False, index=osc.index)

    for idx in troughs:
        confirm_idx = idx + confirm_bars
        if confirm_idx < n:
            buy_signal.iloc[confirm_idx] = True

    for idx in peaks:
        confirm_idx = idx + confirm_bars
        if confirm_idx < n:
            sell_signal.iloc[confirm_idx] = True

    return buy_signal, sell_signal


# ----------------------------------------------------------------------
# Notifications
# ----------------------------------------------------------------------

def send_alert(ticker: str, action: str, price: float, bar_time: pd.Timestamp):
    emoji = "🟢" if action == "BUY" else "🔴"
    color = 3066993 if action == "BUY" else 15158332   # green / red, Discord embed color

    embed = {
        "title": f"{emoji} {action} signal: {ticker}",
        "description": f"**{ticker}** {action} at ~${price:.2f}",
        "color": color,
        "fields": [
            {"name": "Bar time", "value": bar_time.strftime("%Y-%m-%d %H:%M"), "inline": True},
            {"name": "Oscillator", "value": OSCILLATOR.upper(), "inline": True},
        ],
    }

    try:
        requests.post(
            DISCORD_WEBHOOK_URL,
            json={"embeds": [embed]},
            timeout=10,
        )
        print(f"  -> Sent alert: {action} {ticker}")
    except requests.RequestException as e:
        print(f"  -> Failed to send alert for {ticker}: {e}")


# ----------------------------------------------------------------------
# State tracking (avoid re-alerting on the same bar every scan)
# ----------------------------------------------------------------------

def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


# ----------------------------------------------------------------------
# Data fetching in batches
# ----------------------------------------------------------------------

def fetch_batch(tickers: list) -> dict:
    """Download a batch of tickers in one call, return {ticker: DataFrame}."""
    data = yf.download(
        tickers, interval=INTERVAL, period=LOOKBACK_PERIOD,
        progress=False, auto_adjust=True, group_by="ticker", threads=True,
    )

    result = {}
    if len(tickers) == 1:
        df = data.dropna()
        if not df.empty:
            result[tickers[0]] = df
        return result

    for t in tickers:
        try:
            df = data[t].dropna()
            if not df.empty:
                result[t] = df
        except KeyError:
            continue
    return result


def fetch_all(tickers: list) -> dict:
    all_data = {}
    for i in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[i:i + BATCH_SIZE]
        try:
            all_data.update(fetch_batch(batch))
        except Exception as e:
            print(f"Batch {batch} failed: {e}")
    return all_data


# ----------------------------------------------------------------------
# Main scan loop
# ----------------------------------------------------------------------

def scan_once(state: dict):
    print(f"\nScanning {len(TICKERS)} tickers...")
    data = fetch_all(TICKERS)

    for ticker, df in data.items():
        close = df["Close"]
        if len(close) < 40:   # not enough bars to warm up the oscillator
            continue

        osc = compute_tsi(close) if OSCILLATOR == "tsi" else compute_rsi(close)
        buy_signal, sell_signal = detect_extrema(osc, PROMINENCE, DISTANCE, CONFIRM_BARS)

        last_time = df.index[-1]
        last_price = float(close.iloc[-1])

        # only care about signals on the most recent handful of bars
        # (covers the case the scanner missed a run or just started up)
        recent_window = 3
        recent_buys = buy_signal.iloc[-recent_window:]
        recent_sells = sell_signal.iloc[-recent_window:]

        last_alert_key = f"{ticker}"
        already_alerted_time = state.get(last_alert_key)

        for t, is_signal in recent_buys.items():
            if is_signal and str(t) != already_alerted_time:
                send_alert(ticker, "BUY", float(close.loc[t]), t)
                state[last_alert_key] = str(t)

        for t, is_signal in recent_sells.items():
            if is_signal and str(t) != already_alerted_time:
                send_alert(ticker, "SELL", float(close.loc[t]), t)
                state[last_alert_key] = str(t)

    save_state(state)
    print("Scan complete.")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--loop", action="store_true",
                         help="Run continuously, scanning every SCAN_EVERY_SECONDS. "
                              "Omit this flag for a single scan-and-exit run "
                              "(used by the GitHub Actions scheduled workflow).")
    args = parser.parse_args()

    print(f"Starting alert scanner for {len(TICKERS)} tickers")
    print(f"Notifications -> Discord webhook")

    state = load_state()

    if args.loop:
        print(f"Running continuously, scanning every {SCAN_EVERY_SECONDS} seconds\n")
        while True:
            try:
                scan_once(state)
            except Exception as e:
                print(f"Scan error: {e}")
            time.sleep(SCAN_EVERY_SECONDS)
    else:
        # Single run: for use with a scheduler (GitHub Actions cron,
        # cron on a VM, Task Scheduler, etc.) that handles the "every hour"
        # part externally.
        scan_once(state)


if __name__ == "__main__":
    main()
