import sys
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from io import StringIO
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pytz

def print_current_pkt_time():
    tz = pytz.timezone('Asia/Karachi')
    now = datetime.now(tz)
    print(f"Current date/time: {now.strftime('%A, %B %d, %Y, %I:%M %p %Z')}")

def fetch_month(symbol, month, year):
    url = "https://dps.psx.com.pk/historical"
    payload = {"month": str(month), "year": str(year), "symbol": symbol}
    headers = {"User-Agent": "Mozilla/5.0"}

    response = requests.post(url, data=payload, headers=headers)
    if response.status_code != 200:
        print(f"Failed to fetch data for {symbol} {month}/{year}")
        return None

    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find("table", id="historicalTable")
    if not table:
        print(f"No data table found for {symbol} {month}/{year}")
        return None

    dfs = pd.read_html(StringIO(str(table)))
    if not dfs:
        print(f"Failed to parse table for {symbol} {month}/{year}")
        return None

    df = dfs[0]
    df.columns = df.columns.str.strip()
    df["DATE"] = pd.to_datetime(df["DATE"])
    for col in ["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def fetch_history(symbol, months=3):
    today = datetime.today()
    dfs = []
    for i in range(months):
        dt = today - relativedelta(months=i)
        month, year = dt.month, dt.year
        print(f"Fetching {symbol} data for {month}/{year} ...")
        df_month = fetch_month(symbol, month, year)
        if df_month is not None:
            print(f"  Rows fetched: {len(df_month)}")
            print(df_month.to_string(index=False))  # show full monthly data
            dfs.append(df_month)
        else:
            print(f"  No data found for {month}/{year}")
    if dfs:
        full_df = pd.concat(dfs).drop_duplicates(subset=['DATE']).reset_index(drop=True)
        full_df = full_df.sort_values('DATE').reset_index(drop=True)
        return full_df
    return None

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line

def calculate_sma(series, period=20):
    return series.rolling(window=period).mean()

def calculate_ema(series, period=20):
    return series.ewm(span=period, adjust=False).mean()

def calculate_bollinger_bands(series, period=20):
    sma = calculate_sma(series, period)
    std = series.rolling(period).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    return upper, lower

def calculate_stochastic_oscillator(df, k_period=14, d_period=3):
    low_min = df['LOW'].rolling(window=k_period, min_periods=k_period).min()
    high_max = df['HIGH'].rolling(window=k_period, min_periods=k_period).max()
    k = 100 * ((df['CLOSE'] - low_min) / (high_max - low_min))
    d = k.rolling(window=d_period, min_periods=d_period).mean()
    return k, d

def signal_from_rsi(rsi):
    if pd.isna(rsi): return 'NEUTRAL'
    if rsi < 30: return 'BUY'
    if rsi > 70: return 'SELL'
    return 'NEUTRAL'

def signal_from_macd(macd_line, signal_line):
    if len(macd_line) < 2 or len(signal_line) < 2: return 'NEUTRAL'
    if macd_line.iat[-2] < signal_line.iat[-2] and macd_line.iat[-1] > signal_line.iat[-1]: return 'BUY'
    if macd_line.iat[-2] > signal_line.iat[-2] and macd_line.iat[-1] < signal_line.iat[-1]: return 'SELL'
    return 'NEUTRAL'

def signal_from_sma_crossover(series, short_p=12, long_p=26):
    short_sma = calculate_sma(series, short_p)
    long_sma = calculate_sma(series, long_p)
    if len(short_sma) < 2 or len(long_sma) < 2 or pd.isna(short_sma.iat[-2]) or pd.isna(long_sma.iat[-2]): return 'NEUTRAL'
    if short_sma.iat[-2] < long_sma.iat[-2] and short_sma.iat[-1] > long_sma.iat[-1]: return 'BUY'
    if short_sma.iat[-2] > long_sma.iat[-2] and short_sma.iat[-1] < long_sma.iat[-1]: return 'SELL'
    return 'NEUTRAL'

def signal_from_ema_crossover(series, short_p=12, long_p=26):
    short_ema = calculate_ema(series, short_p)
    long_ema = calculate_ema(series, long_p)
    if len(short_ema) < 2 or len(long_ema) < 2 or pd.isna(short_ema.iat[-2]) or pd.isna(long_ema.iat[-2]): return 'NEUTRAL'
    if short_ema.iat[-2] < long_ema.iat[-2] and short_ema.iat[-1] > long_ema.iat[-1]: return 'BUY'
    if short_ema.iat[-2] > long_ema.iat[-2] and short_ema.iat[-1] < long_ema.iat[-1]: return 'SELL'
    return 'NEUTRAL'

def signal_from_bollinger(series):
    upper, lower = calculate_bollinger_bands(series)
    if pd.isna(upper.iat[-1]) or pd.isna(lower.iat[-1]): return 'NEUTRAL'
    price = series.iat[-1]
    if price < lower.iat[-1]: return 'BUY'
    if price > upper.iat[-1]: return 'SELL'
    return 'NEUTRAL'

def signal_from_stochastic(k, d):
    if len(k) < 2 or len(d) < 2 or pd.isna(k.iat[-2]) or pd.isna(d.iat[-2]): return 'NEUTRAL'
    if k.iat[-2] < d.iat[-2] and k.iat[-1] > d.iat[-1] and k.iat[-1] < 20: return 'BUY'
    if k.iat[-2] > d.iat[-2] and k.iat[-1] < d.iat[-1] and k.iat[-1] > 80: return 'SELL'
    return 'NEUTRAL'

def signal_from_volume(df):
    avg_vol = df['VOLUME'].rolling(window=20, min_periods=20).mean()
    if pd.isna(avg_vol.iat[-1]): return 'NEUTRAL'
    if df['VOLUME'].iat[-1] > 1.5 * avg_vol.iat[-1]: return 'BUY'
    return 'NEUTRAL'

def combine_signals(signals, weights):
    score = 0
    max_score = sum(weights.values())
    for ind, sig in signals.items():
        w = weights[ind]
        if sig == 'BUY': score += w
        elif sig == 'SELL': score -= w
    if max_score == 0: return 'NEUTRAL', 0.0
    confidence = abs(score)/max_score*100
    if score > 0: action = 'BUY'
    elif score < 0: action = 'SELL'
    else: action = 'NEUTRAL'
    return action, round(confidence, 2)

def main():
    print_current_pkt_time()

    if len(sys.argv) < 2:
        print("Usage: python PSX_Analyze.py SYMBOLS [months]")
        print("Example: python PSX_Analyze.py EFERT,MARI,MEBL,FABL 2")
        sys.exit(1)

    symbols_raw = sys.argv[1]
    symbols = [s.strip().upper() for s in symbols_raw.split(",") if s.strip()]
    months = 3
    if len(sys.argv) >= 3:
        try:
            months = int(sys.argv[2])
            if months <= 0:
                print("Months must be positive integer, defaulting to 3")
                months = 3
        except:
            print("Invalid months argument, defaulting to 3")
            months = 3

    multiple = len(symbols) > 1
    recommendations = []

    for symbol in symbols:
        print(f"\nFetching last {months} months data for {symbol}...\n")
        df = fetch_history(symbol, months)
        if df is None or df.empty:
            print(f"No data fetched for {symbol}. Skipping.\n")
            continue

        if len(df) < 30:
            print(f"Warning: Only {len(df)} trading days fetched for {symbol} — indicators may be unreliable.")

        close = df['CLOSE']
        rsi_series = calculate_rsi(close)
        macd_line, signal_line = calculate_macd(close)
        short_sma = calculate_sma(close, 12)
        long_sma = calculate_sma(close, 26)
        short_ema = calculate_ema(close, 12)
        long_ema = calculate_ema(close, 26)
        upper_band, lower_band = calculate_bollinger_bands(close)
        k, d = calculate_stochastic_oscillator(df)

        signals = {
            "RSI": signal_from_rsi(rsi_series.iat[-1]),
            "MACD": signal_from_macd(macd_line, signal_line),
            "SMA": signal_from_sma_crossover(close),
            "EMA": signal_from_ema_crossover(close),
            "Bollinger": signal_from_bollinger(close),
            "Stochastic": signal_from_stochastic(k, d),
            "Volume": signal_from_volume(df),
        }

        weights = {
            "RSI": 20,
            "MACD": 20,
            "SMA": 15,
            "EMA": 15,
            "Bollinger": 10,
            "Stochastic": 10,
            "Volume": 10,
        }

        action, confidence = combine_signals(signals, weights)
        latest_price = close.iat[-1]

        if multiple:
            recommendations.append((symbol, latest_price, action, confidence))
        else:
            # Single symbol detailed output:
            print("Latest indicator values:")
            print(f"  RSI: {rsi_series.iat[-1]:.2f}")
            print(f"  MACD: {macd_line.iat[-1]:.2f}, Signal Line: {signal_line.iat[-1]:.2f}")
            print(f"  SMA 12: {short_sma.iat[-1]:.2f}, SMA 26: {long_sma.iat[-1]:.2f}")
            print(f"  EMA 12: {short_ema.iat[-1]:.2f}, EMA 26: {long_ema.iat[-1]:.2f}")
            print(f"  Bollinger Upper: {upper_band.iat[-1]:.2f}, Lower: {lower_band.iat[-1]:.2f}, Price: {latest_price:.2f}")
            print(f"  Stochastic K: {k.iat[-1]:.2f}, D: {d.iat[-1]:.2f}")
            print(f"  Volume: {df['VOLUME'].iat[-1]}, Avg Vol(20): {df['VOLUME'].rolling(20).mean().iat[-1]:.2f}")

            print("\nStep-by-step signals and their weighted contributions:")
            for ind, sig in signals.items():
                w = weights[ind]
                contrib = w if sig == 'BUY' else -w if sig == 'SELL' else 0
                print(f"  {ind.ljust(10)} Signal: {sig.ljust(7)} Weight: {w} Contribution to score: {contrib}")

            print(f"\nFinal recommendation for {symbol} (PKR {latest_price:.2f}): ", end="")
            if action == 'NEUTRAL':
                print("NEUTRAL (Confidence too low)")
            else:
                print(f"{action} with confidence {confidence:.2f}%")

    if multiple and recommendations:
        print("\nFinal recommendations")
        for symbol, price, action, confidence in recommendations:
            if action == 'NEUTRAL':
                print(f"{symbol} (PKR {price:.2f}): NEUTRAL (Confidence too low)")
            else:
                print(f"{symbol} (PKR {price:.2f}): {action} with confidence {confidence:.2f}%")

if __name__ == "__main__":
    main()
