import yfinance as yf
import pandas as pd
import numpy as np
import ta
from .config import TRAIN_PERIOD_DAILY, TARGET_HORIZON_D

def dl_daily(symbol: str, period: str = TRAIN_PERIOD_DAILY) -> pd.DataFrame:
    df = yf.download(symbol, period=period, interval="1d", auto_adjust=False, progress=False)
    return df if df is not None and not df.empty else pd.DataFrame()

def add_daily_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add RSI, MACD, MAs, Bollinger, etc."""
    if df.empty: return df
    df = df.copy()
    c, v = df["Close"], df["Volume"]

    for k in (1,2,3,5,10,20):
        df[f"ret_{k}"] = c.pct_change(k)

    df["ma20"] = c.rolling(20).mean()
    df["ma50"] = c.rolling(50).mean()
    df["rsi14"] = ta.momentum.RSIIndicator(c, window=14).rsi()
    macd = ta.trend.MACD(c)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()

    bb = ta.volatility.BollingerBands(c, 20, 2)
    df["bb_pct"] = bb.bollinger_pband()
    df["vol_ma20"] = v.rolling(20).mean()
    df["vol_ratio20"] = v / df["vol_ma20"]

    df["fwd_ret_3d"] = c.shift(-TARGET_HORIZON_D)/c - 1
    df["y_up_3d"] = (df["fwd_ret_3d"] > 0).astype(int)
    return df
