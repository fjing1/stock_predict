import yfinance as yf
import pandas as pd
import numpy as np
from .config import INTRADAY_INTERVAL, INTRADAY_PERIOD, DEFAULT_RTH

def dl_intraday_7d(symbol: str, interval=INTRADAY_INTERVAL, rth_only=DEFAULT_RTH) -> pd.DataFrame:
    df = yf.download(symbol, period=INTRADAY_PERIOD, interval=interval, progress=False)
    if df is None or df.empty: return pd.DataFrame()
    
    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    
    try:
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC").tz_convert("America/New_York")
        else:
            df.index = df.index.tz_convert("America/New_York")
    except Exception:
        pass
    if rth_only:
        try: df = df.between_time("09:30","16:00")
        except: pass
    return df[~df.index.duplicated(keep="last")].sort_index()

def intraday_enhancers(df1m: pd.DataFrame) -> dict:
    """Compute short-term metrics like VWAP distance, momentum, etc."""
    if df1m is None or df1m.empty: return {}
    enh = {}
    dlast = df1m.index.normalize().max()
    last_day = df1m[df1m.index.normalize() == dlast]
    if not last_day.empty:
        volume_sum = last_day["Volume"].sum()
        vwap = (last_day["Close"]*last_day["Volume"]).sum() / max(volume_sum, 1)
        last_close = last_day["Close"].iloc[-1]
        enh["vwap_dist_last"] = last_close/vwap - 1
        enh["intraday_mom"] = last_close/last_day["Open"].iloc[0] - 1
    close = df1m["Close"]
    enh["drift_7d"] = close.iloc[-1]/close.iloc[0] - 1 if len(close)>1 else np.nan
    return enh