import pandas as pd
from .data_daily import dl_daily, add_daily_indicators
from .config import TRAIN_PERIOD_DAILY

def build_training_set(symbols: list) -> tuple[pd.DataFrame, list]:
    frames = []
    feature_cols = None
    for s in symbols:
        df = dl_daily(s, TRAIN_PERIOD_DAILY)
        if df.empty: continue
        df = add_daily_indicators(df).dropna()
        if df.empty: continue
        feats = ["ret_1","ret_2","ret_3","ma20","ma50","rsi14",
                 "macd","macd_signal","macd_diff","bb_pct","vol_ratio20"]
        available = [f for f in feats if f in df.columns]
        dsub = df[available + ["fwd_ret_3d","y_up_3d"]].dropna().copy()
        dsub["symbol"] = s
        frames.append(dsub)
        feature_cols = available
    big = pd.concat(frames, ignore_index=True)
    return big, feature_cols
