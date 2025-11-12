import pandas as pd
import numpy as np
import ta
try:
    from scipy import stats
except ImportError:
    print("⚠️ scipy not available, some advanced features will be disabled")
    stats = None

def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add sophisticated technical indicators and market microstructure features"""
    if df.empty:
        return df
    
    df = df.copy()
    c, h, l, o, v = df["Close"], df["High"], df["Low"], df["Open"], df["Volume"]
    
    # Price action features
    df["hl_ratio"] = (h - l) / c
    df["body_ratio"] = abs(c - o) / (h - l + 1e-8)
    df["upper_shadow"] = (h - np.maximum(c, o)) / (h - l + 1e-8)
    df["lower_shadow"] = (np.minimum(c, o) - l) / (h - l + 1e-8)
    
    # Multi-timeframe momentum
    for period in [3, 5, 8, 13, 21]:
        df[f"roc_{period}"] = c.pct_change(period)
        df[f"rsi_{period}"] = ta.momentum.RSIIndicator(c, window=period).rsi()
    
    # Volatility features
    df["atr_14"] = ta.volatility.AverageTrueRange(h, l, c, window=14).average_true_range()
    df["volatility_10"] = c.rolling(10).std()
    df["volatility_ratio"] = df["volatility_10"] / df["volatility_10"].rolling(20).mean()
    
    # Volume analysis
    df["volume_sma_20"] = v.rolling(20).mean()
    df["volume_ratio"] = v / df["volume_sma_20"]
    df["price_volume"] = c * v
    df["vwap_20"] = (df["price_volume"].rolling(20).sum() / v.rolling(20).sum())
    df["price_vs_vwap"] = c / df["vwap_20"] - 1
    
    # Trend strength
    df["adx_14"] = ta.trend.ADXIndicator(h, l, c, window=14).adx()
    df["cci_20"] = ta.trend.CCIIndicator(h, l, c, window=20).cci()
    
    # Support/Resistance levels
    df["resistance_20"] = h.rolling(20).max()
    df["support_20"] = l.rolling(20).min()
    df["price_position"] = (c - df["support_20"]) / (df["resistance_20"] - df["support_20"] + 1e-8)
    
    # Market regime detection
    if stats is not None:
        df["trend_20"] = c.rolling(20).apply(lambda x: stats.linregress(range(len(x)), x)[0] if len(x) == 20 else np.nan)
        df["price_momentum"] = c.rolling(10).apply(lambda x: stats.linregress(range(len(x)), x)[0] if len(x) == 10 else np.nan)
        df["volume_momentum"] = v.rolling(10).apply(lambda x: stats.linregress(range(len(x)), x)[0] if len(x) == 10 else np.nan)
    else:
        # Fallback to simple slope calculation
        df["trend_20"] = c.pct_change(20)
        df["price_momentum"] = c.pct_change(10)
        df["volume_momentum"] = v.pct_change(10)
    
    df["trend_strength"] = abs(df["trend_20"]) / (df["volatility_10"] + 1e-8)
    
    # Gap analysis
    df["gap"] = (o - c.shift(1)) / c.shift(1)
    df["gap_filled"] = ((l <= c.shift(1)) & (df["gap"] > 0)) | ((h >= c.shift(1)) & (df["gap"] < 0))
    
    return df

def get_enhanced_feature_list():
    """Return list of enhanced features for model training"""
    return [
        # Basic momentum
        "roc_3", "roc_5", "roc_8", "roc_13", "roc_21",
        "rsi_3", "rsi_5", "rsi_8", "rsi_13", "rsi_21",
        
        # Price action
        "hl_ratio", "body_ratio", "upper_shadow", "lower_shadow",
        
        # Volatility
        "atr_14", "volatility_ratio",
        
        # Volume
        "volume_ratio", "price_vs_vwap",
        
        # Trend
        "adx_14", "cci_20", "price_position",
        
        # Market regime
        "trend_strength", "price_momentum", "volume_momentum",
        
        # Gaps
        "gap"
    ]