import numpy as np
import pandas as pd
from .data_daily import dl_daily
from .data_intraday import dl_intraday_7d, intraday_enhancers
from .enhanced_features import add_advanced_features, get_enhanced_feature_list

class QualityFilter:
    """Filter for high-quality trade ideas"""
    
    @staticmethod
    def volume_filter(df, min_avg_volume=1000000):
        """Filter by minimum average volume"""
        if df.empty:
            return False
        avg_volume = df["Volume"].tail(20).mean()
        return avg_volume >= min_avg_volume
    
    @staticmethod
    def price_filter(df, min_price=5.0, max_price=1000.0):
        """Filter by price range"""
        if df.empty:
            return False
        current_price = df["Close"].iloc[-1]
        return min_price <= current_price <= max_price
    
    @staticmethod
    def volatility_filter(df, min_vol=0.01, max_vol=0.15):
        """Filter by volatility range"""
        if df.empty:
            return False
        returns = df["Close"].pct_change().tail(20)
        volatility = returns.std() * np.sqrt(252)  # Annualized
        return min_vol <= volatility <= max_vol
    
    @staticmethod
    def trend_consistency_filter(df, min_consistency=0.6):
        """Filter by trend consistency"""
        if df.empty or len(df) < 10:
            return False
        
        # Check if recent trend is consistent
        recent_returns = df["Close"].pct_change().tail(10)
        positive_days = (recent_returns > 0).sum()
        consistency = max(positive_days, 10 - positive_days) / 10
        return consistency >= min_consistency
    
    @staticmethod
    def liquidity_filter(df, min_dollar_volume=10000000):
        """Filter by dollar volume (liquidity)"""
        if df.empty:
            return False
        recent_dollar_vol = (df["Close"] * df["Volume"]).tail(5).mean()
        return recent_dollar_vol >= min_dollar_volume

def enhanced_score_today(symbols, feature_cols, model, rth_only=True, top_n=1):
    """Enhanced scoring with quality filters and confidence measures"""
    rows = []
    
    for s in symbols:
        try:
            # Get daily data
            d = dl_daily(s, "1y")  # More data for better feature calculation
            if d.empty:
                continue
                
            # Apply quality filters
            if not QualityFilter.volume_filter(d):
                continue
            if not QualityFilter.price_filter(d):
                continue
            if not QualityFilter.volatility_filter(d):
                continue
            if not QualityFilter.liquidity_filter(d):
                continue
            
            # Add enhanced features
            d = add_advanced_features(d).dropna()
            if d.empty or len(d) < 50:  # Need sufficient data
                continue
            
            # Get latest features
            x = d[feature_cols].iloc[-1:].astype(float)
            if x.isnull().any().any():
                continue
            
            # Get model predictions and confidence
            p_up, exp_ret = model.predict(x, feature_cols)
            cls_conf, reg_conf = model.get_prediction_confidence(x, feature_cols)
            
            p_up = float(p_up[0])
            exp_ret = float(exp_ret[0])
            cls_conf = float(cls_conf[0])
            reg_conf = float(reg_conf[0])
            
            # Get intraday enhancers
            enh = intraday_enhancers(dl_intraday_7d(s, rth_only=rth_only))
            mom = enh.get("intraday_mom", 0)
            vwap = enh.get("vwap_dist_last", 0)
            drift = enh.get("drift_7d", 0)
            
            # Enhanced scoring system
            base_score = calculate_enhanced_score(p_up, exp_ret, mom, vwap, drift)
            
            # Apply confidence multiplier
            confidence_multiplier = (cls_conf + reg_conf) / 2
            final_score = base_score * confidence_multiplier
            
            # Additional quality metrics
            current_price = float(d["Close"].iloc[-1])
            volume_ratio = float(d["Volume"].tail(5).mean() / d["Volume"].tail(20).mean())
            
            # Risk-adjusted return
            volatility = d["Close"].pct_change().tail(20).std()
            risk_adj_return = exp_ret / (volatility + 1e-8) if volatility > 0 else 0
            
            # Technical strength
            rsi_14 = d["rsi_14"].iloc[-1] if "rsi_14" in d.columns else 50
            tech_strength = calculate_technical_strength(d)
            
            rows.append({
                'symbol': s,
                'p_up_3d': p_up,
                'exp_ret_3d': exp_ret,
                'base_score': base_score,
                'confidence': confidence_multiplier,
                'final_score': final_score,
                'close': current_price,
                'volume_ratio': volume_ratio,
                'risk_adj_return': risk_adj_return,
                'tech_strength': tech_strength,
                'rsi_14': rsi_14,
                'intraday_mom': mom,
                'vwap_dist': vwap
            })
            
        except Exception as e:
            print(f"⚠️ Error processing {s}: {e}")
            continue
    
    if not rows:
        return pd.DataFrame()
    
    df = pd.DataFrame(rows)
    
    # Apply final quality filters
    df = apply_final_filters(df)
    
    # Sort by final score and return top N
    df = df.sort_values("final_score", ascending=False)
    return df.head(top_n)

def calculate_enhanced_score(p_up, exp_ret, mom, vwap, drift):
    """Enhanced scoring algorithm"""
    # Base probability and return components
    prob_component = 100 * p_up
    return_component = 500 * exp_ret  # Increased weight on expected return
    
    # Momentum components
    momentum_component = 30 * mom
    
    # VWAP component (penalize being too far from VWAP)
    vwap_component = -15 * abs(vwap) if abs(vwap) > 0.02 else 5 * abs(vwap)
    
    # Drift component
    drift_component = 10 * drift
    
    # Combine components
    score = prob_component + return_component + momentum_component + vwap_component + drift_component
    
    return score

def calculate_technical_strength(df):
    """Calculate overall technical strength"""
    if df.empty or len(df) < 20:
        return 0
    
    strength = 0
    
    # Price vs moving averages
    close = df["Close"].iloc[-1]
    ma_20 = df["Close"].rolling(20).mean().iloc[-1]
    ma_50 = df["Close"].rolling(50).mean().iloc[-1] if len(df) >= 50 else ma_20
    
    if close > ma_20:
        strength += 25
    if close > ma_50:
        strength += 25
    if ma_20 > ma_50:
        strength += 25
    
    # RSI in optimal range
    if "rsi_14" in df.columns:
        rsi = df["rsi_14"].iloc[-1]
        if 40 <= rsi <= 70:  # Not oversold or overbought
            strength += 25
    
    return strength

def apply_final_filters(df):
    """Apply final quality filters to the scored dataframe"""
    if df.empty:
        return df
    
    # Filter by minimum confidence
    df = df[df["confidence"] >= 0.7]
    
    # Filter by minimum probability
    df = df[df["p_up_3d"] >= 0.55]
    
    # Filter by minimum expected return
    df = df[df["exp_ret_3d"] >= 0.01]  # At least 1% expected return
    
    # Filter by technical strength
    df = df[df["tech_strength"] >= 50]
    
    # Filter by volume (avoid low volume spikes)
    df = df[df["volume_ratio"] >= 0.8]  # Recent volume not too low
    
    return df

def get_daily_top_pick(symbols, feature_cols, model, rth_only=True):
    """Get the single best trade idea for the day"""
    top_picks = enhanced_score_today(symbols, feature_cols, model, rth_only, top_n=5)
    
    if top_picks.empty:
        return None
    
    # Additional validation for the top pick
    best_pick = top_picks.iloc[0]
    
    # Ensure the pick meets strict criteria
    if (best_pick["confidence"] >= 0.8 and 
        best_pick["p_up_3d"] >= 0.6 and 
        best_pick["exp_ret_3d"] >= 0.015 and
        best_pick["tech_strength"] >= 75):
        return best_pick
    
    return None