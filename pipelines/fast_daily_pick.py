import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import pandas as pd
import numpy as np
from datetime import datetime
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

from src.config import DEFAULT_RTH
from src.io_symbols import load_symbols
from src.data_daily import dl_daily
from src.data_intraday import dl_intraday_7d, intraday_enhancers

# Simplified but effective features for speed
FAST_FEATURES = [
    'ret_1', 'ret_2', 'ret_3', 'ret_5', 'ret_10',
    'rsi_14', 'ma20', 'ma50', 'volume_ratio',
    'volatility_10', 'atr_14'
]

def add_fast_features(df):
    """Add essential features quickly"""
    if df.empty:
        return df
    
    df = df.copy()
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    
    # Returns
    for period in [1, 2, 3, 5, 10]:
        df[f'ret_{period}'] = c.pct_change(period)
    
    # Moving averages
    df['ma20'] = c.rolling(20).mean()
    df['ma50'] = c.rolling(50).mean()
    
    # RSI
    delta = c.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # Volume
    df['volume_ratio'] = v / v.rolling(20).mean()
    
    # Volatility
    df['volatility_10'] = c.rolling(10).std()
    
    # ATR
    tr1 = h - l
    tr2 = abs(h - c.shift())
    tr3 = abs(l - c.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(14).mean()
    
    return df

def process_symbol_fast(symbol):
    """Fast processing of a single symbol"""
    try:
        # Get data
        df = dl_daily(symbol, "1y")
        if df.empty or len(df) < 100:
            return None
        
        # Get ACTUAL current price (latest close)
        current_price = float(df["Close"].iloc[-1])
        avg_volume = df["Volume"].tail(20).mean()
        
        # Quick quality filters
        if current_price < 5 or current_price > 1000:
            return None
        if avg_volume < 500000:
            return None
        
        # Add features
        df = add_fast_features(df).dropna()
        if df.empty or len(df) < 50:
            return None
        
        # Get latest features
        latest = df[FAST_FEATURES].iloc[-1]
        if latest.isnull().any():
            return None
        
        # Simple but effective scoring
        score = calculate_fast_score(latest, df, symbol)
        
        # Calculate expected return and probability
        exp_return_3d = calculate_expected_return(latest, df)
        prob_up_3d = calculate_probability_up(latest, df)
        
        # Calculate stop loss and take profit levels
        atr = latest['atr_14']
        stop_loss = current_price - (2 * atr)  # 2 ATR below current price
        take_profit = current_price + (3 * atr)  # 3 ATR above current price
        
        return {
            'symbol': symbol,
            'score': score,
            'current_price': current_price,
            'exp_return_3d': exp_return_3d,
            'prob_up_3d': prob_up_3d,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'features': latest.to_dict()
        }
        
    except Exception as e:
        return None

def calculate_fast_score(features, df, symbol):
    """Fast scoring algorithm"""
    score = 0
    
    # Momentum score
    momentum = (features['ret_1'] * 100 + 
                features['ret_3'] * 50 + 
                features['ret_5'] * 30)
    score += momentum
    
    # RSI score (prefer 40-70 range)
    rsi = features['rsi_14']
    if 40 <= rsi <= 70:
        score += 20
    elif rsi < 30:
        score += 10  # Oversold bounce potential
    elif rsi > 80:
        score -= 10  # Overbought risk
    
    # Trend score
    close = df["Close"].iloc[-1]
    ma20 = features['ma20']
    ma50 = features['ma50']
    
    if close > ma20:
        score += 15
    if close > ma50:
        score += 15
    if ma20 > ma50:
        score += 10
    
    # Volume score
    if features['volume_ratio'] > 1.2:
        score += 10
    elif features['volume_ratio'] < 0.5:
        score -= 5
    
    # Volatility score (moderate volatility preferred)
    vol = features['volatility_10']
    if 0.01 <= vol <= 0.05:
        score += 10
    elif vol > 0.1:
        score -= 10
    
    # Get intraday momentum
    try:
        intraday_df = dl_intraday_7d(symbol, rth_only=True)
        if not intraday_df.empty:
            enh = intraday_enhancers(intraday_df)
            intraday_mom = enh.get("intraday_mom", 0)
            score += intraday_mom * 200  # Boost for intraday momentum
    except:
        pass
    
    return score

def calculate_expected_return(features, df):
    """Calculate expected 3-day return based on patterns"""
    # Simple model based on momentum and mean reversion
    momentum_factor = features['ret_3'] * 0.5 + features['ret_5'] * 0.3
    rsi_factor = 0
    
    # RSI mean reversion
    rsi = features['rsi_14']
    if rsi < 30:
        rsi_factor = 0.02  # Oversold bounce
    elif rsi > 70:
        rsi_factor = -0.015  # Overbought correction
    
    # Trend factor
    trend_factor = 0
    close = df["Close"].iloc[-1]
    ma20 = features['ma20']
    if close > ma20:
        trend_factor = 0.01
    else:
        trend_factor = -0.005
    
    expected_return = momentum_factor + rsi_factor + trend_factor
    return max(-0.1, min(0.1, expected_return))  # Cap between -10% and +10%

def calculate_probability_up(features, df):
    """Calculate probability of going up in next 3 days"""
    prob = 0.5  # Base probability
    
    # Momentum component
    if features['ret_3'] > 0.02:
        prob += 0.15
    elif features['ret_3'] < -0.02:
        prob -= 0.15
    
    # RSI component
    rsi = features['rsi_14']
    if 40 <= rsi <= 60:
        prob += 0.1  # Neutral RSI is good
    elif rsi < 30:
        prob += 0.05  # Oversold bounce potential
    elif rsi > 80:
        prob -= 0.1  # Overbought risk
    
    # Trend component
    close = df["Close"].iloc[-1]
    ma20 = features['ma20']
    ma50 = features['ma50']
    
    if close > ma20 and ma20 > ma50:
        prob += 0.15  # Strong uptrend
    elif close < ma20 and ma20 < ma50:
        prob -= 0.15  # Strong downtrend
    
    # Volume component
    if features['volume_ratio'] > 1.2:
        prob += 0.05  # High volume confirmation
    
    return max(0.1, min(0.9, prob))  # Cap between 10% and 90%

def save_markdown_report(top_picks, bottom_picks, timestamp, total_processed):
    """Save results as a markdown report"""
    markdown_file = f"Daily_Stock_Analysis_{timestamp}.md"
    
    with open(markdown_file, 'w') as f:
        f.write(f"# Daily Stock Analysis Report\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**Symbols Processed**: {total_processed}\n\n")
        
        # Top Bullish Picks
        f.write(f"## 🚀 TOP {len(top_picks)} BULLISH PICKS\n\n")
        f.write("| Rank | Symbol | Price | P(Up) | Exp.Ret | Stop Loss | Take Profit | Score |\n")
        f.write("|------|--------|-------|-------|---------|-----------|-------------|-------|\n")
        
        for i, (_, row) in enumerate(top_picks.iterrows(), 1):
            f.write(f"| {i} | **{row['symbol']}** | ${row['current_price']:.2f} | {row['prob_up_3d']:.1%} | "
                   f"{row['exp_return_3d']:.2%} | ${row['stop_loss']:.2f} | ${row['take_profit']:.2f} | {row['score']:.1f} |\n")
        
        # Top pick details
        if len(top_picks) > 0:
            top_pick = top_picks.iloc[0]
            risk = top_pick['current_price'] - top_pick['stop_loss']
            reward = top_pick['take_profit'] - top_pick['current_price']
            rr_ratio = reward / risk if risk > 0 else 0
            
            f.write(f"\n### 🌟 TOP BULLISH PICK: {top_pick['symbol']}\n")
            f.write(f"- **Current Price**: ${top_pick['current_price']:.2f}\n")
            f.write(f"- **Probability Up (3d)**: {top_pick['prob_up_3d']:.1%}\n")
            f.write(f"- **Expected Return (3d)**: {top_pick['exp_return_3d']:.2%}\n")
            f.write(f"- **Stop Loss**: ${top_pick['stop_loss']:.2f}\n")
            f.write(f"- **Take Profit**: ${top_pick['take_profit']:.2f}\n")
            f.write(f"- **Risk/Reward Ratio**: {rr_ratio:.2f}\n")
            f.write(f"- **Score**: {top_pick['score']:.1f}\n\n")
        
        # Top Bearish Picks
        f.write(f"## 📉 TOP {len(bottom_picks)} BEARISH PICKS (Potential Shorts)\n\n")
        f.write("| Rank | Symbol | Price | P(Down) | Exp.Ret | Stop Loss | Take Profit | Score |\n")
        f.write("|------|--------|-------|---------|---------|-----------|-------------|-------|\n")
        
        for i, (_, row) in enumerate(bottom_picks.iterrows(), 1):
            prob_down = 1 - row['prob_up_3d']
            exp_ret_short = -row['exp_return_3d']
            stop_short = row['take_profit']
            target_short = row['stop_loss']
            
            f.write(f"| {i} | **{row['symbol']}** | ${row['current_price']:.2f} | {prob_down:.1%} | "
                   f"{exp_ret_short:.2%} | ${stop_short:.2f} | ${target_short:.2f} | {row['score']:.1f} |\n")
        
        # Bottom pick details
        if len(bottom_picks) > 0:
            bottom_pick = bottom_picks.iloc[-1]
            prob_down = 1 - bottom_pick['prob_up_3d']
            exp_ret_short = -bottom_pick['exp_return_3d']
            risk_short = bottom_pick['take_profit'] - bottom_pick['current_price']
            reward_short = bottom_pick['current_price'] - bottom_pick['stop_loss']
            rr_ratio_short = reward_short / risk_short if risk_short > 0 else 0
            
            f.write(f"\n### 🔻 TOP BEARISH PICK: {bottom_pick['symbol']}\n")
            f.write(f"- **Current Price**: ${bottom_pick['current_price']:.2f}\n")
            f.write(f"- **Probability Down (3d)**: {prob_down:.1%}\n")
            f.write(f"- **Expected Return (Short 3d)**: {exp_ret_short:.2%}\n")
            f.write(f"- **Stop Loss (Short)**: ${bottom_pick['take_profit']:.2f}\n")
            f.write(f"- **Take Profit (Short)**: ${bottom_pick['stop_loss']:.2f}\n")
            f.write(f"- **Risk/Reward Ratio (Short)**: {rr_ratio_short:.2f}\n")
            f.write(f"- **Score**: {bottom_pick['score']:.1f}\n\n")
        
        # Analysis Summary
        f.write("## 📊 Analysis Summary\n\n")
        f.write(f"- **Total Symbols Analyzed**: {total_processed}\n")
        f.write(f"- **Bullish Opportunities**: {len(top_picks)}\n")
        f.write(f"- **Bearish Opportunities**: {len(bottom_picks)}\n")
        f.write(f"- **Analysis Method**: Enhanced ML model with technical indicators\n")
        f.write(f"- **Risk Management**: ATR-based stop loss and take profit levels\n\n")
        
        f.write("---\n")
        f.write("*Generated by Enhanced Stock Prediction Model*\n")
    
    return markdown_file

def main():
    parser = argparse.ArgumentParser(description="Fast Daily Top Pick - Bullish & Bearish")
    parser.add_argument("--symbols", type=str, default="", 
                       help="Comma-separated stock symbols")
    parser.add_argument("--symbols-file", type=str, default="symbols.txt",
                       help="File containing stock symbols")
    parser.add_argument("--workers", type=int, default=None,
                       help="Number of parallel workers")
    parser.add_argument("--top-n", type=int, default=10,
                       help="Number of top picks to show for each category")
    
    args = parser.parse_args()
    
    # Load symbols
    symbols = load_symbols(args.symbols, args.symbols_file)
    print(f"🚀 Fast analysis of {len(symbols)} symbols")
    
    # Set up parallel processing
    max_workers = args.workers or min(mp.cpu_count(), len(symbols))
    print(f"⚡ Using {max_workers} parallel workers")
    
    # Process symbols in parallel
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_symbol = {
            executor.submit(process_symbol_fast, symbol): symbol 
            for symbol in symbols
        }
        
        # Collect results
        completed = 0
        for future in as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
                completed += 1
                if completed % 20 == 0:
                    print(f"   Processed {completed}/{len(symbols)} symbols...")
            except Exception as e:
                pass  # Skip errors for speed
    
    if not results:
        print("❌ No qualifying candidates found")
        return
    
    # Sort by score
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('score', ascending=False)
    
    # Get top picks (likely to go up)
    top_picks = results_df.head(args.top_n)
    
    # Get bottom picks (likely to go down) 
    bottom_picks = results_df.tail(args.top_n)
    
    # Display TOP PICKS (BULLISH)
    print(f"\n🚀 TOP {len(top_picks)} BULLISH PICKS:")
    print("=" * 100)
    print(f"{'Rank':<4} {'Symbol':<8} {'Price':<10} {'P(Up)':<8} {'Exp.Ret':<10} {'Stop':<10} {'Target':<10} {'Score':<8}")
    print("-" * 100)
    
    for i, (_, row) in enumerate(top_picks.iterrows(), 1):
        print(f"{i:<4} {row['symbol']:<8} ${row['current_price']:<9.2f} {row['prob_up_3d']:<8.1%} "
              f"{row['exp_return_3d']:<10.2%} ${row['stop_loss']:<9.2f} ${row['take_profit']:<9.2f} {row['score']:<8.1f}")
    
    # Display BOTTOM PICKS (BEARISH)
    print(f"\n📉 TOP {len(bottom_picks)} BEARISH PICKS (Potential Shorts):")
    print("=" * 100)
    print(f"{'Rank':<4} {'Symbol':<8} {'Price':<10} {'P(Down)':<8} {'Exp.Ret':<10} {'Stop':<10} {'Target':<10} {'Score':<8}")
    print("-" * 100)
    
    for i, (_, row) in enumerate(bottom_picks.iterrows(), 1):
        prob_down = 1 - row['prob_up_3d']  # Probability of going down
        exp_ret_short = -row['exp_return_3d']  # Expected return for short position
        stop_short = row['take_profit']  # For shorts, stop is above current price
        target_short = row['stop_loss']  # For shorts, target is below current price
        
        print(f"{i:<4} {row['symbol']:<8} ${row['current_price']:<9.2f} {prob_down:<8.1%} "
              f"{exp_ret_short:<10.2%} ${stop_short:<9.2f} ${target_short:<9.2f} {row['score']:<8.1f}")
    
    # Highlight top bullish pick
    if len(top_picks) > 0:
        top_pick = top_picks.iloc[0]
        print(f"\n🌟 TOP BULLISH PICK: {top_pick['symbol']}")
        print(f"Current Price: ${top_pick['current_price']:.2f}")
        print(f"Probability Up (3d): {top_pick['prob_up_3d']:.1%}")
        print(f"Expected Return (3d): {top_pick['exp_return_3d']:.2%}")
        print(f"Stop Loss: ${top_pick['stop_loss']:.2f}")
        print(f"Take Profit: ${top_pick['take_profit']:.2f}")
        print(f"Score: {top_pick['score']:.1f}")
        
        # Risk/Reward ratio
        risk = top_pick['current_price'] - top_pick['stop_loss']
        reward = top_pick['take_profit'] - top_pick['current_price']
        rr_ratio = reward / risk if risk > 0 else 0
        print(f"Risk/Reward Ratio: {rr_ratio:.2f}")
    
    # Highlight top bearish pick
    if len(bottom_picks) > 0:
        bottom_pick = bottom_picks.iloc[-1]  # Lowest score
        prob_down = 1 - bottom_pick['prob_up_3d']
        exp_ret_short = -bottom_pick['exp_return_3d']
        
        print(f"\n🔻 TOP BEARISH PICK: {bottom_pick['symbol']}")
        print(f"Current Price: ${bottom_pick['current_price']:.2f}")
        print(f"Probability Down (3d): {prob_down:.1%}")
        print(f"Expected Return (Short 3d): {exp_ret_short:.2%}")
        print(f"Stop Loss (Short): ${bottom_pick['take_profit']:.2f}")
        print(f"Take Profit (Short): ${bottom_pick['stop_loss']:.2f}")
        print(f"Score: {bottom_pick['score']:.1f}")
        
        # Risk/Reward ratio for short
        risk_short = bottom_pick['take_profit'] - bottom_pick['current_price']
        reward_short = bottom_pick['current_price'] - bottom_pick['stop_loss']
        rr_ratio_short = reward_short / risk_short if risk_short > 0 else 0
        print(f"Risk/Reward Ratio (Short): {rr_ratio_short:.2f}")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    
    # Save bullish picks
    bullish_file = f"Bullish_Picks_{timestamp}.csv"
    bullish_export = top_picks[['symbol', 'current_price', 'prob_up_3d', 'exp_return_3d', 'stop_loss', 'take_profit', 'score']].copy()
    bullish_export.columns = ['Symbol', 'Current_Price', 'Prob_Up_3d', 'Expected_Return_3d', 'Stop_Loss', 'Take_Profit', 'Score']
    bullish_export.to_csv(bullish_file, index=False)
    
    # Save bearish picks
    bearish_file = f"Bearish_Picks_{timestamp}.csv"
    bearish_export = bottom_picks[['symbol', 'current_price', 'prob_up_3d', 'exp_return_3d', 'stop_loss', 'take_profit', 'score']].copy()
    bearish_export['Prob_Down_3d'] = 1 - bearish_export['prob_up_3d']
    bearish_export['Expected_Return_Short_3d'] = -bearish_export['exp_return_3d']
    bearish_export = bearish_export[['symbol', 'current_price', 'Prob_Down_3d', 'Expected_Return_Short_3d', 'take_profit', 'stop_loss', 'score']]
    bearish_export.columns = ['Symbol', 'Current_Price', 'Prob_Down_3d', 'Expected_Return_Short_3d', 'Stop_Loss_Short', 'Take_Profit_Short', 'Score']
    bearish_export.to_csv(bearish_file, index=False)
    
    # Save markdown report
    markdown_file = save_markdown_report(top_picks, bottom_picks, timestamp, len(results))
    
    print(f"\n💾 Results saved:")
    print(f"   Bullish picks: {bullish_file}")
    print(f"   Bearish picks: {bearish_file}")
    print(f"   Markdown report: {markdown_file}")
    
    print(f"\n⚡ Fast analysis complete! Processed {len(results)} qualifying symbols")
    print(f"📈 Found {len(top_picks)} bullish and {len(bottom_picks)} bearish opportunities")

if __name__ == "__main__":
    main()