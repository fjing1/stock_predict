#!/usr/bin/env python3
"""
Ultra High-Performance Stock Analysis Pipeline
Optimized for M3 Max with 1000+ stocks using parallel data fetching
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import argparse
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

from src.data_daily import dl_daily, add_daily_indicators
from src.enhanced_features import add_advanced_features
from src.enhanced_models import train_enhanced_models
from src.deep_learning import enhance_with_deep_learning, TENSORFLOW_AVAILABLE
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_stock_data_parallel(stock_data_dict, max_workers=64):
    """Process fetched stock data in parallel"""
    print(f"🔧 Processing {len(stock_data_dict)} stocks with {max_workers} workers...")
    
    def process_single_stock(symbol_df_tuple):
        symbol, df = symbol_df_tuple
        try:
            if df is None or df.empty or len(df) < 30:
                return None
            
            # Quick filters
            current_price = float(df["Close"].iloc[-1])
            if current_price < 1 or current_price > 50000:  # Extreme price filter
                return None
            
            recent_volume = df["Volume"].tail(5).mean()
            if recent_volume < 10000:  # Very low volume filter
                return None
            
            # Add indicators
            df = add_daily_indicators(df)
            df = add_advanced_features(df)
            
            # Remove rows with NaN values
            df = df.dropna()
            if len(df) < 20:
                return None
            
            return symbol, df
            
        except Exception as e:
            return None
    
    # Process in parallel
    processed_data = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_symbol = {
            executor.submit(process_single_stock, item): item[0] 
            for item in stock_data_dict.items()
        }
        
        completed = 0
        for future in as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                result = future.result()
                if result is not None:
                    symbol, processed_df = result
                    processed_data[symbol] = processed_df
                
                completed += 1
                if completed % 100 == 0:
                    print(f"   Processed {completed}/{len(stock_data_dict)} stocks...")
                    
            except Exception as e:
                continue
    
    print(f"✅ Processing complete: {len(processed_data)} stocks ready for analysis")
    return processed_data

def calculate_stock_scores_parallel(processed_data, max_workers=64):
    """Calculate trading scores for all stocks in parallel"""
    print(f"📊 Calculating scores for {len(processed_data)} stocks...")
    
    def calculate_single_score(symbol_df_tuple):
        symbol, df = symbol_df_tuple
        try:
            latest = df.iloc[-1]
            
            # Short-term momentum score
            momentum_score = 0
            if 'roc_3' in df.columns:
                momentum_score += latest['roc_3'] * 1000 if not pd.isna(latest['roc_3']) else 0
            if 'roc_5' in df.columns:
                momentum_score += latest['roc_5'] * 500 if not pd.isna(latest['roc_5']) else 0
            
            # RSI score
            rsi_score = 0
            if 'rsi_5' in df.columns and not pd.isna(latest['rsi_5']):
                rsi = latest['rsi_5']
                if 30 <= rsi <= 70:
                    rsi_score = 50
                elif rsi < 25:
                    rsi_score = 30  # Oversold
                elif rsi > 75:
                    rsi_score = -30  # Overbought
            
            # Volume score
            volume_score = 0
            if 'volume_ratio' in df.columns and not pd.isna(latest['volume_ratio']):
                vol_ratio = latest['volume_ratio']
                if vol_ratio > 1.5:
                    volume_score = 30
                elif vol_ratio > 1.2:
                    volume_score = 15
                elif vol_ratio < 0.7:
                    volume_score = -15
            
            # Trend score
            trend_score = 0
            if 'ma20' in df.columns and not pd.isna(latest['ma20']):
                if latest['Close'] > latest['ma20']:
                    trend_score = 25
                else:
                    trend_score = -15
            
            # Volatility score
            volatility_score = 0
            if 'atr_14' in df.columns and not pd.isna(latest['atr_14']):
                atr_pct = latest['atr_14'] / latest['Close']
                if 0.01 <= atr_pct <= 0.05:  # Good volatility
                    volatility_score = 20
                elif atr_pct > 0.1:  # Too volatile
                    volatility_score = -20
            
            total_score = momentum_score + rsi_score + volume_score + trend_score + volatility_score
            
            # Calculate probability and expected return
            prob_up = max(0.1, min(0.9, 0.5 + total_score / 1000))
            exp_return = max(-0.05, min(0.05, total_score / 10000))
            
            # Risk management
            atr = latest.get('atr_14', latest['Close'] * 0.02)
            stop_loss = latest['Close'] - (1.5 * atr)
            take_profit = latest['Close'] + (2.0 * atr)
            
            return {
                'symbol': symbol,
                'score': total_score,
                'current_price': latest['Close'],
                'prob_up': prob_up,
                'exp_return': exp_return,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'volume_ratio': latest.get('volume_ratio', 1.0),
                'rsi': latest.get('rsi_5', 50),
                'atr_pct': atr / latest['Close']
            }
            
        except Exception as e:
            return None
    
    # Calculate scores in parallel
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_symbol = {
            executor.submit(calculate_single_score, item): item[0] 
            for item in processed_data.items()
        }
        
        completed = 0
        for future in as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
                
                completed += 1
                if completed % 100 == 0:
                    print(f"   Scored {completed}/{len(processed_data)} stocks...")
                    
            except Exception as e:
                continue
    
    return pd.DataFrame(results)

def ultra_high_performance_analysis(symbols_file="symbols_1000_plus.txt", max_workers=64, top_n=50):
    """
    Ultra high-performance analysis pipeline for 1000+ stocks
    """
    print("🚀 ULTRA HIGH-PERFORMANCE STOCK ANALYSIS PIPELINE")
    print(f"⚡ M3 Max Optimization: {max_workers} parallel workers")
    print("=" * 80)
    
    start_time = time.time()
    
    # Load symbols
    with open(symbols_file, 'r') as f:
        symbols = [line.strip() for line in f if line.strip()]
    
    print(f"📊 Analyzing {len(symbols)} symbols")
    
    # Step 1: Parallel data fetching using original working method
    print(f"\n🔥 STEP 1: Ultra-Fast Parallel Data Fetching (Original Method)")
    
    def fetch_single_symbol(symbol):
        try:
            df = dl_daily(symbol, "6mo")
            if df is not None and not df.empty and len(df) > 30:
                return symbol, df
            return symbol, None
        except:
            return symbol, None
    
    # Use ThreadPoolExecutor like the working pipeline
    stock_data = {}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_symbol = {
            executor.submit(fetch_single_symbol, symbol): symbol
            for symbol in symbols
        }
        
        completed = 0
        for future in as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                result_symbol, df = future.result()
                if df is not None:
                    stock_data[result_symbol] = df
                
                completed += 1
                if completed % 50 == 0:
                    print(f"   Fetched {completed}/{len(symbols)} symbols...")
                    
            except Exception as e:
                continue
    
    if not stock_data:
        print("❌ No data fetched")
        return
    
    print(f"✅ Successfully fetched {len(stock_data)}/{len(symbols)} symbols ({len(stock_data)/len(symbols)*100:.1f}%)")
    
    # Step 2: Parallel data processing
    print(f"\n🔥 STEP 2: Parallel Data Processing & Feature Engineering")
    processed_data = process_stock_data_parallel(stock_data, max_workers)
    
    if not processed_data:
        print("❌ No data processed")
        return
    
    # Step 3: Parallel scoring
    print(f"\n🔥 STEP 3: Parallel Score Calculation")
    results_df = calculate_stock_scores_parallel(processed_data, max_workers)
    
    if results_df.empty:
        print("❌ No scores calculated")
        return
    
    # Step 4: Results analysis
    print(f"\n🔥 STEP 4: Results Analysis & Ranking")
    results_df = results_df.sort_values('score', ascending=False)
    
    # Get top picks
    bullish_picks = results_df.head(top_n)
    bearish_picks = results_df.tail(top_n)
    
    # Display results
    total_time = time.time() - start_time
    
    print(f"\n🎯 ULTRA HIGH-PERFORMANCE RESULTS")
    print("=" * 80)
    print(f"📊 Total symbols analyzed: {len(results_df)}")
    print(f"⏱️  Total processing time: {total_time:.2f} seconds")
    print(f"🚀 Processing rate: {len(results_df)/total_time:.1f} stocks/second")
    print(f"💪 M3 Max utilization: {max_workers} parallel workers")
    
    print(f"\n🚀 TOP {len(bullish_picks)} BULLISH PICKS:")
    print("-" * 100)
    print(f"{'Rank':<4} {'Symbol':<8} {'Price':<10} {'Score':<8} {'P(Up)':<8} {'Exp.Ret':<10} {'Vol.Ratio':<8} {'RSI':<6}")
    print("-" * 100)
    
    for i, (_, row) in enumerate(bullish_picks.iterrows(), 1):
        print(f"{i:<4} {row['symbol']:<8} ${row['current_price']:<9.2f} {row['score']:<8.1f} "
              f"{row['prob_up']:<8.1%} {row['exp_return']:<10.2%} {row['volume_ratio']:<8.1f} {row['rsi']:<6.0f}")
    
    print(f"\n📉 TOP {len(bearish_picks)} BEARISH PICKS:")
    print("-" * 100)
    print(f"{'Rank':<4} {'Symbol':<8} {'Price':<10} {'Score':<8} {'P(Down)':<8} {'Exp.Ret':<10} {'Vol.Ratio':<8} {'RSI':<6}")
    print("-" * 100)
    
    for i, (_, row) in enumerate(bearish_picks.iterrows(), 1):
        prob_down = 1 - row['prob_up']
        exp_ret_short = -row['exp_return']
        print(f"{i:<4} {row['symbol']:<8} ${row['current_price']:<9.2f} {row['score']:<8.1f} "
              f"{prob_down:<8.1%} {exp_ret_short:<10.2%} {row['volume_ratio']:<8.1f} {row['rsi']:<6.0f}")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    
    bullish_file = f"Ultra_Bullish_{timestamp}.csv"
    bearish_file = f"Ultra_Bearish_{timestamp}.csv"
    
    bullish_picks.to_csv(bullish_file, index=False)
    bearish_picks.to_csv(bearish_file, index=False)
    
    print(f"\n💾 Results saved:")
    print(f"   Bullish picks: {bullish_file}")
    print(f"   Bearish picks: {bearish_file}")
    
    # Performance summary
    print(f"\n🏆 PERFORMANCE SUMMARY:")
    print(f"   Symbols processed: {len(results_df):,}")
    print(f"   Processing time: {total_time:.2f} seconds")
    print(f"   Rate: {len(results_df)/total_time:.1f} stocks/second")
    print(f"   M3 Max workers: {max_workers}")
    print(f"   Success rate: {len(results_df)/len(symbols)*100:.1f}%")
    
    return results_df

def main():
    parser = argparse.ArgumentParser(description='Ultra High-Performance Stock Analysis for M3 Max')
    parser.add_argument('--symbols-file', default='symbols_1000_plus.txt',
                       help='File containing stock symbols')
    parser.add_argument('--workers', type=int, default=64,
                       help='Number of parallel workers (default: 64 for M3 Max)')
    parser.add_argument('--top-n', type=int, default=50,
                       help='Number of top picks to show')
    
    args = parser.parse_args()
    
    ultra_high_performance_analysis(
        symbols_file=args.symbols_file,
        max_workers=args.workers,
        top_n=args.top_n
    )

if __name__ == "__main__":
    main()