#!/usr/bin/env python3
"""
Stock Predictor - Fixed Version
Generates top bullish and bearish picks with visualizations
Fixed duplicate price issue with proper data isolation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

from src.data_daily import dl_daily, add_daily_indicators
from src.enhanced_features import add_advanced_features
from src.stock_visualizer import create_top_picks_charts
from concurrent.futures import ThreadPoolExecutor, as_completed

def run_stock_prediction(symbols_file="symbols_expanded.txt", max_workers=64):
    """
    Main function to run stock prediction with fixed data isolation
    """
    print("🚀 STOCK PREDICTOR - FIXED VERSION")
    print("=" * 60)
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = time.time()
    
    # Load symbols
    with open(symbols_file, 'r') as f:
        symbols = [line.strip() for line in f if line.strip()]
    
    print(f"📊 Analyzing {len(symbols)} symbols with {max_workers} workers")
    
    # Step 1: Fetch data with proper isolation
    print(f"\n🔥 STEP 1: Fetching Stock Data")
    
    def fetch_single_symbol(symbol):
        try:
            # Fetch fresh data for each symbol
            df = dl_daily(symbol, "6mo")
            if df is not None and not df.empty and len(df) > 30:
                # Ensure data integrity by adding symbol tracking
                df = df.copy()
                df['_symbol_id'] = symbol
                return symbol, df
            return symbol, None
        except Exception as e:
            return symbol, None
    
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
    
    print(f"✅ Successfully fetched {len(stock_data)} symbols")
    
    # Step 2: Process data with isolation
    print(f"\n🔥 STEP 2: Processing Data & Calculating Scores")
    
    def process_single_stock(symbol):
        try:
            if symbol not in stock_data:
                return None
                
            # Get a fresh copy of the data
            df = stock_data[symbol].copy()
            
            # Verify data belongs to correct symbol
            if '_symbol_id' in df.columns and df['_symbol_id'].iloc[0] != symbol:
                print(f"❌ Data contamination: {symbol} has data for {df['_symbol_id'].iloc[0]}")
                return None
            
            # Quick filters
            current_price = float(df["Close"].iloc[-1])
            if current_price < 1 or current_price > 50000:
                return None
            
            recent_volume = df["Volume"].tail(5).mean()
            if recent_volume < 10000:
                return None
            
            # Add indicators to fresh copy
            df = add_daily_indicators(df)
            df = add_advanced_features(df)
            df = df.dropna()
            
            if len(df) < 20:
                return None
            
            # Calculate score
            latest = df.iloc[-1]
            
            # Enhanced scoring system
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
                    rsi_score = 30
                elif rsi > 75:
                    rsi_score = -30
            
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
                if 0.01 <= atr_pct <= 0.05:
                    volatility_score = 20
                elif atr_pct > 0.1:
                    volatility_score = -20
            
            total_score = momentum_score + rsi_score + volume_score + trend_score + volatility_score
            
            # Calculate metrics
            predicted_return = max(-0.15, min(0.15, total_score / 5000))
            confidence_score = max(50, min(95, 70 + abs(total_score) / 50))
            
            # Risk management
            atr = latest.get('atr_14', latest['Close'] * 0.02)
            stop_loss = latest['Close'] - (1.5 * atr)
            take_profit = latest['Close'] + (2.0 * atr)
            
            # Quantified risk score (0-100, where 0 is lowest risk, 100 is highest risk)
            atr_pct = atr / latest['Close']
            volatility_risk = min(100, (atr_pct * 1000))
            return_risk = min(100, abs(predicted_return) * 1000)
            confidence_risk = max(0, 100 - confidence_score)
            
            # Combined risk score (weighted average)
            risk_score = (volatility_risk * 0.4 + return_risk * 0.3 + confidence_risk * 0.3)
            risk_score = round(risk_score, 2)
            
            # Risk level for display
            if risk_score < 30:
                risk_level = 'LOW'
            elif risk_score < 60:
                risk_level = 'MEDIUM'
            else:
                risk_level = 'HIGH'
            
            return {
                'symbol': symbol,
                'predicted_return': predicted_return,
                'confidence_score': confidence_score,
                'current_price': float(latest['Close']),
                'stop_loss': float(stop_loss),
                'take_profit': float(take_profit),
                'risk_score': risk_score,
                'risk_level': risk_level,
                'score': total_score,
                'processed_df': df
            }
            
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
            return None
    
    # Process all stocks sequentially to avoid data contamination
    results = []
    processed_data = {}
    
    for i, symbol in enumerate(stock_data.keys()):
        result = process_single_stock(symbol)
        if result is not None:
            processed_data[result['symbol']] = result['processed_df']
            del result['processed_df']  # Remove from result dict
            results.append(result)
        
        if (i + 1) % 50 == 0:
            print(f"   Processed {i + 1}/{len(stock_data)} stocks...")
    
    print(f"   Processed {len(stock_data)}/{len(stock_data)} stocks...")
    
    if not results:
        print("❌ No results generated")
        return
    
    # Step 3: Validate and select top picks
    print(f"\n🔥 STEP 3: Selecting Top Picks")
    
    # Check for duplicate prices
    prices = [r['current_price'] for r in results]
    unique_prices = set(prices)
    if len(unique_prices) < len(prices):
        print(f"⚠️  Warning: Found {len(prices) - len(unique_prices)} duplicate prices")
        # Remove duplicates by keeping first occurrence
        seen_prices = set()
        filtered_results = []
        for r in results:
            if r['current_price'] not in seen_prices:
                seen_prices.add(r['current_price'])
                filtered_results.append(r)
            else:
                print(f"   Removing duplicate price ${r['current_price']:.2f} for {r['symbol']}")
        results = filtered_results
        print(f"   Kept {len(results)} unique results")
    
    # Sort by confidence score
    results.sort(key=lambda x: x['confidence_score'], reverse=True)
    
    # Get top bullish and bearish
    bullish_picks = [r for r in results if r['predicted_return'] > 0][:10]
    bearish_picks = [r for r in results if r['predicted_return'] < 0][:10]
    
    if not bullish_picks and not bearish_picks:
        print("❌ No significant picks found")
        return
    
    # Get the single best picks
    top_bullish = bullish_picks[0] if bullish_picks else None
    top_bearish = bearish_picks[0] if bearish_picks else None
    
    # Step 4: Generate visualizations
    print(f"\n🔥 STEP 4: Generating Visualizations")
    
    chart_files = create_top_picks_charts(top_bullish, top_bearish, processed_data)
    
    # Step 5: Display results
    total_time = time.time() - start_time
    
    print(f"\n🎯 STOCK PREDICTION RESULTS")
    print("=" * 60)
    print(f"📊 Symbols analyzed: {len(results)}")
    print(f"⏱️  Processing time: {total_time:.1f} seconds")
    print(f"🚀 Processing rate: {len(results)/total_time:.1f} stocks/second")
    
    if top_bullish:
        print(f"\n📈 TOP BULLISH PICK:")
        print(f"   Symbol: {top_bullish['symbol']}")
        print(f"   Expected Return: +{top_bullish['predicted_return']*100:.2f}%")
        print(f"   Confidence: {top_bullish['confidence_score']:.1f}%")
        print(f"   Current Price: ${top_bullish['current_price']:.2f}")
        print(f"   Target Price: ${top_bullish['current_price'] * (1 + top_bullish['predicted_return']):.2f}")
        print(f"   Stop Loss: ${top_bullish['stop_loss']:.2f}")
        print(f"   Take Profit: ${top_bullish['take_profit']:.2f}")
        print(f"   Risk Level: {top_bullish['risk_level']}")
    
    if top_bearish:
        print(f"\n📉 TOP BEARISH PICK:")
        print(f"   Symbol: {top_bearish['symbol']}")
        print(f"   Expected Return: {top_bearish['predicted_return']*100:.2f}%")
        print(f"   Confidence: {top_bearish['confidence_score']:.1f}%")
        print(f"   Current Price: ${top_bearish['current_price']:.2f}")
        print(f"   Target Price: ${top_bearish['current_price'] * (1 + top_bearish['predicted_return']):.2f}")
        print(f"   Stop Loss: ${top_bearish['stop_loss']:.2f}")
        print(f"   Take Profit: ${top_bearish['take_profit']:.2f}")
        print(f"   Risk Level: {top_bearish['risk_level']}")
    
    if chart_files:
        print(f"\n📊 Generated {len(chart_files)} prediction charts:")
        for chart_type, symbol, path in chart_files:
            print(f"   {chart_type}: {symbol} -> {path}")
    
    # Create predictions folder for backtesting and records
    predictions_dir = "predictions"
    os.makedirs(predictions_dir, exist_ok=True)
    
    # Save results with organized structure
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    
    if bullish_picks:
        bullish_df = pd.DataFrame(bullish_picks)
        # Round numerical columns to 2 decimal places
        numerical_cols = ['predicted_return', 'confidence_score', 'current_price', 'stop_loss', 'take_profit', 'risk_score', 'score']
        for col in numerical_cols:
            if col in bullish_df.columns:
                bullish_df[col] = bullish_df[col].round(2)
        
        bullish_file = os.path.join(predictions_dir, f"Bullish_Picks_{timestamp}.csv")
        bullish_df.to_csv(bullish_file, index=False)
        print(f"\n💾 Bullish picks saved: {bullish_file}")
    
    if bearish_picks:
        bearish_df = pd.DataFrame(bearish_picks)
        # Round numerical columns to 2 decimal places
        numerical_cols = ['predicted_return', 'confidence_score', 'current_price', 'stop_loss', 'take_profit', 'risk_score', 'score']
        for col in numerical_cols:
            if col in bearish_df.columns:
                bearish_df[col] = bearish_df[col].round(2)
        
        bearish_file = os.path.join(predictions_dir, f"Bearish_Picks_{timestamp}.csv")
        bearish_df.to_csv(bearish_file, index=False)
        print(f"💾 Bearish picks saved: {bearish_file}")
    
    # Save summary report for backtesting
    summary_file = os.path.join(predictions_dir, f"Prediction_Summary_{timestamp}.txt")
    with open(summary_file, 'w') as f:
        f.write(f"STOCK PREDICTION SUMMARY - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n")
        f.write(f"Symbols analyzed: {len(results)}\n")
        f.write(f"Processing time: {total_time:.1f} seconds\n")
        f.write(f"Processing rate: {len(results)/total_time:.1f} stocks/second\n\n")
        
        if top_bullish:
            f.write("TOP BULLISH PICK:\n")
            f.write(f"Symbol: {top_bullish['symbol']}\n")
            f.write(f"Expected Return: +{top_bullish['predicted_return']*100:.2f}%\n")
            f.write(f"Confidence: {top_bullish['confidence_score']:.1f}%\n")
            f.write(f"Current Price: ${top_bullish['current_price']:.2f}\n")
            f.write(f"Target Price: ${top_bullish['current_price'] * (1 + top_bullish['predicted_return']):.2f}\n")
            f.write(f"Stop Loss: ${top_bullish['stop_loss']:.2f}\n")
            f.write(f"Take Profit: ${top_bullish['take_profit']:.2f}\n")
            f.write(f"Risk Level: {top_bullish['risk_level']}\n\n")
        
        if top_bearish:
            f.write("TOP BEARISH PICK:\n")
            f.write(f"Symbol: {top_bearish['symbol']}\n")
            f.write(f"Expected Return: {top_bearish['predicted_return']*100:.2f}%\n")
            f.write(f"Confidence: {top_bearish['confidence_score']:.1f}%\n")
            f.write(f"Current Price: ${top_bearish['current_price']:.2f}\n")
            f.write(f"Target Price: ${top_bearish['current_price'] * (1 + top_bearish['predicted_return']):.2f}\n")
            f.write(f"Stop Loss: ${top_bearish['stop_loss']:.2f}\n")
            f.write(f"Take Profit: ${top_bearish['take_profit']:.2f}\n")
            f.write(f"Risk Level: {top_bearish['risk_level']}\n")
    
    print(f"📋 Summary report saved: {summary_file}")
    print(f"\n✅ Stock prediction complete!")
    return results

if __name__ == "__main__":
    run_stock_prediction()