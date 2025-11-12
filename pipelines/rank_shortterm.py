import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import pandas as pd
from src.config import OUTPUT_FILE, DEFAULT_TOPK, DEFAULT_RTH
from src.io_symbols import load_symbols
from src.dataset import build_training_set
from src.models import train_models
from src.scoring import score_today

def main():
    parser = argparse.ArgumentParser(description="Stock Prediction Pipeline - Rank stocks for short-term performance")
    parser.add_argument("--symbols", type=str, default="", 
                       help="Comma-separated stock symbols (e.g., AAPL,MSFT,GOOGL)")
    parser.add_argument("--symbols-file", type=str, default="symbols.txt",
                       help="File containing stock symbols (one per line)")
    parser.add_argument("--topk", type=int, default=DEFAULT_TOPK,
                       help="Number of top stocks to output")
    parser.add_argument("--rth", type=str, default=str(DEFAULT_RTH).lower(),
                       help="Use regular trading hours only (true/false)")
    parser.add_argument("--output", type=str, default="",
                       help="Output file path (default: ML_Picks_YYYYMMDD.xlsx)")
    
    args = parser.parse_args()
    
    # Load symbols
    symbols = load_symbols(args.symbols, args.symbols_file)
    print(f"📊 Loaded {len(symbols)} symbols for analysis")
    
    # Build training dataset
    print("🔄 Building training dataset...")
    train_df, feature_cols = build_training_set(symbols)
    if train_df.empty:
        print("❌ No training data available. Please check your symbols and data.")
        return
    
    print(f"✅ Training dataset: {len(train_df)} samples, {len(feature_cols)} features")
    print(f"📈 Features: {', '.join(feature_cols)}")
    
    # Train models
    print("🤖 Training machine learning models...")
    clf, rgr, metrics = train_models(train_df, feature_cols)
    print(f"✅ Model performance - AUC: {metrics['auc']:.3f}, Accuracy: {metrics['acc']:.3f}, R²: {metrics['r2']:.3f}")
    
    # Score current stocks
    rth_only = args.rth.strip().lower() in ("1", "true", "yes", "y")
    print("📊 Scoring current market conditions...")
    scores_df = score_today(symbols, feature_cols, clf, rgr, rth_only)
    
    if scores_df.empty:
        print("❌ No scoring data available. Please check market hours and data availability.")
        return
    
    # Get top K predictions
    topk = min(args.topk, len(scores_df))
    top_stocks = scores_df.head(topk)
    
    # Display results
    print(f"\n🎯 TOP {topk} STOCK PREDICTIONS (3-day horizon):")
    print("=" * 80)
    print(f"{'Rank':<4} {'Symbol':<8} {'Score':<8} {'P(Up)':<8} {'Exp.Ret':<10} {'Price':<10}")
    print("-" * 80)
    
    for i, (_, row) in enumerate(top_stocks.iterrows(), 1):
        print(f"{i:<4} {row['symbol']:<8} {row['score']:<8.1f} {row['p_up_3d']:<8.1%} "
              f"{row['exp_ret_3d']:<10.2%} ${row['close']:<9.2f}")
    
    # Save to file
    output_file = args.output if args.output else OUTPUT_FILE
    try:
        # Add ranking and format for Excel
        top_stocks_export = top_stocks.copy()
        top_stocks_export.insert(0, 'rank', range(1, len(top_stocks_export) + 1))
        top_stocks_export['p_up_3d'] = top_stocks_export['p_up_3d'].apply(lambda x: f"{x:.1%}")
        top_stocks_export['exp_ret_3d'] = top_stocks_export['exp_ret_3d'].apply(lambda x: f"{x:.2%}")
        top_stocks_export['score'] = top_stocks_export['score'].round(1)
        top_stocks_export['close'] = top_stocks_export['close'].round(2)
        
        # Rename columns for better readability
        top_stocks_export.columns = ['Rank', 'Symbol', 'Probability_Up_3d', 'Expected_Return_3d', 'Score', 'Current_Price']
        
        top_stocks_export.to_excel(output_file, index=False)
        print(f"\n💾 Results saved to: {output_file}")
    except Exception as e:
        # Fallback to CSV if Excel fails
        csv_file = output_file.replace('.xlsx', '.csv')
        top_stocks.to_csv(csv_file, index=False)
        print(f"\n💾 Results saved to: {csv_file} (Excel failed: {e})")
    
    print(f"\n📈 Analysis complete! Found {len(scores_df)} valid predictions.")
    print(f"🎯 Top stock: {top_stocks.iloc[0]['symbol']} (Score: {top_stocks.iloc[0]['score']:.1f})")

if __name__ == "__main__":
    main()
