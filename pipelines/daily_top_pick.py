import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import pandas as pd
from datetime import datetime
from src.config import OUTPUT_FILE, DEFAULT_RTH
from src.io_symbols import load_symbols
from src.data_daily import dl_daily
from src.enhanced_features import add_advanced_features, get_enhanced_feature_list
from src.enhanced_models import train_enhanced_models
from src.enhanced_scoring import get_daily_top_pick, enhanced_score_today

def build_enhanced_training_set(symbols):
    """Build training dataset with enhanced features"""
    frames = []
    feature_cols = get_enhanced_feature_list()
    
    print(f"🔄 Processing {len(symbols)} symbols for enhanced training...")
    
    for i, s in enumerate(symbols, 1):
        if i % 20 == 0:
            print(f"   Processed {i}/{len(symbols)} symbols...")
            
        try:
            df = dl_daily(s, "3y")  # 3 years of data
            if df.empty or len(df) < 100:
                continue
                
            df = add_advanced_features(df).dropna()
            if df.empty or len(df) < 50:
                continue
            
            # Get available features
            available_features = [f for f in feature_cols if f in df.columns]
            if len(available_features) < 10:  # Need minimum features
                continue
            
            # Create target variables
            df["fwd_ret_3d"] = df["Close"].shift(-3) / df["Close"] - 1
            df["y_up_3d"] = (df["fwd_ret_3d"] > 0).astype(int)
            
            # Select features and targets
            dsub = df[available_features + ["fwd_ret_3d", "y_up_3d"]].dropna().copy()
            if dsub.empty:
                continue
                
            dsub["symbol"] = s
            frames.append(dsub)
            
        except Exception as e:
            print(f"⚠️ Error processing {s}: {e}")
            continue
    
    if not frames:
        return pd.DataFrame(), []
    
    big_df = pd.concat(frames, ignore_index=True)
    final_features = [col for col in feature_cols if col in big_df.columns]
    
    print(f"✅ Enhanced training dataset: {len(big_df)} samples, {len(final_features)} features")
    return big_df, final_features

def main():
    parser = argparse.ArgumentParser(description="Enhanced Stock Prediction - Daily Top Pick")
    parser.add_argument("--symbols", type=str, default="", 
                       help="Comma-separated stock symbols")
    parser.add_argument("--symbols-file", type=str, default="symbols.txt",
                       help="File containing stock symbols")
    parser.add_argument("--rth", type=str, default=str(DEFAULT_RTH).lower(),
                       help="Use regular trading hours only (true/false)")
    parser.add_argument("--output", type=str, default="",
                       help="Output file path")
    parser.add_argument("--top-n", type=int, default=5,
                       help="Number of top candidates to analyze")
    
    args = parser.parse_args()
    
    # Load symbols
    symbols = load_symbols(args.symbols, args.symbols_file)
    print(f"📊 Loaded {len(symbols)} symbols for enhanced analysis")
    
    # Build enhanced training dataset
    print("🔄 Building enhanced training dataset...")
    train_df, feature_cols = build_enhanced_training_set(symbols)
    
    if train_df.empty:
        print("❌ No training data available. Please check your symbols and data.")
        return
    
    print(f"📈 Enhanced features: {', '.join(feature_cols[:10])}{'...' if len(feature_cols) > 10 else ''}")
    
    # Train enhanced models
    print("🤖 Training enhanced ensemble models...")
    model, metrics = train_enhanced_models(train_df, feature_cols)
    
    print(f"✅ Enhanced model performance:")
    print(f"   AUC: {metrics['auc']:.3f} ± {metrics['auc_std']:.3f}")
    print(f"   Accuracy: {metrics['acc']:.3f} ± {metrics['acc_std']:.3f}")
    print(f"   R²: {metrics['r2']:.3f} ± {metrics['r2_std']:.3f}")
    
    # Get enhanced scoring
    rth_only = args.rth.strip().lower() in ("1", "true", "yes", "y")
    print("📊 Analyzing current market with enhanced scoring...")
    
    top_candidates = enhanced_score_today(symbols, feature_cols, model, rth_only, args.top_n)
    
    if top_candidates.empty:
        print("❌ No qualifying candidates found with current quality filters.")
        return
    
    # Get the daily top pick
    daily_pick = get_daily_top_pick(symbols, feature_cols, model, rth_only)
    
    print(f"\n🎯 TOP {len(top_candidates)} ENHANCED CANDIDATES:")
    print("=" * 100)
    print(f"{'Rank':<4} {'Symbol':<8} {'Score':<8} {'P(Up)':<8} {'Exp.Ret':<10} {'Conf':<8} {'Tech':<6} {'Price':<10}")
    print("-" * 100)
    
    for i, (_, row) in enumerate(top_candidates.iterrows(), 1):
        print(f"{i:<4} {row['symbol']:<8} {row['final_score']:<8.1f} {row['p_up_3d']:<8.1%} "
              f"{row['exp_ret_3d']:<10.2%} {row['confidence']:<8.2f} {row['tech_strength']:<6.0f} "
              f"${row['close']:<9.2f}")
    
    # Highlight the daily top pick
    if daily_pick is not None:
        print(f"\n🌟 DAILY TOP PICK: {daily_pick['symbol']}")
        print("=" * 50)
        print(f"Score: {daily_pick['final_score']:.1f}")
        print(f"Probability Up (3d): {daily_pick['p_up_3d']:.1%}")
        print(f"Expected Return (3d): {daily_pick['exp_ret_3d']:.2%}")
        print(f"Confidence: {daily_pick['confidence']:.2f}")
        print(f"Technical Strength: {daily_pick['tech_strength']:.0f}/100")
        print(f"Current Price: ${daily_pick['close']:.2f}")
        print(f"Risk-Adj Return: {daily_pick['risk_adj_return']:.2f}")
        print(f"Volume Ratio: {daily_pick['volume_ratio']:.2f}")
        
        # Risk assessment
        print(f"\n📊 RISK ASSESSMENT:")
        risk_level = assess_risk_level(daily_pick)
        print(f"Risk Level: {risk_level}")
        
        # Position sizing suggestion
        position_size = suggest_position_size(daily_pick)
        print(f"Suggested Position Size: {position_size}")
        
    else:
        print(f"\n⚠️ NO DAILY TOP PICK")
        print("No candidate meets the strict quality criteria for today.")
        print("Consider waiting for better opportunities.")
    
    # Save results
    output_file = args.output if args.output else f"Enhanced_Picks_{datetime.now().strftime('%Y%m%d')}.xlsx"
    
    try:
        # Prepare export data
        export_df = top_candidates.copy()
        export_df.insert(0, 'rank', range(1, len(export_df) + 1))
        
        # Format for readability
        export_df['p_up_3d'] = export_df['p_up_3d'].apply(lambda x: f"{x:.1%}")
        export_df['exp_ret_3d'] = export_df['exp_ret_3d'].apply(lambda x: f"{x:.2%}")
        export_df['final_score'] = export_df['final_score'].round(1)
        export_df['confidence'] = export_df['confidence'].round(3)
        export_df['close'] = export_df['close'].round(2)
        
        # Rename columns
        export_df.columns = [
            'Rank', 'Symbol', 'Probability_Up_3d', 'Expected_Return_3d', 
            'Base_Score', 'Confidence', 'Final_Score', 'Current_Price',
            'Volume_Ratio', 'Risk_Adj_Return', 'Tech_Strength', 'RSI_14',
            'Intraday_Momentum', 'VWAP_Distance'
        ]
        
        export_df.to_excel(output_file, index=False)
        print(f"\n💾 Enhanced results saved to: {output_file}")
        
    except Exception as e:
        csv_file = output_file.replace('.xlsx', '.csv')
        top_candidates.to_csv(csv_file, index=False)
        print(f"\n💾 Results saved to: {csv_file} (Excel failed: {e})")
    
    print(f"\n📈 Enhanced analysis complete!")
    if daily_pick is not None:
        print(f"🎯 Daily top pick: {daily_pick['symbol']} (Score: {daily_pick['final_score']:.1f})")
    print(f"📊 Analyzed {len(top_candidates)} high-quality candidates")

def assess_risk_level(pick):
    """Assess risk level of the pick"""
    risk_score = 0
    
    # Volatility risk
    if pick.get('risk_adj_return', 0) < 0.5:
        risk_score += 2
    elif pick.get('risk_adj_return', 0) < 1.0:
        risk_score += 1
    
    # Technical risk
    if pick['tech_strength'] < 75:
        risk_score += 1
    
    # Confidence risk
    if pick['confidence'] < 0.85:
        risk_score += 1
    
    # Price momentum risk
    if abs(pick.get('intraday_mom', 0)) > 0.05:
        risk_score += 1
    
    if risk_score <= 1:
        return "LOW"
    elif risk_score <= 3:
        return "MEDIUM"
    else:
        return "HIGH"

def suggest_position_size(pick):
    """Suggest position size based on confidence and risk"""
    base_size = 0.02  # 2% base position
    
    # Adjust for confidence
    confidence_multiplier = pick['confidence']
    
    # Adjust for expected return
    return_multiplier = min(2.0, 1 + pick['exp_ret_3d'] * 10)
    
    # Adjust for technical strength
    tech_multiplier = pick['tech_strength'] / 100
    
    suggested_size = base_size * confidence_multiplier * return_multiplier * tech_multiplier
    suggested_size = min(0.05, max(0.005, suggested_size))  # Cap between 0.5% and 5%
    
    return f"{suggested_size:.1%} of portfolio"

if __name__ == "__main__":
    main()