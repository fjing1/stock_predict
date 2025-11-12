import multiprocessing as mp
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

def process_symbol_data(args):
    """Process a single symbol's data - designed for parallel execution"""
    symbol, period = args
    try:
        from .data_daily import dl_daily
        from .enhanced_features import add_advanced_features
        
        df = dl_daily(symbol, period)
        if df.empty or len(df) < 100:
            return None
            
        df = add_advanced_features(df).dropna()
        if df.empty or len(df) < 50:
            return None
        
        # Create target variables
        df["fwd_ret_3d"] = df["Close"].shift(-3) / df["Close"] - 1
        df["y_up_3d"] = (df["fwd_ret_3d"] > 0).astype(int)
        
        # Add symbol identifier
        df["symbol"] = symbol
        
        return df
        
    except Exception as e:
        print(f"⚠️ Error processing {symbol}: {e}")
        return None

def parallel_data_processing(symbols, period="3y", max_workers=None):
    """Process multiple symbols in parallel"""
    if max_workers is None:
        max_workers = min(mp.cpu_count(), len(symbols))
    
    print(f"🚀 Using {max_workers} parallel workers for data processing...")
    
    # Prepare arguments
    args_list = [(symbol, period) for symbol in symbols]
    
    frames = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_symbol = {
            executor.submit(process_symbol_data, args): args[0] 
            for args in args_list
        }
        
        # Collect results as they complete
        completed = 0
        for future in as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                result = future.result()
                if result is not None:
                    frames.append(result)
                completed += 1
                if completed % 10 == 0:
                    print(f"   Processed {completed}/{len(symbols)} symbols...")
            except Exception as e:
                print(f"⚠️ Error with {symbol}: {e}")
    
    return frames

def process_scoring_batch(args):
    """Process a batch of symbols for scoring - parallel execution"""
    symbols_batch, feature_cols, model_data, rth_only = args
    
    try:
        from .enhanced_scoring import enhanced_score_today
        from .enhanced_models import EnhancedEnsembleModel
        
        # Reconstruct model (simplified for parallel processing)
        # Note: In practice, you'd want to serialize/deserialize the full model
        # For now, we'll use a simplified approach
        
        results = []
        for symbol in symbols_batch:
            # Process individual symbol scoring here
            # This is a simplified version - you'd implement the full scoring logic
            pass
            
        return results
        
    except Exception as e:
        print(f"⚠️ Error in scoring batch: {e}")
        return []

def optimize_model_training(train_df, feature_cols, max_workers=None):
    """Optimize model training with parallel processing"""
    if max_workers is None:
        max_workers = min(mp.cpu_count() // 2, 4)  # Use half cores for training
    
    from .enhanced_models import EnhancedEnsembleModel
    
    # Set n_jobs for sklearn models
    model = EnhancedEnsembleModel()
    
    # Override model creation to use parallel processing
    original_create = model._create_base_models
    
    def create_parallel_models():
        models = original_create()
        # Set n_jobs for models that support it
        for name, model_obj in models.items():
            if hasattr(model_obj, 'n_jobs'):
                model_obj.n_jobs = max_workers
        return models
    
    model._create_base_models = create_parallel_models
    
    return model

def batch_process_symbols(symbols, batch_size=20):
    """Process symbols in batches to manage memory"""
    for i in range(0, len(symbols), batch_size):
        yield symbols[i:i + batch_size]