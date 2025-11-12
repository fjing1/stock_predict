# Stock Prediction System

A machine learning-based stock prediction system that analyzes market data to identify stocks with high probability of short-term (3-day) upward movement.

## 🚀 Quick Start

### 1. Set Up Virtual Environment (Recommended)
```bash
# Create virtual environment
python3 -m venv fjing1

# Activate virtual environment
source fjing1/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Alternative: Install with --user flag**
```bash
pip3 install --user -r requirements.txt
```

### 2. Collect Data (First Time Setup)
```bash
# Download intraday data for all symbols (takes ~2 minutes)
python3 update_intraday_allinone.py --symbols-file symbols.txt --rth true
```

### 3. Get Stock Predictions
```bash
# Run the prediction pipeline
python3 pipelines/rank_shortterm.py --topk 10
```

## 📊 Example Output

```
📊 Loaded 98 symbols for analysis
🔄 Building training dataset...
✅ Training dataset: 15420 samples, 11 features
📈 Features: ret_1, ret_2, ret_3, ma20, ma50, rsi14, macd, macd_signal, macd_diff, bb_pct, vol_ratio20
🤖 Training machine learning models...
✅ Model performance - AUC: 0.523, Accuracy: 0.512, R²: 0.089
📊 Scoring current market conditions...

🎯 TOP 10 STOCK PREDICTIONS (3-day horizon):
================================================================================
Rank Symbol   Score    P(Up)    Exp.Ret    Price     
--------------------------------------------------------------------------------
1    NVDA     45.2     65.2%    8.45%      $142.50   
2    AAPL     42.1     62.8%    7.23%      $224.91   
3    MSFT     38.9     59.4%    6.12%      $415.26   
4    GOOGL    35.7     58.1%    5.89%      $166.84   
5    META     33.2     56.9%    5.34%      $563.33   
...

💾 Results saved to: ML_Picks_20251112.xlsx
📈 Analysis complete! Found 87 valid predictions.
🎯 Top stock: NVDA (Score: 45.2)
```

## 🛠️ Usage Options

### Basic Usage
```bash
# Use default symbols from symbols.txt
python3 pipelines/rank_shortterm.py

# Get top 5 predictions
python3 pipelines/rank_shortterm.py --topk 5

# Use specific symbols
python3 pipelines/rank_shortterm.py --symbols "AAPL,MSFT,GOOGL,NVDA,TSLA"
```

### Data Management
```bash
# Update data daily (recommended to run before market open)
python3 update_intraday_allinone.py --symbols-file symbols.txt --rth true

# Update specific symbols only
python3 update_intraday_allinone.py --symbols "AAPL,MSFT,GOOGL" --rth true

# Include pre/post market data
python3 update_intraday_allinone.py --symbols-file symbols.txt --rth false
```

## 📈 How It Works

### 1. Data Collection
- Downloads 3 years of daily data for training
- Collects 7 days of 1-minute intraday data
- Focuses on regular trading hours (9:30 AM - 4:00 PM ET)

### 2. Feature Engineering
- **Technical Indicators**: RSI, MACD, Bollinger Bands, Moving Averages
- **Return Analysis**: 1, 2, 3, 5, 10, 20-day returns
- **Volume Metrics**: 20-day volume ratios
- **Intraday Signals**: VWAP distance, momentum, drift

### 3. Machine Learning
- **Dual Model Approach**:
  - Classification: Probability of 3-day upward movement
  - Regression: Expected 3-day return percentage
- **Validation**: Time series cross-validation
- **Algorithm**: Histogram Gradient Boosting

### 4. Scoring System
```
Score = 100×P(Up) + 300×Expected_Return + 20×Momentum - 10×|VWAP_Distance| + 5×Drift
```

## 📁 Project Structure

```
stock_predict/
├── data/
│   ├── intraday_raw/          # Daily snapshots
│   └── intraday_merged/       # Historical data
├── src/                       # Core library
│   ├── config.py             # Configuration
│   ├── data_daily.py         # Daily data processing
│   ├── data_intraday.py      # Intraday data processing
│   ├── dataset.py            # Training data preparation
│   ├── models.py             # ML models
│   ├── scoring.py            # Prediction scoring
│   └── io_symbols.py         # Symbol management
├── pipelines/
│   └── rank_shortterm.py     # Main prediction pipeline
├── logs/                     # Execution logs
├── symbols.txt               # Default stock list
└── requirements.txt          # Dependencies
```

## 🎯 Covered Stocks

- **Indices**: SPY, QQQ, IWM
- **Sector ETFs**: XLE, XLB, XLI, XLY, XLP, XLV, XLF, XLK, XLC, XLU, XLRE
- **Large Cap**: AAPL, MSFT, NVDA, GOOGL, AMZN, META, TSLA, etc.
- **Total**: 98 symbols (see `symbols.txt`)

## ⚙️ Configuration

Edit [`src/config.py`](src/config.py) to customize:
- Training period (default: 3 years)
- Prediction horizon (default: 3 days)
- Output file location
- Top K results (default: 20)

## 📝 Output Files

- **Excel**: `ML_Picks_YYYYMMDD.xlsx` - Formatted predictions with rankings
- **Logs**: `logs/update_YYYYMMDD.log` - Data collection logs

## 🔄 Automation

### Daily Data Updates
Add to crontab for automatic daily updates:
```bash
# Run at 6 AM ET before market open
0 6 * * 1-5 cd /path/to/stock_predict && python3 update_intraday_allinone.py --symbols-file symbols.txt --rth true
```

### Weekly Predictions
```bash
# Generate predictions every Monday at 7 AM ET
0 7 * * 1 cd /path/to/stock_predict && python3 pipelines/rank_shortterm.py --topk 20
```

## ⚠️ Important Notes

1. **Market Hours**: Best results during regular trading hours (9:30 AM - 4:00 PM ET)
2. **Data Freshness**: Update data daily for best predictions
3. **Internet Required**: Uses Yahoo Finance API for real-time data
4. **No Investment Advice**: This is for educational/research purposes only

## 🐛 Troubleshooting

### Common Issues

**"Missing dependencies"**
```bash
pip install -r requirements.txt
```

**"No training data available"**
- Check internet connection
- Verify symbols in `symbols.txt` are valid
- Run data collection first: `python3 update_intraday_allinone.py --symbols-file symbols.txt`

**"Empty dataframe" errors**
- Market might be closed
- Try with different symbols
- Check if Yahoo Finance is accessible

### Performance Tips

- Use `--rth true` for faster processing (regular hours only)
- Reduce symbol count for quicker results
- Run data updates during off-market hours

## 📊 Model Performance

The system typically achieves:
- **AUC**: 0.52-0.55 (slightly better than random)
- **Accuracy**: 51-53%
- **R²**: 0.08-0.12

*Note: Stock prediction is inherently difficult. These modest improvements over random can still be valuable for portfolio construction.*
