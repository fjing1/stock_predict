# 🚀 Stock Predictor - Streamlined Version

A high-performance stock prediction system optimized for M3 Max that generates top bullish and bearish picks with professional visualizations.

## 🎯 Features

- **Single Command Execution**: Run everything with one simple command
- **M3 Max Optimized**: 64 parallel workers for maximum performance
- **Professional Visualizations**: Automatic chart generation with predicted price paths
- **Top Pick Selection**: Identifies single best bullish and bearish opportunities
- **Risk Management**: Includes stop-loss and take-profit levels
- **Clean Output**: Only keeps the latest results

## 🚀 Quick Start

```bash
# Run the stock predictor
fjing1/bin/python stock_predictor.py
```

That's it! The system will:
1. Analyze 231 stocks in ~10 seconds
2. Generate top bullish and bearish picks
3. Create professional prediction charts
4. Save results to CSV files

## 📊 Output

### Latest Results (2025-11-11 23:27)

**📈 TOP BULLISH PICK: KLAC**
- Expected Return: +2.48%
- Confidence: 72.5%
- Current Price: $312.42
- Target Price: $320.17
- Risk Level: LOW

**📉 TOP BEARISH PICK: RL**
- Expected Return: -1.92%
- Confidence: 71.9%
- Current Price: $101.28
- Target Price: $99.33
- Risk Level: LOW

### Generated Files
- `Top_Bullish_Picks_YYYYMMDD_HHMM.csv` - Top 10 bullish opportunities
- `Top_Bearish_Picks_YYYYMMDD_HHMM.csv` - Top 10 bearish opportunities
- `charts/TOP_BULLISH_[SYMBOL]_YYYYMMDD_HHMM.png` - Bullish prediction chart
- `charts/TOP_BEARISH_[SYMBOL]_YYYYMMDD_HHMM.png` - Bearish prediction chart

## 📈 Chart Features

Each prediction chart includes:
- **Historical Price Data**: Last 30 days with moving averages
- **Predicted Price Path**: 5-day forward projection with confidence bands
- **Key Levels**: Current price, stop-loss, take-profit, target price
- **Technical Indicators**: RSI analysis with overbought/oversold levels
- **Volume Analysis**: Color-coded trading volume
- **Prediction Summary**: Expected return, confidence, risk level

## 🏗️ System Architecture

### Core Components
- `stock_predictor.py` - Main execution script
- `src/data_daily.py` - Data fetching and processing
- `src/enhanced_features.py` - Technical indicator calculations
- `src/enhanced_models.py` - Prediction models
- `src/stock_visualizer.py` - Chart generation
- `symbols_expanded.txt` - 231 validated stock symbols

### Performance
- **Processing Speed**: ~21 stocks/second
- **Parallel Workers**: 64 (M3 Max optimized)
- **Success Rate**: 90%+ data fetch success
- **Analysis Time**: ~10 seconds for 231 stocks

## 🎯 Trading Signals

The system uses an advanced scoring algorithm that considers:
- **Momentum**: 3-day and 5-day rate of change
- **RSI**: Relative Strength Index for overbought/oversold conditions
- **Volume**: Trading volume ratios vs. historical averages
- **Trend**: Price position relative to moving averages
- **Volatility**: Average True Range for risk assessment

## 📋 Requirements

- Python 3.9+ with virtual environment `fjing1`
- Required packages: pandas, numpy, yfinance, matplotlib, seaborn, scikit-learn
- Internet connection for real-time data fetching

## 🔧 Installation

```bash
# Install required packages
fjing1/bin/pip install pandas numpy yfinance matplotlib seaborn scikit-learn
```

## 📝 Usage Notes

- Run daily for fresh predictions
- Charts are automatically saved to `charts/` directory
- Old results are automatically cleaned up
- System optimized for short-term trading (1-3 day holds)
- Risk levels: LOW (safest), MEDIUM (moderate), HIGH (aggressive)

---

**Last Updated**: 2025-11-11 23:27  
**Performance**: 208 stocks analyzed in 9.7 seconds (21.4 stocks/second)
