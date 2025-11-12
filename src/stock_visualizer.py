#!/usr/bin/env python3
"""
Stock Price Prediction Visualizer
Creates charts for top bullish and bearish stock picks only
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class StockPredictionVisualizer:
    """Creates visualizations for top stock predictions"""
    
    def __init__(self, figsize=(15, 10)):
        self.figsize = figsize
        self.colors = {
            'bullish': '#00ff88',
            'bearish': '#ff4444', 
            'neutral': '#ffaa00',
            'price': '#2E86AB',
            'volume': '#A23B72',
            'prediction': '#ff6b35'
        }
    
    def create_prediction_chart(self, symbol, df, prediction_data, save_path=None):
        """
        Create prediction chart showing predicted stock price path
        
        Args:
            symbol: Stock symbol
            df: Historical price data with indicators
            prediction_data: Dictionary with prediction results
            save_path: Optional path to save the chart
        """
        
        # Create figure with subplots
        fig = plt.figure(figsize=self.figsize)
        gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], hspace=0.3, wspace=0.3)
        
        # Main price chart with prediction path
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_price_and_prediction_path(ax1, symbol, df, prediction_data)
        
        # Volume chart
        ax2 = fig.add_subplot(gs[1, :])
        self._plot_volume(ax2, df)
        
        # Technical indicators
        ax3 = fig.add_subplot(gs[2, 0])
        self._plot_rsi(ax3, df)
        
        ax4 = fig.add_subplot(gs[2, 1])
        self._plot_prediction_summary(ax4, prediction_data)
        
        # Add title and metadata
        self._add_chart_title(fig, symbol, prediction_data)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Chart saved: {save_path}")
        
        return fig
    
    def _plot_price_and_prediction_path(self, ax, symbol, df, prediction_data):
        """Plot main price chart with predicted price path"""
        
        # Ensure we have Date column
        if 'Date' not in df.columns and df.index.name != 'Date':
            df = df.reset_index()
        
        dates = pd.to_datetime(df['Date']) if 'Date' in df.columns else df.index
        
        # Plot historical prices (last 30 days)
        recent_df = df.tail(30)
        recent_dates = dates[-30:] if len(dates) > 30 else dates
        
        ax.plot(recent_dates, recent_df['Close'], color=self.colors['price'], 
               linewidth=3, label='Historical Price', marker='o', markersize=3)
        
        # Plot moving averages
        if 'sma_20' in recent_df.columns:
            ax.plot(recent_dates, recent_df['sma_20'], color='orange', 
                   alpha=0.8, linewidth=2, label='SMA 20')
        
        # Generate and plot prediction path
        prediction_path = self._generate_prediction_path(recent_df, prediction_data)
        if prediction_path is not None:
            pred_dates, pred_prices, confidence_upper, confidence_lower = prediction_path
            
            # Plot prediction line
            ax.plot(pred_dates, pred_prices, color=self.colors['prediction'], 
                   linewidth=4, linestyle='--', marker='s', markersize=6,
                   label='🎯 Predicted Path')
            
            # Plot confidence interval
            ax.fill_between(pred_dates, confidence_lower, confidence_upper, 
                           color=self.colors['prediction'], alpha=0.3, 
                           label='Confidence Band')
        
        # Add key levels
        self._add_key_levels(ax, recent_df, prediction_data)
        
        # Formatting
        direction = "📈 BULLISH" if prediction_data.get('predicted_return', 0) > 0 else "📉 BEARISH"
        ax.set_title(f'{symbol} - {direction} Price Prediction', fontsize=16, fontweight='bold')
        ax.set_ylabel('Price ($)', fontsize=12)
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    def _generate_prediction_path(self, df, prediction_data, days_ahead=5):
        """Generate predicted price path for next 5 days"""
        
        try:
            current_price = float(df['Close'].iloc[-1])
            predicted_return = prediction_data.get('predicted_return', 0.0)
            confidence = prediction_data.get('confidence', 0.5)
            
            # Create future dates (skip weekends)
            last_date = pd.to_datetime(df['Date'].iloc[-1]) if 'Date' in df.columns else df.index[-1]
            future_dates = []
            current_date = last_date
            
            for i in range(days_ahead):
                current_date += timedelta(days=1)
                # Skip weekends
                while current_date.weekday() >= 5:
                    current_date += timedelta(days=1)
                future_dates.append(current_date)
            
            # Generate realistic prediction path
            target_price = current_price * (1 + predicted_return)
            
            # Create smooth path to target with some volatility
            price_path = []
            for i in range(days_ahead):
                progress = (i + 1) / days_ahead
                # Smooth interpolation to target
                predicted_price = current_price + (target_price - current_price) * progress
                
                # Add small random walk for realism
                if i > 0:
                    volatility = 0.01 * np.random.normal(0, 1)
                    predicted_price += predicted_price * volatility
                
                price_path.append(predicted_price)
            
            # Calculate confidence intervals based on model confidence
            volatility_factor = 0.03 * (1 - confidence)  # Lower confidence = wider bands
            confidence_upper = [p * (1 + volatility_factor) for p in price_path]
            confidence_lower = [p * (1 - volatility_factor) for p in price_path]
            
            return future_dates, price_path, confidence_upper, confidence_lower
            
        except Exception as e:
            print(f"Warning: Could not generate prediction path: {e}")
            return None
    
    def _add_key_levels(self, ax, df, prediction_data):
        """Add key price levels to the chart"""
        
        try:
            current_price = float(df['Close'].iloc[-1])
            
            # Current price line
            ax.axhline(y=current_price, color='black', linestyle='-', 
                      alpha=0.8, linewidth=2, label=f'Current: ${current_price:.2f}')
            
            # Stop loss level
            if 'stop_loss' in prediction_data:
                stop_loss = prediction_data['stop_loss']
                ax.axhline(y=stop_loss, color='red', linestyle='--', alpha=0.8, 
                          linewidth=2, label=f'🛑 Stop Loss: ${stop_loss:.2f}')
            
            # Take profit level
            if 'take_profit' in prediction_data:
                take_profit = prediction_data['take_profit']
                ax.axhline(y=take_profit, color='green', linestyle='--', alpha=0.8, 
                          linewidth=2, label=f'🎯 Take Profit: ${take_profit:.2f}')
            
            # Target price
            if 'predicted_return' in prediction_data:
                target_price = current_price * (1 + prediction_data['predicted_return'])
                ax.axhline(y=target_price, color=self.colors['prediction'], 
                          linestyle='-', alpha=0.9, linewidth=3,
                          label=f'🚀 Target: ${target_price:.2f}')
            
        except Exception as e:
            pass
    
    def _plot_volume(self, ax, df):
        """Plot volume chart with color coding"""
        
        dates = pd.to_datetime(df['Date']) if 'Date' in df.columns else df.index
        recent_df = df.tail(30)
        recent_dates = dates[-30:] if len(dates) > 30 else dates
        
        # Color volume bars based on price movement
        colors = []
        for i in range(len(recent_df)):
            if i == 0:
                colors.append(self.colors['neutral'])
            else:
                if recent_df['Close'].iloc[i] > recent_df['Close'].iloc[i-1]:
                    colors.append(self.colors['bullish'])
                else:
                    colors.append(self.colors['bearish'])
        
        ax.bar(recent_dates, recent_df['Volume'], color=colors, alpha=0.7, width=0.8)
        ax.set_ylabel('Volume', fontsize=10)
        ax.set_title('📊 Trading Volume (30 days)', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Format volume numbers
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
    
    def _plot_rsi(self, ax, df):
        """Plot RSI indicator"""
        
        if 'rsi' not in df.columns:
            ax.text(0.5, 0.5, 'RSI data not available', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_title('RSI Indicator', fontsize=12)
            return
        
        dates = pd.to_datetime(df['Date']) if 'Date' in df.columns else df.index
        recent_df = df.tail(30)
        recent_dates = dates[-30:] if len(dates) > 30 else dates
        
        # Plot RSI
        ax.plot(recent_dates, recent_df['rsi'], color='purple', linewidth=2, marker='o', markersize=2)
        
        # Add overbought/oversold levels
        ax.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Overbought (70)')
        ax.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Oversold (30)')
        ax.fill_between(recent_dates, 30, 70, alpha=0.1, color='gray')
        
        # Current RSI value
        current_rsi = recent_df['rsi'].iloc[-1]
        rsi_status = "🔴 Overbought" if current_rsi > 70 else "🟢 Oversold" if current_rsi < 30 else "🟡 Neutral"
        
        ax.set_ylabel('RSI', fontsize=10)
        ax.set_title(f'📈 RSI: {current_rsi:.1f} ({rsi_status})', fontsize=12)
        ax.set_ylim(0, 100)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    def _plot_prediction_summary(self, ax, prediction_data):
        """Plot prediction summary as visual metrics"""
        
        # Clear the axes
        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # Create summary text
        summary_lines = []
        
        # Expected return
        if 'predicted_return' in prediction_data:
            return_pct = prediction_data['predicted_return'] * 100
            direction_emoji = "📈" if return_pct > 0 else "📉"
            summary_lines.append(f"{direction_emoji} Expected Return: {return_pct:+.2f}%")
        
        # Confidence
        if 'confidence' in prediction_data:
            conf_pct = prediction_data['confidence'] * 100
            conf_emoji = "🟢" if conf_pct > 70 else "🟡" if conf_pct > 50 else "🔴"
            summary_lines.append(f"{conf_emoji} Confidence: {conf_pct:.1f}%")
        
        # Risk level
        if 'risk_level' in prediction_data:
            risk = prediction_data['risk_level']
            risk_emoji = "🟢" if risk == "LOW" else "🟡" if risk == "MEDIUM" else "🔴"
            summary_lines.append(f"{risk_emoji} Risk Level: {risk}")
        
        # Holding period
        if 'holding_period' in prediction_data:
            days = prediction_data['holding_period']
            summary_lines.append(f"⏰ Holding Period: {days} days")
        
        # Display summary
        summary_text = "\n".join(summary_lines)
        ax.text(0.05, 0.95, "🎯 PREDICTION SUMMARY", fontsize=14, fontweight='bold', 
               transform=ax.transAxes, verticalalignment='top')
        ax.text(0.05, 0.75, summary_text, fontsize=12, transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
               facecolor="lightblue", alpha=0.3))
    
    def _add_chart_title(self, fig, symbol, prediction_data):
        """Add main title to the chart"""
        
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M')
        return_pct = prediction_data.get('predicted_return', 0) * 100
        confidence = prediction_data.get('confidence', 0) * 100
        
        title = f'{symbol} Stock Prediction - {return_pct:+.2f}% Expected Return ({confidence:.1f}% Confidence) - {current_time}'
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)

def create_top_picks_charts(bullish_pick, bearish_pick, stock_data, output_dir="predictions/charts"):
    """
    Create charts for the top bullish and bearish picks only
    
    Args:
        bullish_pick: Top bullish prediction result
        bearish_pick: Top bearish prediction result  
        stock_data: Dictionary of stock data
        output_dir: Directory to save charts
    """
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    visualizer = StockPredictionVisualizer()
    chart_files = []
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    
    # Create chart for top bullish pick
    if bullish_pick and bullish_pick['symbol'] in stock_data:
        symbol = bullish_pick['symbol']
        df = stock_data[symbol].copy()
        
        prediction_data = {
            'predicted_return': bullish_pick['predicted_return'],
            'confidence': bullish_pick['confidence_score'] / 100,
            'stop_loss': bullish_pick['stop_loss'],
            'take_profit': bullish_pick['take_profit'],
            'holding_period': 3,
            'risk_level': bullish_pick['risk_level']
        }
        
        filename = f"TOP_BULLISH_{symbol}_{timestamp}.png"
        save_path = os.path.join(output_dir, filename)
        
        fig = visualizer.create_prediction_chart(symbol, df, prediction_data, save_path)
        chart_files.append(('BULLISH', symbol, save_path))
        
        import matplotlib.pyplot as plt
        plt.close(fig)
    
    # Create chart for top bearish pick
    if bearish_pick and bearish_pick['symbol'] in stock_data:
        symbol = bearish_pick['symbol']
        df = stock_data[symbol].copy()
        
        prediction_data = {
            'predicted_return': bearish_pick['predicted_return'],
            'confidence': bearish_pick['confidence_score'] / 100,
            'stop_loss': bearish_pick['stop_loss'],
            'take_profit': bearish_pick['take_profit'],
            'holding_period': 3,
            'risk_level': bearish_pick['risk_level']
        }
        
        filename = f"TOP_BEARISH_{symbol}_{timestamp}.png"
        save_path = os.path.join(output_dir, filename)
        
        fig = visualizer.create_prediction_chart(symbol, df, prediction_data, save_path)
        chart_files.append(('BEARISH', symbol, save_path))
        
        import matplotlib.pyplot as plt
        plt.close(fig)
    
    return chart_files

if __name__ == "__main__":
    print("📊 Stock Prediction Visualizer")
    print("Use create_top_picks_charts() to generate charts for top bullish and bearish picks")