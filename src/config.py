from datetime import datetime
from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
LOG_DIR = ROOT / "logs"
OUTPUT_FILE = ROOT / f"ML_Picks_{datetime.now().strftime('%Y%m%d')}.xlsx"

# yfinance / model parameters
TRAIN_PERIOD_DAILY = "3y"   # 3 years daily data for training
TARGET_HORIZON_D = 3        # predict next 3 days
INTRADAY_INTERVAL = "1m"
INTRADAY_PERIOD = "7d"
DEFAULT_RTH = True           # regular trading hours only
DEFAULT_TOPK = 20
