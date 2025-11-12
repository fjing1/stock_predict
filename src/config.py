#!/usr/bin/env python3
"""
Configuration settings for stock prediction
"""

# Training period for daily data
TRAIN_PERIOD_DAILY = "2y"

# Target horizon in days
TARGET_HORIZON_D = 3

# Default parameters
DEFAULT_LOOKBACK_DAYS = 252  # 1 year of trading days
DEFAULT_MIN_VOLUME = 100000
DEFAULT_MIN_PRICE = 1.0
DEFAULT_MAX_PRICE = 10000.0