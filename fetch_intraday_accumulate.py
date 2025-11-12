# -*- coding: utf-8 -*-
"""
使用 yfinance 下载最近7天的1分钟数据，并日积月累地合并保存为本地历史库
- 原始快照：data/intraday_raw/{symbol}/{YYYYMMDD}.parquet
- 累计合并：data/intraday_merged/{symbol}.parquet（自动去重、排序）
- 默认保存含盘前/盘后；可选仅保留常规时段（09:30-16:00）

运行示例：
    python fetch_intraday_accumulate.py --symbols AAPL,MSFT,SPY --interval 1m --rth true
"""

import argparse
from datetime import datetime, date
from pathlib import Path
import time

import pandas as pd
import yfinance as yf

# ---------------- 配置默认参数 ----------------
DEFAULT_SYMBOLS = symbols = [
    # === 指数与板块ETF ===
    "SPY", "QQQ", "IWM",
    "XLE", "XLB", "XLI", "XLY", "XLP", "XLV", "XLF", "XLK", "XLC", "XLU", "XLRE",

    # === 前20大市值公司 ===
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "BRK-B", "TSLA",
    "LLY", "AVGO", "JPM", "V", "UNH", "MA", "PG", "XOM", "HD", "ORCL", "COST", "JNJ",

    # === 第二梯队 ===
    "ADBE", "CRM", "INTC", "AMD", "CSCO", "NOW", "INTU", "TXN",
    "NFLX", "DIS", "ABBV", "MRK", "NVO", "PFE", "TMO",
    "KO", "PEP", "MCD", "WMT", "NKE", "SBUX",
    "CAT", "GE", "UPS", "DE",
    "BAC", "GS", "MS", "BLK",
    "CVX", "COP", "QCOM", "NEE",

    "ADP", "SHOP", "SNOW", "PANW", "PLTR", "UBER", "ABNB", "ZM",
    "MU", "ASML", "NXPI",
    "C", "AXP", "SCHW", "PYPL",
    "HON", "LMT", "RTX", "BA", "MMM",
    "BMY", "AMGN", "GILD", "REGN",
    "SLB", "EOG",
    "LIN", "APD",
    "LOW", "TGT", "BKNG"
]


DEFAULT_INTERVAL = "1m"   # 1m | 2m | 5m | 15m | 30m | 60m | 90m
DEFAULT_PERIOD = "7d"     # 1分钟最大支持7d
DEFAULT_RTH = False       # 是否只保留常规交易时段 09:30-16:00（美东）
DEFAULT_SLEEP = 0.3       # 每只股票请求之间的间隔，避免被限流
# ============ 参数设置 ============


# 目录
RAW_DIR = Path("data/intraday_raw")
MERGED_DIR = Path("data/intraday_merged")
for d in [RAW_DIR, MERGED_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def save_df(df: pd.DataFrame, path: Path):
    """优先保存 parquet；若无 pyarrow 则回退 csv。"""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
    except Exception:
        # 回退到 CSV
        csv_path = path.with_suffix(".csv")
        df.to_csv(csv_path)
        print(f"⚠️ 未安装 pyarrow 或写入失败，已回退 CSV: {csv_path.name}")


def load_df(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        # 尝试CSV
        csv_path = path.with_suffix(".csv")
        if csv_path.exists():
            return pd.read_csv(csv_path, index_col=0, parse_dates=True)
        return None


def download_intraday(symbol: str, interval: str = DEFAULT_INTERVAL, period: str = DEFAULT_PERIOD,
                      rth_only: bool = DEFAULT_RTH) -> pd.DataFrame:
    """
    从 yfinance 下载最近 period 的分钟数据（默认 7d 的 1m）。
    返回带时区的 DataFrame（索引为时间）
    """
    df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=False)
    if df is None or df.empty:
        return pd.DataFrame()

    # yfinance 返回通常为 UTC；转美东，便于对齐交易时段
    try:
        # 如果已带tz，直接转换；若无tz，先localize UTC 再转
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC").tz_convert("America/New_York")
        else:
            df.index = df.index.tz_convert("America/New_York")
    except Exception:
        # 兜底：不处理时区
        pass

    if rth_only:
        # 仅保留常规交易时段
        try:
            df = df.between_time("09:30", "16:00")
        except Exception:
            pass

    # 去掉完全重复的索引（有时API会重复）
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def append_to_merged(symbol: str, new_df: pd.DataFrame):
    """
    将 new_df 合并到累计库：data/intraday_merged/{symbol}.parquet
    自动去重、排序
    """
    merged_path = MERGED_DIR / f"{symbol}.parquet"
    old_df = load_df(merged_path)

    if old_df is None or old_df.empty:
        merged = new_df.copy()
    else:
        # 合并去重
        merged = pd.concat([old_df, new_df])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()

    if not merged.empty:
        save_df(merged, merged_path)
        return merged.shape[0]
    return 0


def snapshot_save(symbol: str, df: pd.DataFrame):
    """
    保存当日原始快照：data/intraday_raw/{symbol}/{YYYYMMDD}.parquet
    """
    day_str = date.today().strftime("%Y%m%d")
    snap_dir = RAW_DIR / symbol
    snap_dir.mkdir(parents=True, exist_ok=True)
    snap_path = snap_dir / f"{day_str}.parquet"
    save_df(df, snap_path)


def run(symbols: list[str], interval: str, period: str, rth_only: bool, sleep_s: float):
    print(f"启动下载：symbols={len(symbols)}, interval={interval}, period={period}, RTH={rth_only}")
    for i, sym in enumerate(symbols, 1):
        try:
            print(f"[{i}/{len(symbols)}] ⬇️ {sym} ...")
            df = download_intraday(sym, interval=interval, period=period, rth_only=rth_only)
            if df.empty:
                print(f"⚠️ {sym}: 返回空数据，跳过")
            else:
                # 保存原始快照
                snapshot_save(sym, df)
                # 写入累计库
                total_rows = append_to_merged(sym, df)
                first_ts = df.index[0] if not df.empty else "NA"
                last_ts = df.index[-1] if not df.empty else "NA"
                print(f"✅ {sym}: 新增 {len(df)} 行，累计 {total_rows} 行，区间 {first_ts} → {last_ts}")
            time.sleep(sleep_s)
        except Exception as e:
            print(f"❌ {sym} 错误：{e}")


def build_argparser():
    p = argparse.ArgumentParser(description="yfinance 7天分钟线下载 & 本地累计")
    p.add_argument("--symbols", type=str, default=",".join(DEFAULT_SYMBOLS),
                   help="逗号分隔的股票代码列表，例如：AAPL,MSFT,SPY")
    p.add_argument("--interval", type=str, default=DEFAULT_INTERVAL,
                   help="分钟频率：1m/2m/5m/15m/30m/60m/90m（1m最大支持7d）")
    p.add_argument("--period", type=str, default=DEFAULT_PERIOD,
                   help="时间窗口（1m最大7d；>=2m 可到 60d）")
    p.add_argument("--rth", type=str, default=str(DEFAULT_RTH).lower(),
                   help="是否仅保留常规交易时段(09:30-16:00)，true/false")
    p.add_argument("--sleep", type=float, default=DEFAULT_SLEEP,
                   help="每个请求之间的休眠秒数（避免限流）")
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    interval = args.interval.strip()
    period = args.period.strip()
    rth_only = args.rth.strip().lower() in ("1", "true", "yes", "y")
    sleep_s = float(args.sleep)

    # 对 1m 做个简单保护
    if interval == "1m" and period != "7d":
        print("⚠️ 提示：yfinance 对 1m 的最大 period 为 7d，已自动改为 7d。")
        period = "7d"

    run(symbols, interval, period, rth_only, sleep_s)

