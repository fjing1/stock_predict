# -*- coding: utf-8 -*-
"""
一体化分钟数据更新器（yfinance, 7d窗口）
功能：
- 下载指定symbols的最近 N 天分钟数据（默认1m/7d）
- 保存当日“原始快照”：data/intraday_raw/{SYMBOL}/{YYYYMMDD}.parquet
- 合并入“累计历史库”：data/intraday_merged/{SYMBOL}.parquet（按时间去重、排序）
- 支持仅保留常规交易时段(09:30-16:00 America/New_York)
- 可选清理早于N天的raw快照
- 输出日志到 logs/update_YYYYMMDD.log

用法示例：
    python update_intraday_allinone.py --symbols AAPL,MSFT,SPY --rth true
    python update_intraday_allinone.py --symbols-file symbols.txt --interval 1m --period 7d
    python update_intraday_allinone.py --interval 5m --period 60d --raw-keep-days 30
"""

import argparse
from datetime import date, datetime, timedelta
from pathlib import Path
import sys
import time
import traceback

import pandas as pd
import yfinance as yf

# ---------------- 目录与默认参数 ----------------
RAW_DIR = Path("data/intraday_raw")
MERGED_DIR = Path("data/intraday_merged")
LOG_DIR = Path("logs")

DEFAULT_INTERVAL = "1m"     # 1m | 2m | 5m | 15m | 30m | 60m | 90m
DEFAULT_PERIOD = "7d"       # 1m 最大7d；>=2m 可到60d
DEFAULT_RTH = False         # 仅保留常规交易时段(09:30-16:00 America/New_York)
DEFAULT_SLEEP = 0.3         # 每只之间的间隔，避免限流
DEFAULT_RETRY = 2           # 下载失败重试次数
DEFAULT_RAW_KEEP_DAYS = 0   # >0时清理早于N天的原始快照

for d in [RAW_DIR, MERGED_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ---------------- 小工具 ----------------
def log(msg: str, fp=None):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    if fp:
        fp.write(line + "\n")
        fp.flush()

def save_df(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path)
    except Exception:
        df.to_csv(path.with_suffix(".csv"))
        # 不打印也行，这里提醒一下
        print(f"⚠️ 未安装 pyarrow 或写 parquet 失败，已回退 CSV: {path.with_suffix('.csv').name}")

def load_df(path: Path):
    if not path.exists():
        # 尝试csv
        csv = path.with_suffix(".csv")
        if csv.exists():
            try:
                return pd.read_csv(csv, index_col=0, parse_dates=True)
            except Exception:
                return None
        return None
    # 先尝试parquet
    try:
        return pd.read_parquet(path)
    except Exception:
        # 回退csv
        csv = path.with_suffix(".csv")
        if csv.exists():
            try:
                return pd.read_csv(csv, index_col=0, parse_dates=True)
            except Exception:
                return None
        return None

def normalize_tz(df: pd.DataFrame, rth_only: bool) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    try:
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC").tz_convert("America/New_York")
        else:
            df.index = df.index.tz_convert("America/New_York")
    except Exception:
        pass
    # 去重 + 排序
    df = df[~df.index.duplicated(keep="last")].sort_index()
    # 仅RTH
    if rth_only:
        try:
            df = df.between_time("09:30", "16:00")
        except Exception:
            pass
    return df

def download_intraday(symbol: str, interval: str, period: str, rth_only: bool,
                      retries: int, sleep_s: float, log_fp=None) -> pd.DataFrame:
    # 保护：1m时强制period=7d
    if interval == "1m" and period != "7d":
        log(f"⚠️ {symbol}: 1m 仅支持 7d，自动改为 7d", log_fp)
        period = "7d"

    last_err = None
    for attempt in range(1, retries + 2):
        try:
            log(f"⬇️ 下载 {symbol} ({interval},{period}) 尝试 {attempt}", log_fp)
            df = yf.download(symbol, period=period, interval=interval,
                             progress=False, auto_adjust=False)
            if df is None or df.empty:
                raise RuntimeError("Empty dataframe")
            df = normalize_tz(df, rth_only)
            return df
        except Exception as e:
            last_err = e
            log(f"❌ {symbol} 下载失败: {e}", log_fp)
            if attempt <= retries:
                time.sleep(min(2.0, sleep_s + 0.5))
            else:
                break
    log(f"⚠️ {symbol}: 多次尝试仍失败，跳过。最后错误：{last_err}", log_fp)
    return pd.DataFrame()

def snapshot_save(symbol: str, df: pd.DataFrame, log_fp=None):
    if df is None or df.empty:
        return None
    day_str = date.today().strftime("%Y%m%d")
    snap_dir = RAW_DIR / symbol
    snap_dir.mkdir(parents=True, exist_ok=True)
    snap_path = snap_dir / f"{day_str}.parquet"
    save_df(df, snap_path)
    log(f"📝 保存原始快照: {snap_path}", log_fp)
    return snap_path

def append_to_merged(symbol: str, new_df: pd.DataFrame, log_fp=None):
    merged_path = MERGED_DIR / f"{symbol}.parquet"
    old_df = load_df(merged_path)

    before_rows = 0 if old_df is None or old_df.empty else old_df.shape[0]
    if old_df is None or old_df.empty:
        merged = new_df.copy()
    else:
        merged = pd.concat([old_df, new_df])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()

    after_rows = 0 if merged is None or merged.empty else merged.shape[0]
    if after_rows > 0:
        save_df(merged, merged_path)
    log(f"📦 合并 {symbol}: 新增 {max(0, after_rows - before_rows)} 行，累计 {after_rows} 行", log_fp)
    return (max(0, after_rows - before_rows), after_rows)

def clean_old_raw(symbols, keep_days: int, log_fp=None):
    if keep_days <= 0:
        return
    cutoff = date.today() - timedelta(days=keep_days)
    removed = 0
    for sym in symbols:
        d = RAW_DIR / sym
        if not d.exists():
            continue
        for p in d.glob("*.parquet"):
            try:
                day = p.stem  # YYYYMMDD
                dt = datetime.strptime(day, "%Y%m%d").date()
                if dt < cutoff:
                    p.unlink(missing_ok=True)
                    # 同名csv也删一下
                    csv = p.with_suffix(".csv")
                    if csv.exists():
                        csv.unlink(missing_ok=True)
                    removed += 1
            except Exception:
                continue
    log(f"🧹 清理原始快照完成：删除 {removed} 个旧文件（早于 {keep_days} 天）", log_fp)

def load_symbols(args):
    symbols = []
    if args.symbols:
        symbols += [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if args.symbols_file:
        txt = Path(args.symbols_file)
        if txt.exists():
            symbols += [line.strip().upper() for line in txt.read_text().splitlines() if line.strip()]
    # 去重保序
    return list(dict.fromkeys(symbols))

def build_argparser():
    p = argparse.ArgumentParser(description="一体化分钟数据更新器（yfinance, 7d窗口，合并累计库）")
    p.add_argument("--symbols", type=str, default="",
                   help="逗号分隔股票代码，如：AAPL,MSFT,SPY")
    p.add_argument("--symbols-file", type=str, default="",
                   help="从文本文件读取代码（每行一个）")
    p.add_argument("--interval", type=str, default=DEFAULT_INTERVAL,
                   help="1m/2m/5m/15m/30m/60m/90m。1m 最大仅支持 7d")
    p.add_argument("--period", type=str, default=DEFAULT_PERIOD,
                   help="时间窗口。1m=7d；>=2m 可到 60d")
    p.add_argument("--rth", type=str, default=str(DEFAULT_RTH).lower(),
                   help="仅保留常规交易时段（09:30-16:00），true/false")
    p.add_argument("--sleep", type=float, default=DEFAULT_SLEEP,
                   help="两只股票之间休眠秒数")
    p.add_argument("--retries", type=int, default=DEFAULT_RETRY,
                   help="下载失败重试次数")
    p.add_argument("--raw-keep-days", type=int, default=DEFAULT_RAW_KEEP_DAYS,
                   help="清理早于N天的原始快照（0表示不清理）")
    return p


def main():
    args = build_argparser().parse_args()
    symbols = load_symbols(args)
    if not symbols:
        print("❗请使用 --symbols 或 --symbols-file 指定股票列表")
        sys.exit(1)

    interval = args.interval.strip()
    period = args.period.strip()
    rth_only = args.rth.strip().lower() in ("1","true","yes","y")
    sleep_s = float(args.sleep)
    retries = int(args.retries)
    raw_keep_days = int(args.raw_keep_days)

    # 日志文件
    log_path = LOG_DIR / f"update_{date.today().strftime('%Y%m%d')}.log"
    with open(log_path, "a", encoding="utf-8") as log_fp:
        log(f"=== 启动更新：symbols={len(symbols)}, interval={interval}, period={period}, RTH={rth_only} ===", log_fp)

        total_success = 0
        total_added_rows = 0
        t0 = time.time()

        for i, sym in enumerate(symbols, 1):
            try:
                log(f"[{i}/{len(symbols)}] 处理 {sym}", log_fp)
                df = download_intraday(sym, interval, period, rth_only, retries, sleep_s, log_fp)
                if df.empty:
                    log(f"⚠️ {sym}: 无数据或下载失败，跳过合并", log_fp)
                else:
                    snapshot_save(sym, df, log_fp)
                    added, total = append_to_merged(sym, df, log_fp)
                    total_added_rows += added
                    total_success += 1
                time.sleep(sleep_s)
            except Exception as e:
                log(f"❌ {sym}: 未处理异常: {e}", log_fp)
                traceback.print_exc(file=log_fp)
                time.sleep(min(2.0, sleep_s + 0.5))

        if raw_keep_days > 0:
            clean_old_raw(symbols, raw_keep_days, log_fp)

        dt = time.time() - t0
        log(f"✅ 完成：成功 {total_success}/{len(symbols)}，新增记录 {total_added_rows} 行，用时 {dt:.1f}s", log_fp)
        log(f"日志位置：{log_path}", log_fp)


if __name__ == "__main__":
    main()
