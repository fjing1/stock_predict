from pathlib import Path

def load_symbols(symbols_arg: str = "", symbols_file: str = "") -> list[str]:
    """Load symbols from comma-separated string or text file."""
    syms = []
    if symbols_arg:
        syms += [s.strip().upper() for s in symbols_arg.split(",") if s.strip()]
    if symbols_file:
        p = Path(symbols_file)
        if p.exists():
            syms += [line.strip().upper() for line in p.read_text().splitlines() if line.strip() and not line.startswith("#")]
    if not syms:
        syms = ["SPY","QQQ","IWM","AAPL","MSFT","NVDA","AMZN","META","GOOGL","TSLA"]
    return list(dict.fromkeys(syms))
