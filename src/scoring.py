import numpy as np
import pandas as pd
from .data_daily import dl_daily, add_daily_indicators
from .data_intraday import dl_intraday_7d, intraday_enhancers

def score_today(symbols, feature_cols, clf, rgr, rth_only=True):
    rows = []
    for s in symbols:
        d = dl_daily(s, "6mo")
        if d.empty: continue
        d = add_daily_indicators(d).dropna()
        if d.empty: continue
        x = d[feature_cols].iloc[-1:].astype(float)
        p_up = float(clf.predict_proba(x)[:,1][0])
        exp_ret = float(rgr.predict(x)[0])

        enh = intraday_enhancers(dl_intraday_7d(s, rth_only=rth_only))
        mom = enh.get("intraday_mom", 0)
        vwap = enh.get("vwap_dist_last", 0)
        drift = enh.get("drift_7d", 0)

        score = 100*p_up + 300*exp_ret + 20*mom - 10*abs(vwap) + 5*drift
        close = float(d["Close"].iloc[-1])
        rows.append(dict(symbol=s, p_up_3d=p_up, exp_ret_3d=exp_ret, score=score, close=close))
    return pd.DataFrame(rows).sort_values("score", ascending=False)
