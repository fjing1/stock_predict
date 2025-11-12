# src/models.py
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, accuracy_score, r2_score

def train_models(train_df, feature_cols):
    """
    Train two models:
      - classifier: P(up in next 3 days)
      - regressor : expected 3-day return
    Returns: clf, rgr, metrics(dict)
    """
    X = train_df[feature_cols].astype(float)
    y_cls = train_df["y_up_3d"].astype(int)
    y_reg = train_df["fwd_ret_3d"].astype(float)

    clf = HistGradientBoostingClassifier(max_depth=6, learning_rate=0.05, max_iter=400, random_state=42)
    rgr = HistGradientBoostingRegressor(max_depth=6, learning_rate=0.05, max_iter=400, random_state=42)

    # simple time-series CV
    tscv = TimeSeriesSplit(n_splits=3)
    aucs, accs, r2s = [], [], []
    for tr, te in tscv.split(X):
        clf.fit(X.iloc[tr], y_cls.iloc[tr])
        rgr.fit(X.iloc[tr], y_reg.iloc[tr])

        proba = clf.predict_proba(X.iloc[te])[:, 1]
        aucs.append(roc_auc_score(y_cls.iloc[te], proba))
        accs.append(accuracy_score(y_cls.iloc[te], (proba >= 0.5).astype(int)))

        pred = rgr.predict(X.iloc[te])
        r2s.append(r2_score(y_reg.iloc[te], pred))

    # fit on all data
    clf.fit(X, y_cls)
    rgr.fit(X, y_reg)

    metrics = {
        "auc": float(np.mean(aucs)),
        "acc": float(np.mean(accs)),
        "r2":  float(np.mean(r2s)),
        "n":   int(len(X)),
    }
    return clf, rgr, metrics
