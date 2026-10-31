"""Shared helpers for Task-2: data loading, leakage-safe feature engineering, evaluation.

Every transformer follows the same contract: it is *fit on the training part only*
and then applied to the validation part, so no validation information leaks into
encodings, group statistics or clipping bounds.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

DATA_PATH = "cybersecurity_intrusion_data.csv"
CAT = ["protocol_type", "encryption_used", "browser_type"]
TARGET = "attack_detected"
SEED = 42


def load_data(path=DATA_PATH):
    # IMPORTANT: encryption_used contains the literal string "None" (= no encryption).
    # pandas' default parser silently turns it into NaN, which looks like 1,966
    # "missing values". keep_default_na=False keeps it as a genuine category.
    df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    return df


def split_xy(df):
    y = df[TARGET]
    X = df.drop(columns=[TARGET, "session_id"])
    return X, y


# --------------------------------------------------------------------------- #
# Feature-engineering methods. Signature: f(Xtr, ytr, Xva) -> (Xtr', Xva')
# --------------------------------------------------------------------------- #
def onehot(Xtr, ytr, Xva):
    a = pd.get_dummies(Xtr, columns=CAT, dtype=int)
    b = pd.get_dummies(Xva, columns=CAT, dtype=int).reindex(columns=a.columns, fill_value=0)
    return a, b


def frequency_encode(Xtr, ytr, Xva):
    """Replace each categorical by its relative frequency in the training set."""
    a, b = Xtr.copy(), Xva.copy()
    for c in CAT:
        f = a[c].value_counts(normalize=True)
        a[c + "_freq"] = a[c].map(f)
        b[c + "_freq"] = b[c].map(f).fillna(0.0)
    return a, b


def target_encode_oof(Xtr, ytr, Xva, k=5, smoothing=20, seed=0):
    """Out-of-fold, smoothed target encoding.

    Train rows are encoded with statistics from the *other* folds only, so a row
    never sees its own label. Validation rows use statistics from all of train.
    enc = (sum_y + m * prior) / (n + m)   (additive smoothing toward the global mean)
    """
    a, b = Xtr.copy(), Xva.copy()
    prior = ytr.mean()
    skf = StratifiedKFold(k, shuffle=True, random_state=seed)
    for c in CAT:
        oof = np.zeros(len(a))
        for i, j in skf.split(a, ytr):
            s = ytr.iloc[i].groupby(a[c].iloc[i]).agg(["sum", "count"])
            enc = (s["sum"] + smoothing * prior) / (s["count"] + smoothing)
            oof[j] = a[c].iloc[j].map(enc).fillna(prior).values
        a[c + "_te"] = oof
        s = ytr.groupby(a[c]).agg(["sum", "count"])
        enc = (s["sum"] + smoothing * prior) / (s["count"] + smoothing)
        b[c + "_te"] = b[c].map(enc).fillna(prior)
    return a, b


def interactions(Xtr, ytr, Xva):
    """Ratios, differences and products of the numeric behavioural signals."""
    def f(d):
        d = d.copy()
        d["fail_ratio"] = d.failed_logins / (d.login_attempts + 1)
        d["attempts_minus_fail"] = d.login_attempts - d.failed_logins
        d["dur_per_attempt"] = d.session_duration / (d.login_attempts + 1)
        d["pkt_per_sec"] = d.network_packet_size / (d.session_duration + 1)
        d["rep_x_fail"] = d.ip_reputation_score * d.failed_logins
        d["rep_x_attempts"] = d.ip_reputation_score * d.login_attempts
        d["unknown_browser"] = (d.browser_type == "Unknown").astype(int)
        d["log_duration"] = np.log1p(d.session_duration)
        return d
    return f(Xtr), f(Xva)


def group_aggregations(Xtr, ytr, Xva):
    """Group mean/median of numeric columns by each category, plus deviation from group mean."""
    a, b = Xtr.copy(), Xva.copy()
    nums = ["session_duration", "network_packet_size", "failed_logins", "ip_reputation_score"]
    for c in CAT:
        for col in nums:
            g = Xtr.groupby(c)[col].agg(["mean", "median"])
            for s in ["mean", "median"]:
                a[f"{c}_{col}_{s}"] = a[c].map(g[s])
                b[f"{c}_{col}_{s}"] = b[c].map(g[s])
            a[f"{col}_dev_{c}"] = a[col] - a[f"{c}_{col}_mean"]
            b[f"{col}_dev_{c}"] = b[col] - b[f"{c}_{col}_mean"]
    return a, b


def winsorize_outliers(Xtr, ytr, Xva, cols=("session_duration", "network_packet_size")):
    """Clip to Tukey fences (Q1-1.5 IQR, Q3+1.5 IQR) computed on train only."""
    a, b = Xtr.copy(), Xva.copy()
    for c in cols:
        q1, q3 = Xtr[c].quantile([0.25, 0.75])
        lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        a[c], b[c] = a[c].clip(lo, hi), b[c].clip(lo, hi)
    return a, b


def pipeline(*steps, final_encode=onehot):
    """Chain methods, then one-hot whatever categorical columns are still raw."""
    def run(Xtr, ytr, Xva):
        a, b = Xtr, Xva
        for s in steps:
            a, b = s(a, ytr, b)
        raw = [c for c in CAT if c in a.columns]
        if raw:
            a = pd.get_dummies(a, columns=raw, dtype=int)
            b = pd.get_dummies(b, columns=raw, dtype=int).reindex(columns=a.columns, fill_value=0)
        return a, b
    return run


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def metrics(y, proba, thr=0.5):
    pred = (proba >= thr).astype(int)
    return {"accuracy": accuracy_score(y, pred), "f1": f1_score(y, pred), "roc_auc": roc_auc_score(y, proba)}


def make_rf(**kw):
    p = dict(n_estimators=300, random_state=SEED, n_jobs=-1)
    p.update(kw)
    return RandomForestClassifier(**p)


def holdout_eval(fe, Xtr, ytr, Xva, yva, model_fn=make_rf):
    a, b = fe(Xtr, ytr, Xva)
    m = model_fn().fit(a, ytr)
    return metrics(yva, m.predict_proba(b)[:, 1]), m, a.columns


def cv_eval(fe, X, y, model_fn=make_rf, n_splits=5, n_repeats=2):
    """Repeated stratified CV with the FE fitted inside each training fold."""
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=SEED)
    rows = []
    for tr, va in rskf.split(X, y):
        Xtr, Xva, ytr, yva = X.iloc[tr], X.iloc[va], y.iloc[tr], y.iloc[va]
        a, b = fe(Xtr, ytr, Xva)
        m = model_fn().fit(a, ytr)
        rows.append(metrics(yva, m.predict_proba(b)[:, 1]))
    r = pd.DataFrame(rows)
    return r.mean().to_dict(), r.std().to_dict()
