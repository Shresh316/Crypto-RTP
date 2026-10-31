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
    # encryption_used has the text "None" in it (it means no encryption)
    # pandas reads "None" as NaN by default, so it looks like 1966 missing values
    # keep_default_na=False stops this and keeps "None" as a normal category
    df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    return df


def split_xy(df):
    # y is the answer column, session_id is just an identifier so we drop it
    y = df[TARGET]
    X = df.drop(columns=[TARGET, "session_id"])
    return X, y


# Feature engineering methods
# Every method takes (X_train, y_train, X_val) and returns (X_train_new, X_val_new)

def onehot(X_train, y_train, X_val):
    X_train_encoded = pd.get_dummies(X_train, columns=CAT, dtype=int)
    X_val_encoded = pd.get_dummies(X_val, columns=CAT, dtype=int)

    # validation must have exactly the same columns as training
    X_val_encoded = X_val_encoded.reindex(
        columns=X_train_encoded.columns,
        fill_value=0
    )

    return X_train_encoded, X_val_encoded


def frequency_encode(X_train, y_train, X_val):
    # replace each category with how often it appears in the training data
    X_train_new = X_train.copy()
    X_val_new = X_val.copy()

    for col in CAT:
        freq = X_train_new[col].value_counts(normalize=True)

        X_train_new[col + "_freq"] = X_train_new[col].map(freq)

        # a category that was never seen in training gets 0
        X_val_new[col + "_freq"] = X_val_new[col].map(freq).fillna(0.0)

    return X_train_new, X_val_new


def target_encode_oof(X_train, y_train, X_val, k=5, smoothing=20, seed=0):
    # replace each category with the mean of the target for that category
    # "oof" means out-of-fold: a training row is encoded using the other folds only,
    # so it never sees its own label
    # smoothing pulls rare categories towards the overall mean:
    # encoded value = (sum of y + smoothing * prior) / (count + smoothing)
    X_train_new = X_train.copy()
    X_val_new = X_val.copy()

    prior = y_train.mean()
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

    for col in CAT:
        oof_values = np.zeros(len(X_train_new))

        for fit_idx, enc_idx in skf.split(X_train_new, y_train):
            # statistics come from the fit folds only
            stats = y_train.iloc[fit_idx].groupby(X_train_new[col].iloc[fit_idx]).agg(["sum", "count"])
            enc = (stats["sum"] + smoothing * prior) / (stats["count"] + smoothing)

            # rows in the held out fold are encoded with those statistics
            oof_values[enc_idx] = X_train_new[col].iloc[enc_idx].map(enc).fillna(prior).values

        X_train_new[col + "_te"] = oof_values

        # validation uses the statistics from the whole training set
        stats = y_train.groupby(X_train_new[col]).agg(["sum", "count"])
        enc = (stats["sum"] + smoothing * prior) / (stats["count"] + smoothing)
        X_val_new[col + "_te"] = X_val_new[col].map(enc).fillna(prior)

    return X_train_new, X_val_new


def add_interaction_columns(df):
    df = df.copy()

    # 1 is added in the denominators so we never divide by zero
    df["fail_ratio"] = df["failed_logins"] / (df["login_attempts"] + 1)
    df["attempts_minus_fail"] = df["login_attempts"] - df["failed_logins"]
    df["dur_per_attempt"] = df["session_duration"] / (df["login_attempts"] + 1)
    df["pkt_per_sec"] = df["network_packet_size"] / (df["session_duration"] + 1)

    df["rep_x_fail"] = df["ip_reputation_score"] * df["failed_logins"]
    df["rep_x_attempts"] = df["ip_reputation_score"] * df["login_attempts"]

    df["unknown_browser"] = (df["browser_type"] == "Unknown").astype(int)
    df["log_duration"] = np.log1p(df["session_duration"])

    return df


def interactions(X_train, y_train, X_val):
    # ratios, differences and products of the numeric columns
    # nothing is learned from the data here so train and validation are treated the same
    return add_interaction_columns(X_train), add_interaction_columns(X_val)


def group_aggregations(X_train, y_train, X_val):
    # for every category, take the mean and median of some numeric columns
    # and also how far each row is from its group mean
    # the group values are calculated from the training data only
    X_train_new = X_train.copy()
    X_val_new = X_val.copy()

    numeric_cols = ["session_duration", "network_packet_size", "failed_logins", "ip_reputation_score"]

    for cat_col in CAT:
        for num_col in numeric_cols:
            group_stats = X_train.groupby(cat_col)[num_col].agg(["mean", "median"])

            for stat in ["mean", "median"]:
                new_name = f"{cat_col}_{num_col}_{stat}"
                X_train_new[new_name] = X_train_new[cat_col].map(group_stats[stat])
                X_val_new[new_name] = X_val_new[cat_col].map(group_stats[stat])

            # deviation from the group mean
            mean_name = f"{cat_col}_{num_col}_mean"
            X_train_new[f"{num_col}_dev_{cat_col}"] = X_train_new[num_col] - X_train_new[mean_name]
            X_val_new[f"{num_col}_dev_{cat_col}"] = X_val_new[num_col] - X_val_new[mean_name]

    return X_train_new, X_val_new


def winsorize_outliers(X_train, y_train, X_val, cols=("session_duration", "network_packet_size")):
    # clip extreme values to the IQR limits (Q1 - 1.5 * IQR and Q3 + 1.5 * IQR)
    # the limits come from the training data only
    X_train_new = X_train.copy()
    X_val_new = X_val.copy()

    for col in cols:
        q1 = X_train[col].quantile(0.25)
        q3 = X_train[col].quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        X_train_new[col] = X_train_new[col].clip(lower, upper)
        X_val_new[col] = X_val_new[col].clip(lower, upper)

    return X_train_new, X_val_new


def pipeline(*steps):
    # runs the given methods one after another
    # at the end any category column that is still raw gets one-hot encoded
    def run(X_train, y_train, X_val):
        X_train_new, X_val_new = X_train, X_val

        for step in steps:
            X_train_new, X_val_new = step(X_train_new, y_train, X_val_new)

        raw_cols = [col for col in CAT if col in X_train_new.columns]

        if raw_cols:
            X_train_new = pd.get_dummies(X_train_new, columns=raw_cols, dtype=int)
            X_val_new = pd.get_dummies(X_val_new, columns=raw_cols, dtype=int)
            X_val_new = X_val_new.reindex(
                columns=X_train_new.columns,
                fill_value=0
            )

        return X_train_new, X_val_new

    return run


# Evaluation

def metrics(y_true, proba, threshold=0.5):
    y_pred = (proba >= threshold).astype(int)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, proba)
    }


def make_rf(**kwargs):
    params = dict(n_estimators=300, random_state=SEED, n_jobs=-1)

    # anything passed in replaces the default value
    params.update(kwargs)

    return RandomForestClassifier(**params)


def holdout_eval(fe, X_train, y_train, X_val, y_val, model_fn=make_rf):
    # one train/validation split: build features, train the model, score it
    X_train_new, X_val_new = fe(X_train, y_train, X_val)

    model = model_fn()
    model.fit(X_train_new, y_train)

    proba = model.predict_proba(X_val_new)[:, 1]
    scores = metrics(y_val, proba)

    return scores, model, X_train_new.columns


def cv_eval(fe, X, y, model_fn=make_rf, n_splits=5, n_repeats=2):
    # repeated stratified cross validation
    # the feature engineering is fit again inside every training fold
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=SEED
    )

    fold_scores = []

    for train_idx, val_idx in rskf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        X_train_new, X_val_new = fe(X_train, y_train, X_val)

        model = model_fn()
        model.fit(X_train_new, y_train)

        proba = model.predict_proba(X_val_new)[:, 1]
        fold_scores.append(metrics(y_val, proba))

    fold_scores = pd.DataFrame(fold_scores)

    return fold_scores.mean().to_dict(), fold_scores.std().to_dict()
