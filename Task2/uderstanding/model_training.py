
import joblib
import pandas as pd

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold

from features import *

# show all columns when printing tables, otherwise pandas hides the middle ones
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 140)


# Load the dataset

df = load_data()

print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:", df.isna().sum().sum())
print("Duplicate rows:", df.drop(columns=["session_id"]).duplicated().sum())

print("\nTarget distribution:")
print(df[TARGET].value_counts())

print("\nTarget distribution (%):")
print(df[TARGET].value_counts(normalize=True) * 100)


# Separate the features and the target

X, y = split_xy(df)

print("\nFeature shape:", X.shape)
print("Target shape:", y.shape)

# stratify keeps the same attack percentage in both splits
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=SEED,
    stratify=y
)

print("\nTraining features:", X_train.shape)
print("Validation features:", X_val.shape)
print("Training target:", y_train.shape)
print("Validation target:", y_val.shape)


# Baseline Random Forest (only one-hot encoding, nothing else)

baseline_fe = pipeline()

baseline_scores, baseline_model, baseline_cols = holdout_eval(
    baseline_fe,
    X_train,
    y_train,
    X_val,
    y_val
)

print("\nBaseline model training completed")
print("Baseline Accuracy:", baseline_scores["accuracy"])
print("Baseline F1-score:", baseline_scores["f1"])
print("Baseline ROC-AUC:", baseline_scores["roc_auc"])

# scores we need to match or beat
reference_scores = {
    "accuracy": 0.8831,
    "f1": 0.8508,
    "roc_auc": 0.8762
}

# probabilities of the baseline model, we need these for the confusion matrix later
X_train_base, X_val_base = baseline_fe(X_train, y_train, X_val)
baseline_proba = baseline_model.predict_proba(X_val_base)[:, 1]

print("\nBaseline Confusion Matrix:")
print(confusion_matrix(y_val, (baseline_proba >= 0.5).astype(int)))


# Test the 5 feature engineering methods one by one
# Each method is checked twice:
# 1. on the fixed validation split
# 2. with repeated 5-fold cross validation (more reliable because one split can be lucky)

methods = {
    "Baseline (one-hot)": pipeline(),
    "Frequency encoding": pipeline(frequency_encode),
    "OOF target encoding": pipeline(target_encode_oof),
    "Interaction features": pipeline(interactions),
    "Group aggregations": pipeline(group_aggregations),
    "Outlier winsorizing": pipeline(winsorize_outliers)
}

method_rows = []

for name, fe in methods.items():
    hold_scores, _, _ = holdout_eval(fe, X_train, y_train, X_val, y_val)
    cv_mean, cv_std = cv_eval(fe, X, y)

    method_rows.append({
        "method": name,
        "hold_accuracy": hold_scores["accuracy"],
        "hold_f1": hold_scores["f1"],
        "hold_roc_auc": hold_scores["roc_auc"],
        "cv_accuracy": cv_mean["accuracy"],
        "cv_f1": cv_mean["f1"],
        "cv_roc_auc": cv_mean["roc_auc"],
        "cv_roc_auc_std": cv_std["roc_auc"]
    })

    print("Finished testing:", name)

method_results = pd.DataFrame(method_rows).set_index("method")

# difference from the baseline for every score
for score in ["accuracy", "f1", "roc_auc"]:
    method_results["delta_hold_" + score] = method_results["hold_" + score] - method_results.loc["Baseline (one-hot)", "hold_" + score]
    method_results["delta_cv_" + score] = method_results["cv_" + score] - method_results.loc["Baseline (one-hot)", "cv_" + score]

print("\nValidation split results:")
print(method_results[["hold_accuracy", "hold_f1", "hold_roc_auc", "delta_hold_accuracy", "delta_hold_f1", "delta_hold_roc_auc"]].round(4))

print("\nCross validation results:")
print(method_results[["cv_accuracy", "cv_f1", "cv_roc_auc", "delta_cv_accuracy", "delta_cv_f1", "delta_cv_roc_auc", "cv_roc_auc_std"]].round(4))

method_results.to_csv("feature_method_results.csv")


# Combine the methods
# interactions was the most useful one so we try it together with each of the others

combos = {
    "interactions": pipeline(interactions),
    "interactions + OOF target encoding": pipeline(interactions, target_encode_oof),
    "interactions + winsorize": pipeline(interactions, winsorize_outliers),
    "interactions + frequency encoding": pipeline(interactions, frequency_encode),
    "interactions + aggregations": pipeline(interactions, group_aggregations)
}

combo_rows = []

for name, fe in combos.items():
    cv_mean, cv_std = cv_eval(fe, X, y)
    hold_scores, _, _ = holdout_eval(fe, X_train, y_train, X_val, y_val)

    combo_rows.append({
        "combo": name,
        "cv_accuracy": cv_mean["accuracy"],
        "cv_f1": cv_mean["f1"],
        "cv_roc_auc": cv_mean["roc_auc"],
        "hold_accuracy": hold_scores["accuracy"],
        "hold_f1": hold_scores["f1"],
        "hold_roc_auc": hold_scores["roc_auc"]
    })

    print("Finished testing:", name)

combo_results = pd.DataFrame(combo_rows).set_index("combo")

print("\nCombination results:")
print(combo_results.round(4))

# we pick the combination using the CV ROC-AUC, not the validation split
best_name = combo_results["cv_roc_auc"].idxmax()
best_fe = combos[best_name]

print("\nSelected combination:", best_name)

combo_results.to_csv("feature_combo_results.csv")


# Hyperparameter tuning
# the search only uses the training split, the validation split is not touched

X_train_best, X_val_best = best_fe(X_train, y_train, X_val)

param_space = {
    "n_estimators": [200, 300, 400],
    "max_depth": [None, 6, 8, 10, 12, 16],
    "min_samples_leaf": [1, 2, 3, 5, 8, 12],
    "min_samples_split": [2, 5, 10, 20],
    "max_features": ["sqrt", "log2", 0.3, 0.5, 0.7],
    "class_weight": [None, "balanced"],
    "criterion": ["gini", "entropy"]
}

search = RandomizedSearchCV(
    make_rf(),
    param_space,
    n_iter=20,
    scoring="roc_auc",
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED),
    random_state=SEED,
    n_jobs=1,
    refit=True
)

search.fit(X_train_best, y_train)

print("\nHyperparameter tuning completed")
print("Best parameters:", search.best_params_)
print("Best CV ROC-AUC:", search.best_score_)

tuned_model = search.best_estimator_


# Final evaluation on the validation split

default_model = make_rf()
default_model.fit(X_train_best, y_train)

default_proba = default_model.predict_proba(X_val_best)[:, 1]
tuned_proba = tuned_model.predict_proba(X_val_best)[:, 1]

final_scores = {
    "Baseline RF": baseline_scores,
    "Best FE + default RF": metrics(y_val, default_proba),
    "Best FE + tuned RF": metrics(y_val, tuned_proba)
}

final_results = pd.DataFrame(final_scores).T
final_results.loc["Reference target"] = reference_scores

# change in accuracy compared to the baseline
final_results["accuracy_change"] = final_results["accuracy"] - baseline_scores["accuracy"]

print("\nFinal results:")
print(final_results.round(4))

tuned_scores = final_scores["Best FE + tuned RF"]
beats_reference = all(tuned_scores[key] >= reference_scores[key] for key in reference_scores)

print("\nTuned model beats all three reference targets:", beats_reference)

tuned_pred = (tuned_proba >= 0.5).astype(int)

print("\nTuned Confusion Matrix:")
print(confusion_matrix(y_val, tuned_pred))

print("\nTuned Classification Report:")
print(classification_report(y_val, tuned_pred, digits=4))


# Top 10 most important features of the tuned model

importances = pd.Series(
    tuned_model.feature_importances_,
    index=X_train_best.columns
).sort_values(ascending=False)

print("\nTop 10 feature importances:")
print(importances.head(10))


# Save the results and the model

final_results.to_csv("final_results.csv")
importances.to_csv("feature_importances.csv")
joblib.dump(tuned_model, "tuned_random_forest.joblib")

print("\nResults and model saved")
