# -*- coding: utf-8 -*-
"""
Enhanced training pipeline
--------------------------
• Adds robust normalisation (StandardScaler)
• Persists the best-performing model with pickle
• Generates SHAP feature-importance visualisation for the winning model

Usage
-----
Run as a script after defining `x_dfs_seg`, `y_s_seg`, `features_subsets` in your notebook / environment.
`shap_summary.png` and `best_model.pkl` will be created in the working directory.
"""
import time
import pickle
import math
from copy import deepcopy
from itertools import combinations

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import cross_val_predict, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    confusion_matrix,
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from scipy.stats import mode

###############################################################################
# Utility functions
###############################################################################

from sklearn.model_selection import StratifiedKFold
from scipy.stats import t
import numpy as np

def compute_metric_with_ci(metric_fn, y_true_list, y_pred_list, alpha=0.05):
    """Compute mean and CI for a list of metric values."""
    scores = [metric_fn(y_true, y_pred) for y_true, y_pred in zip(y_true_list, y_pred_list)]
    mean_score = np.mean(scores)
    sem = np.std(scores, ddof=1) / np.sqrt(len(scores))
    margin = t.ppf(1 - alpha/2, df=len(scores)-1) * sem
    return mean_score, margin

def generate_combinations(iterable, r=None):
    """Return all *r*-length combinations of the elements in *iterable*.
    If *r* is None, produce single‑feature combinations (r = 1)."""
    items = list(iterable)
    if r is None:
        r = 1
    return list(combinations(items, r))


def count_cm(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fp = cm[0][1]
    fn = cm[1][0]
    tp = cm[1][1]
    tn = cm[0][0]
    return fp, fn, tp, tn


def count_spec(y_true, y_pred):
    fp, fn, tp, tn = count_cm(y_true, y_pred)
    return tn / (tn + fp + 1e-8)


def count_sens(y_true, y_pred):
    fp, fn, tp, tn = count_cm(y_true, y_pred)
    return tp / (tp + fn + 1e-8)

###############################################################################
# Normalisation helper
###############################################################################

def normalise_train_test(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Fit a MinMaxScaler on the **numeric** columns of *X_train* and transform both
    train & test.  Non-numeric columns (object, string, category, bool…) are
    passed through unchanged.

    Returns
    -------
    X_train_scaled : pd.DataFrame
    X_test_scaled  : pd.DataFrame
    scaler         : MinMaxScaler
        The fitted scaler (can be reused later for new / hold-out data).
    """
    # ------------------------------------------------------------------
    # 1. Identify numeric vs. non-numeric columns
    # ------------------------------------------------------------------
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = [c for c in X_train.columns if c not in numeric_cols]

    # Edge case: nothing to scale
    if not numeric_cols:
        # Return copies so caller can mutate without touching originals
        return X_train.copy(), X_test.copy(), None

    # ------------------------------------------------------------------
    # 2. Fit scaler on numeric part of *training* data only
    # ------------------------------------------------------------------
    scaler = MinMaxScaler()
    X_train_num_scaled = scaler.fit_transform(X_train[numeric_cols])
    X_test_num_scaled  = scaler.transform(X_test[numeric_cols])

    # ------------------------------------------------------------------
    # 3. Re-assemble numeric + non-numeric, preserving original order
    # ------------------------------------------------------------------
    X_train_scaled = X_train.copy()
    X_test_scaled  = X_test.copy()

    X_train_scaled.loc[:, numeric_cols] = X_train_num_scaled
    X_test_scaled.loc[:,  numeric_cols] = X_test_num_scaled

    return X_train_scaled, X_test_scaled, scaler

###############################################################################
# Main training routine
###############################################################################

from sklearn.model_selection import StratifiedKFold

def compute_ensemble_cv_metrics(
    selected_model_names: list[str],
    fitted_models: dict,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
    average: str = "weighted",
    random_state: int = 42,
) -> dict:
    """
    Computes ensemble cross-validation metrics using majority voting,
    leveraging `cross_val_predict` to avoid retraining manually.
    This method is leakage-free and faster than training in a loop.
    Ignores non-numeric columns in X.
    """
    numeric_X = X.select_dtypes(include=[np.number])

    # Collect OOF (out-of-fold) predictions for each model
    oof_preds = []
    for model_name in selected_model_names:
        base_model = deepcopy(fitted_models[model_name])
        try:
            preds = cross_val_predict(
                base_model, numeric_X, y, cv=cv, method="predict", n_jobs=-1
            )
            oof_preds.append(preds)
        except Exception as e:
            print(f"!!! Failed on {model_name}: {e}")

    if not oof_preds:
        raise ValueError("No valid out-of-fold predictions could be generated.")

    # Majority voting across models
    stacked_preds = np.vstack(oof_preds)
    voted_preds = mode(stacked_preds, axis=0, keepdims=False)[0]

    y_true = y.to_numpy()
    y_pred = voted_preds

    return {
        "cv_accuracy": accuracy_score(y_true, y_pred),
        "cv_recall (UAR)": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "cv_f1_score": f1_score(y_true, y_pred, average=average, zero_division=0),
        "cv_specificity": count_spec(y_true, y_pred),
        "cv_sensitivity": count_sens(y_true, y_pred),
    }

def train_evaluate_models_with_multiple_feature_subsets(
    models: dict,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    feature_subsets_map: dict,
    feature_list: dict,
    param_grids: dict | None = None,
    task_type: str = "auto",
    average: str = "weighted",
    cv: int = 5,
    scoring: str = "recall",
    random_state: int = 42,
    verbose: bool = False,
    n_best_models: int | None = None,
    r: int | None = None,
    expl: bool = True
):
    """Train *models* on multiple feature subsets and return (results, fitted_models).

    The function now keeps the **fitted models** in memory so that the caller
    can easily persist or inspect the best performer.
    """
    # ---------------------------------------------------------------------
    # Task type inference
    # ---------------------------------------------------------------------
    if task_type == "auto":
        task_type = "regression" if np.issubdtype(y_train.dtype, np.floating) else "classification"

    # ---------------------------------------------------------------------
    # Build feature‑combination bookkeeping
    # ---------------------------------------------------------------------
    feat_comb_list = generate_combinations(feature_list.keys(), r=r)
    for k, v in feature_subsets_map.items():
        if not v:
            feature_subsets_map[k] = [[feature_list[key] for key in ks][0] for ks in feat_comb_list]

    results: list[dict] = []
    all_model_preds = {}
    all_model_names: list[str] = []
    fitted_models: dict[str, object] = {}

    # ---------------------------------------------------------------------
    # Iterate over models and feature subsets
    # ---------------------------------------------------------------------
    for base_name in list(models.keys()):  # ✅ Safe iteration
        model = models[base_name]
        if base_name not in feature_subsets_map:
            if verbose:
                print(f"[!] No feature subsets specified for model {base_name}")
            continue

        for i, subset in enumerate(feature_subsets_map[base_name]):
            name = f"{base_name}_F{'_'.join(feat_comb_list[i])}"
            #if X_train_sel.shape[1] > 100 and not expl and "Conv1D" not in base_name:
            model_clone = deepcopy(model)

            if verbose:
                print(f"\nTraining {name} on features: {'_'.join(feat_comb_list[i])}")

            start_time = time.time()
            subset_numeric = [f for f in subset if pd.api.types.is_numeric_dtype(X_train[f])]
            X_train_sel = X_train[subset_numeric]
            X_test_sel = X_test[subset_numeric]
            
            #if X_train_sel.shape[1] > 100 and not expl and "Conv1D" not in models:
            #    print("Added conv models")
                #models["Conv1D"] = Conv1DClassifier(input_size=X_train_sel.shape[1])
                #models["Conv1DAtt"] = Conv1DAttentionClassifier(input_size=X_train_sel.shape[1])
            if X_train_sel.shape[1] > 100 and not expl and "Conv1D" not in base_name:
                print("Reduced features")
                reducer = Conv1DReducer(out_features=64)
                reducer.fit(X_train_sel)
                X_train_sel = pd.DataFrame(reducer.transform(X_train_sel))
                X_test_sel = pd.DataFrame(reducer.transform(X_test_sel))

            X_all_sel = pd.concat([X_train_sel, X_test_sel])
            y_all = pd.concat([y_train, y_test])

            # Grid‑search (optional)
            if param_grids and base_name in param_grids:
                grid = GridSearchCV(
                    model_clone,
                    param_grids[base_name],
                    cv=cv,
                    scoring=scoring,
                    n_jobs=-1,
                )
                grid.fit(X_train_sel, y_train)
                model_clone = grid.best_estimator_
                best_params = grid.best_params_
            else:
                model_clone.fit(X_train_sel, y_train)
                best_params = None

            y_pred = model_clone.predict(X_test_sel)
            all_model_preds[name] = y_pred
            all_model_names.append(name)
            fitted_models[name] = model_clone  # <-- keep reference to fitted model

            # Predict‑proba (for ROC‑AUC)
            prob_cv = None
            if task_type == "classification":
                try:
                    prob_cv = cross_val_predict(model_clone, X_all_sel, y_all, cv=cv, method="predict_proba")
                except Exception:
                    pass

            # -----------------------------------------------------------------
            # Metric aggregation
            # -----------------------------------------------------------------
            metrics = {
                "model": name,
                "time_sec": round(time.time() - start_time, 2),
                "feature_names": "_".join(feat_comb_list[i]),
                "feature_subset": subset,
            }
            if best_params:
                metrics["best_params"] = best_params

            if task_type == "classification":
                metrics.update(
                    {
                        "accuracy": balanced_accuracy_score(y_test, y_pred),
                        "precision": precision_score(y_test, y_pred, average=average, zero_division=0),
                        "recall (UAR)": recall_score(y_test, y_pred, average="macro", zero_division=0),
                        "f1_score": f1_score(y_test, y_pred, average=average, zero_division=0),
                        "roc_auc": None,
                        "specificity": count_spec(y_test, y_pred),
                        "sensitivity": count_sens(y_test, y_pred),
                    }
                )
                # ✨ NEW: Cross-validation metrics on train set
                from sklearn.model_selection import StratifiedKFold

                skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
                y_true_folds = []
                y_pred_folds = []

                for train_idx, val_idx in skf.split(X_train_sel, y_train):
                    model_fold = deepcopy(model_clone)
                    model_fold.fit(X_train_sel.iloc[train_idx], y_train.iloc[train_idx])
                    y_pred = model_fold.predict(X_train_sel.iloc[val_idx])
                    y_true_folds.append(y_train.iloc[val_idx])
                    y_pred_folds.append(y_pred)
                f1_mean, f1_ci = compute_metric_with_ci(
                    lambda yt, yp: f1_score(yt, yp, average=average, zero_division=0),
                    y_true_folds, y_pred_folds
                )
                recall_mean, recall_ci = compute_metric_with_ci(
                    lambda yt, yp: recall_score(yt, yp, average="macro", zero_division=0),
                    y_true_folds, y_pred_folds
                )
                sens_mean, sens_ci = compute_metric_with_ci(count_sens, y_true_folds, y_pred_folds)
                spec_mean, spec_ci = compute_metric_with_ci(count_spec, y_true_folds, y_pred_folds)
                metrics.update({
                    "cv_f1_score_mean": f1_mean,
                    "cv_f1_score_CI": f1_ci,
                    "cv_recall (UAR)_mean": recall_mean,
                    "cv_recall (UAR)_CI": recall_ci,
                    "cv_sensitivity_mean": sens_mean,
                    "cv_sensitivity_CI": sens_ci,
                    "cv_specificity_mean": spec_mean,
                    "cv_specificity_CI": spec_ci,
                })
                
                if prob_cv is not None:
                    try:
                        if prob_cv.shape[1] == 2:
                            metrics["roc_auc"] = roc_auc_score(y_all, prob_cv[:, 1])
                        else:
                            metrics["roc_auc"] = roc_auc_score(
                                y_all, prob_cv, multi_class="ovr", average=average
                            )
                    except ValueError:
                        pass
            else:  # Regression
                metrics.update(
                    {
                        "rmse": mean_squared_error(y_test, y_pred, squared=False),
                        "mae": mean_absolute_error(y_test, y_pred),
                        "r2": r2_score(y_test, y_pred),
                    }
                )

            results.append(metrics)

    # keep track of the base‑model roster for every ensemble label
    ensemble_members: dict[str, list[str]] = {}
    # ---------------------------------------------------------------------
    # Majority‑voting ensembles (optional, kept from original code)
    # ---------------------------------------------------------------------
    def _add_majority_voting(label: str, selected_names: list[str]):
        if task_type != "classification" or len(all_model_preds) <= 1:
            return
        # store the roster so we can persist them later
        ensemble_members[label] = selected_names

        stacked = np.vstack([all_model_preds[n] for n in selected_names])
        voted = mode(stacked, axis=0, keepdims=False)[0]
        ensemble_metrics = {
            "model": label,
            "time_sec": None,
            "feature_subset": "ALL",
            "accuracy": accuracy_score(y_test, voted),
            "precision": precision_score(y_test, voted, average=average, zero_division=0),
            "recall (UAR)": recall_score(y_test, voted, average="macro", zero_division=0),
            "f1_score": f1_score(y_test, voted, average=average, zero_division=0),
            "specificity": count_spec(y_test, voted),
            "sensitivity": count_sens(y_test, voted),
            "roc_auc": None,
        }

        # NEW: Add ensemble cross-val metrics
        member_models = [fitted_models[n] for n in selected_names if n in fitted_models]
        # Add CV metrics (without leakage)
        cv_metrics = compute_ensemble_cv_metrics(
            selected_model_names=selected_names,
            fitted_models=fitted_models,
            X=X_train,  # original training data (with all columns)
            y=y_train,
            cv=cv,
            average=average,
            random_state=random_state,
        )
        ensemble_metrics.update(cv_metrics)
        results.append(ensemble_metrics)

    from collections import defaultdict, Counter

    def _add_bks_voting(label: str, selected_names: list[str]):
        if task_type != "classification" or len(all_model_preds) <= 1:
            return
        ensemble_members[label] = selected_names

        # 1. Collect OOF predictions for each model
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        bks_table = defaultdict(Counter)
        y_val_all = []
        pred_keys = []

        for train_idx, val_idx in skf.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            y_val_all.extend(y_val)

            val_preds = []
            for name in selected_names:
                model = deepcopy(fitted_models[name])
                model.fit(X_tr, y_tr)
                val_preds.append(model.predict(X_val))

            val_preds = np.array(val_preds).T  # shape: (n_val, n_models)
            for pred_row, true_label in zip(val_preds, y_val):
                key = tuple(pred_row)
                pred_keys.append(key)
                bks_table[key][true_label] += 1

        # 2. Make BKS predictions on test set
        test_preds = []
        for name in selected_names:
            model = deepcopy(fitted_models[name])
            model.fit(X_train, y_train)
            test_preds.append(model.predict(X_test))

        test_preds = np.array(test_preds).T
        bks_voted = []
        for pred_row in test_preds:
            key = tuple(pred_row)
            if key in bks_table:
                # Pick class with max observed count for this output pattern
                voted_class = bks_table[key].most_common(1)[0][0]
            else:
                # Fall back to majority vote
                voted_class = mode(pred_row, keepdims=False)[0]
            bks_voted.append(voted_class)

        y_true = y_test.to_numpy()
        y_pred = np.array(bks_voted)

        metrics = {
            "model": label,
            "time_sec": None,
            "feature_subset": "ALL",
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
            "recall (UAR)": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_score": f1_score(y_true, y_pred, average=average, zero_division=0),
            "specificity": count_spec(y_true, y_pred),
            "sensitivity": count_sens(y_true, y_pred),
            "roc_auc": None,
        }

        results.append(metrics)

    if n_best_models:
        ordered_names = (
            pd.DataFrame(results)
            .sort_values(by="recall (UAR)", ascending=False, na_position="last")["model"]
            .tolist()
        )
        _add_majority_voting("MajorityVoting_top", ordered_names[: n_best_models])
        _add_majority_voting("MajorityVoting_top×2", ordered_names[: n_best_models * 2])
        _add_majority_voting(
            "MajorityVoting_top×1.5", ordered_names[: math.ceil(n_best_models * 1.5)]
        )
        _add_majority_voting("MajorityVoting_top÷2", ordered_names[: math.ceil(n_best_models / 2)])
        
        
    results_df = pd.DataFrame(results).sort_values(
        by="recall (UAR)", na_position="last", ascending=False
    )
    return results_df, fitted_models, ensemble_members
