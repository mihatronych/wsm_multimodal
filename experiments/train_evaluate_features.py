import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from typing import Tuple, Dict
from .training_pipeline import *
from .load_and_merge_feature_csvs import *

def get_feature_subset_by_task(task, features_hc, features_all):
    """
    Returns the appropriate feature dictionary based on task string.
    """
    # Determine if only explainable (hc = human-comprehensible) features should be used
    explainable = "expl" in task

    if explainable:
        feats = features_hc
    else:
        feats = {
            "hc_aud": features_hc["hc_aud"],
            "hc_txt": features_hc["hc_txt"],
            "hc_vid": features_hc["hc_vid"],
            "mfcc": features_all["mfcc"],
            "emo": features_hc["emo"],
            "pose": features_hc["pose_visible"],
            "ocean": features_hc["ocean"],
            "nn": features_all["nn"],

        }

    # Modality selector
    mod = task.split("_")[-1]  # e.g., 'at', 'v', 'avt'

    if mod in ["at", "ta"]:
        if explainable:
            return { "hc_txt+aud": feats["hc_txt+aud"] 
                "hc_txt": feats["hc_txt"],
                "hc_aud": feats["hc_aud"],
                "hc_txt+aud": feats["hc_txt+aud"],
                }
        else:
            return {
                "hc_aud":features_subsets_hc["hc_aud"], 
                "hc_txt":features_subsets_hc["hc_txt"], 
                "hc_txt+aud":features_subsets_hc["hc_txt"] + features_subsets_hc["hc_aud"], 
                "mfcc":features_subsets["mfcc"], 
            }
    elif mod == "a":
        if explainable:
            return { 
                "hc_aud": feats["hc_aud"]
                }
        else:
            return {
                "hc_aud":features_subsets_hc["hc_aud"], 
                "mfcc":features_subsets["mfcc"], 
            }
    elif mod == "t":
        return { "hc_txt": feats["hc_txt"] }
    elif mod == "v":
        if explainable:
            return { 
                "hc_vid": feats["hc_vid"],
                "emo": features_hc["emo"],
                "pose": features_hc["pose_visible"],
                "ocean": features_hc["ocean"],
                }
        else:
            return {
                "hc_vid": feats["hc_vid"],
                "emo": features_hc["emo"],
                "pose": features_hc["pose_visible"],
                "ocean": features_hc["ocean"],
                "nn":features_subsets["nn"], 
            }
    elif mod in ["avt", "tav", "vta"]:
        if explainable:
            return {
                "hc_txt": feats["hc_txt"],
                "hc_aud": feats["hc_aud"],
                "hc_txt+aud": feats["hc_txt+aud"],
                "hc_vid": feats["hc_vid"],
                "emo": features_hc["emo"],
                "pose": features_hc["pose_visible"],
                "ocean": features_hc["ocean"],
                "all_expl": feats["hc_txt"]+feats["hc_aud"]+feats["hc_vid"]+features_hc["emo"]+features_hc["pose_visible"]+features_hc["ocean"]
            }
        else:
            return {
                "hc_txt": feats["hc_txt"],
                "hc_aud": feats["hc_aud"],
                "hc_txt+aud": feats["hc_txt+aud"],
                "hc_vid": feats["hc_vid"],
                "emo": features_hc["emo"],
                "pose": features_hc["pose_visible"],
                "ocean": features_hc["ocean"],
                "mfcc":features_subsets["mfcc"], 
                "nn":features_subsets["nn"], 
                "all_expl": feats["hc_txt"]+feats["hc_aud"]+feats["hc_vid"]+features_hc["emo"]+features_hc["pose_visible"]+features_hc["ocean"],
                "all": feats["hc_txt"]+feats["hc_aud"]+feats["hc_vid"]+features_hc["emo"]+features_hc["pose_visible"]+features_hc["ocean"] +features_subsets["nn"]+features_subsets["mfcc"]
            }
    else:
        raise ValueError(f"Unknown feature modality specifier: {mod}")

def get_data_by_task(task: str, x_dfs_seg: dict, y_s_seg: dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    """Selects appropriate training and test sets based on task keyword."""
    if "parkinson" in task:
        X_train = x_dfs_seg["park_train"].copy()
        y_train = y_s_seg["park_train"].copy()
        X_test = x_dfs_seg["park_test"].copy()
        y_test = y_s_seg["park_test"].copy()
    elif "depression" in task:
        X_train = x_dfs_seg["depr_train"].copy()
        y_train = y_s_seg["depr_train"].copy()
        X_test = x_dfs_seg["depr_test"].copy()
        y_test = y_s_seg["depr_test"].copy()
    else:  # both
        X_train = pd.concat([x_dfs_seg["park_train"], x_dfs_seg["depr_train"]], ignore_index=True)
        y_train = pd.concat([y_s_seg["park_train"], y_s_seg["depr_train"]], ignore_index=True)
        X_test = pd.concat([x_dfs_seg["park_test"], x_dfs_seg["depr_test"]], ignore_index=True)
        y_test = pd.concat([y_s_seg["park_test"], y_s_seg["depr_test"]], ignore_index=True)
    return X_train, X_test, y_train, y_test


def get_default_models() -> Dict[str, object]:
    return {
        "RF": RandomForestClassifier(random_state=42),
        "LogReg": LogisticRegression(max_iter=1000, random_state=42),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "Conv1D":Conv1DClassifier(),
    }


def get_default_param_grids() -> Dict[str, dict]:
    return {
        "RF": {
            "n_estimators": [50, 100],
            "max_features": ["sqrt", "log2"],
        },
        "LogReg": {"C": np.linspace(0.1, 1.0, 5)},
        "DecisionTree": {
            "max_depth": [5, 10, 20],
            "max_features": [30, "sqrt", "log2"],
        },
        "SVM": {
            "C": np.linspace(0.1, 1.0, 5),
            "kernel": ["linear", "rbf"],
        },
    }


def save_best_model_or_ensemble(task, df_results, trained_models, ensemble_members, fitted_scaler, out_dir="models"):
    os.makedirs(out_dir, exist_ok=True)
    out_dir = os.path.join(out_dir, task)
    os.makedirs(out_dir, exist_ok=True)

    best_row = df_results.iloc[0]
    best_model_name = best_row["model"]

    if best_model_name in ensemble_members:
        roster = ensemble_members[best_model_name]
        for name in roster:
            mdl = trained_models[name]
            feats = df_results[df_results.model == name]["feature_subset"].values[0]
            fname = os.path.join(out_dir, f"{task}_{name}.pkl")
            with open(fname, "wb") as f:
                pickle.dump({
                    "model": mdl,
                    "scaler": fitted_scaler,
                    "features": feats,
                    "task": task
                }, f)
        with open(os.path.join(out_dir, f"{task}_ensemble_roster.pkl"), "wb") as f:
            pickle.dump({"members": roster}, f)
    else:
        best_model = trained_models[best_model_name]
        feats = best_row["feature_subset"]
        fname = os.path.join(out_dir, f"{task}_best_model.pkl")
        with open(fname, "wb") as f:
            pickle.dump({
                "model": best_model,
                "scaler": fitted_scaler,
                "features": feats,
                "task": task
            }, f)

if __name__ == "__main__":
    # Define where your data is
    data_dirs = {
        "depression": "depression/",
        "parkinson": "parkinson/"
    }

    # Load datasets
    x_dfs_seg = load_datasets(data_dirs)

    # Extract labels
    y_s_seg = extract_labels(x_dfs_seg)

    # Clean and split features
    x_dfs_seg_clean, features_subsets_seg, features_subsets_seg_hc = split_features(x_dfs_seg, keep_ids=False)

    
    # example
    task = "parkinson_expl_at"

    # Step 1: Get data
    X_train, X_test, y_train, y_test = get_data_by_task(task, x_dfs_seg, y_s_seg)
    print(f"Task: {task} — Train: {len(X_train)} | Test: {len(X_test)}")

    # Step 2: Normalize
    X_train_norm, X_test_norm, fitted_scaler = normalise_train_test(X_train, X_test)

    # Step 3: Models and grids
    models = get_default_models()
    param_grids = get_default_param_grids()

    # Step 4: Get feature subset
    feature_list = get_feature_subset_by_task(
        task=task,
        features_hc=features_subsets_seg_hc,
        features_all=features_subsets_seg
    )
    feature_subsets_map = {model_name: [] for model_name in models.keys()}

    # Step 5: Train and evaluate
    df_results, trained_models, ensemble_members = train_evaluate_models_with_multiple_feature_subsets(
        models=models,
        X_train=X_train_norm,
        X_test=X_test_norm,
        y_train=y_train,
        y_test=y_test,
        feature_subsets_map=feature_subsets_map,
        feature_list=feature_list,
        param_grids=param_grids,
        task_type="classification",
        verbose=True,
        n_best_models=5,
        r=1,
        expl="expl" in task
    )

    print("\nTop Results:\n", df_results.head(5))

    # Step 6: Save model(s)
    save_best_model_or_ensemble(
        task=task,
        df_results=df_results,
        trained_models=trained_models,
        ensemble_members=ensemble_members,
        fitted_scaler=fitted_scaler,
        out_dir="models"
    )