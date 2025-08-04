import os
import pickle
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, f1_score, recall_score,
    balanced_accuracy_score, confusion_matrix
)

def run_full_evaluation(task="depression", modality="avt", use_ensemble=True, id_col="video_id", if_shap=True):
    if if_shap:
        import shap

    # Mapping for human-readable visual features
    hc_mapping = {
        "hc_vid_0_": "Left eye center X",
        "hc_vid_1_": "Left eye center Y",
        "hc_vid_2_": "Right eye center X",
        "hc_vid_3_": "Right eye center Y",
        "hc_vid_4_": "Inter-eye distance",
        "hc_vid_5_": "Head tilt angle (degrees)",
        "hc_vid_6_": "Distance: Left eye center to left corner",
        "hc_vid_7_": "Distance: Left eye center to right corner",
        "hc_vid_8_": "Distance: Right eye center to left corner",
        "hc_vid_9_": "Distance: Right eye center to right corner",
        "hc_vid_10_": "Mouth corner tilt angle",
        "hc_vid_11_": "Eyebrow tilt angle",
    }
    coords_face_mesh_fi = [
        0, 1, 386, 133, 6, 8, 267, 13, 14, 17, 145, 276,
        152, 282, 411, 285, 159, 291, 37, 299, 46, 52, 55,
        187, 61, 69, 331, 334, 336, 102, 105, 362, 107,
        374, 33, 263
    ]

    coords_face_mesh_labels = {
        0:   "face center (between eyes, top of nose bridge)",
        1:   "nose tip",
        6:   "nose bridge (mid-point)",
        8:   "tip of chin",
        13:  "upper lip center",
        14:  "lower lip center",
        17:  "chin (below lower lip)",
        33:  "left eye outer corner",
        37:  "left eye upper lid (middle)",
        46:  "left eyebrow tail",
        52:  "left eyebrow middle",
        55:  "left eyebrow head (inner part)",
        61:  "left mouth corner",
        69:  "lower lip (left side)",
        102: "right eyebrow head (inner part)",
        105: "right eyebrow middle",
        107: "right eyebrow tail",
        133: "left eye inner corner",
        145: "left eye lower lid (middle)",
        152: "chin center (bottom-most point)",
        159: "left eye lower lid (inner)",
        187: "upper lip (left peak)",
        263: "right eye outer corner",
        267: "nose ridge (just above right nostril)",
        274: "right eye lower lid (outer)",
        276: "left eye outer corner (low)",
        282: "lower right cheek",
        285: "right cheek (near corner of mouth)",
        291: "left eye lower lid (outer)",
        299: "upper lip (right peak)",
        331: "right mouth corner",
        334: "lower lip (right side)",
        336: "chin (right side, under lip)",
        362: "right eye inner corner",
        374: "right eye lower lid (middle)",
        386: "right eye upper lid (middle)",
        411: "lower right cheek (near jaw)"
    }

    for i, point in enumerate(coords_face_mesh_fi):
        hc_mapping[f"hc_vid_{12 + i * 2}_"] = f"Face mesh point {coords_face_mesh_labels[point]} X"
        hc_mapping[f"hc_vid_{13 + i * 2}_"] = f"Face mesh point {coords_face_mesh_labels[point]} Y"
    couples_face_mesh_mupta = [
        [133, 46], [133, 52], [133, 55], [362, 285], [362, 282], [362, 276],
        [55, 285], [1, 6], [8, 6], [0, 1], [0, 17], [61, 291],
        [0, 13], [61, 291], [37, 13], [267, 13], [13, 14], [17, 152],
        [102, 331], [102, 133], [331, 362], [291, 362], [61, 133], [386, 274],
        [159, 145], [69, 105], [69, 145], [299, 336], [299, 334], [187, 133],
        [411, 362]
    ]

    for i, (p1, p2) in enumerate(couples_face_mesh_mupta):
        hc_mapping[f"hc_vid_{84 + i}_"] = f"Distance {coords_face_mesh_labels[p1]} <-> {coords_face_mesh_labels[p2]}"

    t = f"{task}_expl_{modality}"

    def clean_feat_name(name):
        name = name.replace("_", " ")
        if name.startswith("hc txt"):
            return name.replace("hc txt", "BlaBla")
        elif name.startswith("hc aud"):
            return name.replace("hc aud", "Audio")
        elif modality in ("v", "avt"):
            for k, v in hc_mapping.items():
                if name.replace(" ", "_").startswith(k):
                    return "Facial: " + name.replace(k.replace("_", " "),v+" ")
        return name

    def load_artifacts(task):
        base = Path(f"models_{t}")
        single = base / f"{t}_best_model.pkl"
        roster = base / f"{t}_ensemble_roster.pkl"

        def _any_pkl(p): return next(p.glob("*.pkl"))

        if use_ensemble:
            with open(roster if roster.exists() else _any_pkl(base), "rb") as f:
                tmp = pickle.load(f)
                first = tmp.get("members", [None])[0]
                model_path = base / f"{t}_{first}.pkl" if first else _any_pkl(base)
                with open(model_path, "rb") as z:
                    tmp_model_blob = pickle.load(z)
                scaler = tmp_model_blob["scaler"]

            if roster.exists():
                with open(roster, "rb") as f:
                    roster_info = pickle.load(f)
                    members = roster_info["members"]
                ensemble = {}
                for name in members:
                    with open(base / f"{t}_{name}.pkl", "rb") as f:
                        b = pickle.load(f)
                    ensemble[name] = (b["model"], b["features"])
                return {"scaler": scaler, "ensemble": ensemble, "members_order": members}
        else:
            with open(single if single.exists() else _any_pkl(base), "rb") as f:
                tmp = pickle.load(f)
            scaler = tmp["scaler"]
            with open(single, "rb") as f:
                blob = pickle.load(f)
            return {"scaler": scaler, "single": (blob["model"], blob["features"])}

        raise FileNotFoundError("No suitable model artefacts found.")

    def majority_vote(mat):
        return np.array([Counter(col).most_common(1)[0][0] for col in mat.T])

    def video_vote(pred_series, id_series):
        df_tmp = pd.DataFrame({"id": id_series, "pred": pred_series})
        return df_tmp.groupby("id")["pred"].agg(lambda x: Counter(x).most_common(1)[0][0])

    def compute_binary_specificity(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape != (2, 2): return np.nan
        tn, fp, _, _ = cm.ravel()
        return tn / (tn + fp)

    def print_metrics(prefix, y_true, y_pred):
        print(f"\n{'='*60}\n{prefix} METRICS\n{'-'*60}")
        print(classification_report(y_true, y_pred, digits=6))
        uar = balanced_accuracy_score(y_true, y_pred)
        spec = compute_binary_specificity(y_true, y_pred)
        sens = recall_score(y_true, y_pred, average="binary" if len(np.unique(y_true))==2 else "macro")
        f1 = f1_score(y_true, y_pred, average= "macro")
        print(f"UAR: {uar:.6f}, Spec: {spec:.6f}, Sens: {sens:.6f}, F1: {f1:.6f}")
        return dict(uar=uar, specificity=spec, sensitivity=sens, f1=f1)

    OUT_DIR = f"models_{task}"; os.makedirs(OUT_DIR, exist_ok=True)
    if task == "parkinson":
        X_test_raw = x_dfs_seg["park_test"].copy(); y_test = y_s_seg["park_test"].copy()
    elif task == "depression":
        X_test_raw = x_dfs_seg["depr_test"].copy(); y_test = y_s_seg["depr_test"].copy()
    else:
        X_test_raw = pd.concat([x_dfs_seg["park_test"], x_dfs_seg["depr_test"]], ignore_index=True)
        y_test = pd.concat([y_s_seg["park_test"], y_s_seg["depr_test"]], ignore_index=True)

    art = load_artifacts(task)
    scaler = art["scaler"]
    X_scaled_numeric = pd.DataFrame(
        scaler.transform(X_test_raw[scaler.feature_names_in_]),
        columns=scaler.feature_names_in_, index=X_test_raw.index
    )
    X_scaled = pd.concat([X_scaled_numeric, X_test_raw.drop(columns=scaler.feature_names_in_, errors="ignore")], axis=1)
    if id_col in X_test_raw.columns:
        X_scaled[id_col] = X_test_raw[id_col]

    shap_results = {}
    model_type = "ensemble" if use_ensemble and "ensemble" in art else "single"
    print(f"🔎 Using {model_type.upper()} model(s)")

    if model_type == "single":
        model, feats = art["single"]
        X_final = X_scaled[feats]
        y_pred_row = model.predict(X_final)
        if if_shap:
            sample = X_final.sample(min(1000, len(X_final)))
            shap_vals = shap.Explainer(model, sample)(sample)
            shap_results["single"] = pd.DataFrame({"feature": [clean_feat_name(f) for f in feats], "mean_abs_shap": np.abs(shap_vals.values).mean(axis=0)})
    else:
        pred_matrix = []
        for name in art["ensemble"]:
            mdl, feats = art["ensemble"][name]
            X_m = X_scaled[feats]
            pred_matrix.append(mdl.predict(X_m))
        y_pred_row = majority_vote(np.vstack(pred_matrix))

        best_name = art["members_order"][0]
        best_model, best_feats = art["ensemble"][best_name]
        X_best = X_scaled[best_feats]
        y_pred_best_row = best_model.predict(X_best)

        if if_shap:
            sample = X_best.sample(min(1000, len(X_best)))
            shap_vals = shap.Explainer(best_model, sample)(sample)
            shap_results["best"] = pd.DataFrame({"feature": [clean_feat_name(f) for f in best_feats], "mean_abs_shap": np.abs(shap_vals.values).mean(axis=0)})

            shap_sum, shap_count = defaultdict(float), defaultdict(float)
            for name in art["ensemble"]:
                mdl, feats = art["ensemble"][name]
                X_m = X_scaled[feats]
                sample = X_m.sample(min(500, len(X_m)))
                shap_vals = shap.Explainer(mdl, sample)(sample)
                for feat, val in zip(feats, np.abs(shap_vals.values).mean(axis=0)):
                    shap_sum[clean_feat_name(feat)] += val; shap_count[clean_feat_name(feat)] += 1
            shap_results["ensemble"] = pd.DataFrame({"feature": list(shap_sum), "mean_abs_shap": [shap_sum[f]/shap_count[f] for f in shap_sum]})

    print_metrics("ROW-LEVEL", y_test, y_pred_row)
    if model_type == "ensemble":
        print_metrics("ROW-LEVEL (BEST MODEL)", y_test, y_pred_best_row)

    pd.DataFrame({"y_true": y_test, "y_pred_row": y_pred_row}).to_csv(Path(OUT_DIR) / f"pred_{t}_rows.csv", index=False)
    if model_type == "ensemble":
        pd.DataFrame({"y_true": y_test, "y_pred_best": y_pred_best_row}).to_csv(Path(OUT_DIR) / f"pred_{t}_rows_best.csv", index=False)

    if id_col in X_scaled:
        y_pred_video = video_vote(y_pred_row, X_scaled[id_col])
        y_true_video = video_vote(y_test.values, X_scaled[id_col])
        print_metrics("VIDEO-LEVEL", y_true_video, y_pred_video)
        pd.DataFrame({"video_id": y_true_video.index, "y_true": y_true_video.values, "y_pred": y_pred_video.values}).to_csv(Path(OUT_DIR) / f"pred_{t}_video.csv", index=False)

        if model_type == "ensemble":
            y_pred_video_best = video_vote(y_pred_best_row, X_scaled[id_col])
            print_metrics("VIDEO-LEVEL (BEST MODEL)", y_true_video, y_pred_video_best)
            pd.DataFrame({"video_id": y_true_video.index, "y_true": y_true_video.values, "y_pred": y_pred_video_best.values}).to_csv(Path(OUT_DIR) / f"pred_{t}_video_best.csv", index=False)

    if if_shap:
        for k, df in shap_results.items():
            df["mean_abs_shap"] = df["mean_abs_shap"].apply(lambda x: float(np.mean(x)) if isinstance(x, (np.ndarray, list)) else float(x))

         # Aggregate SHAP values by modality

        print("\nSaving SHAP violin plots and CSVs…")
        for key, df in shap_results.items():
            df.sort_values("mean_abs_shap", ascending=False).to_csv(Path(OUT_DIR) / f"shap_{t}_{key}.csv", index=False)
            top = df.sort_values("mean_abs_shap", ascending=False).head(20)
            if key == "ensemble":
                plt.figure(figsize=(10,6))
                plt.barh(top["feature"], top["mean_abs_shap"], color="purple")
                plt.xlabel("Mean |SHAP value|")
                plt.title(f"Top 20 SHAP Features ({key})")
                plt.gca().invert_yaxis()
                plt.tight_layout()
                plt.savefig(Path(OUT_DIR) / f"shap_violin_{t}_{key}.png", dpi=300)
            else:
                shap.summary_plot(shap_vals, sample, plot_type="violin", max_display=20, show=False)
                plt.tight_layout()
                plt.savefig(Path(OUT_DIR) / f"shap_violin_{t}_{key}.png", dpi=300)
                plt.close()

    print(" All done!")
