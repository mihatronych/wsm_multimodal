
features_subsets_hc_at = {"hc_aud":features_subsets_hc["hc_aud"], "hc_txt":features_subsets_hc["hc_txt"], "hc_txt+aud":features_subsets_hc["hc_txt"] + features_subsets_hc["hc_aud"]}
features_subsets_at = {"hc_aud":features_subsets_hc["hc_aud"], "hc_txt":features_subsets_hc["hc_txt"], "hc_txt+aud":features_subsets_hc["hc_txt"] + features_subsets_hc["hc_aud"], "mfcc":features_subsets["mfcc"], "hc_txt+aud+mfcc":features_subsets_hc["hc_txt"] + features_subsets_hc["hc_aud"] + features_subsets["mfcc"]}
features_subsets_hc_v = {"emo":features_subsets_hc["emo"], "pose":features_subsets_hc["pose_visible"], "ocean":features_subsets_hc["ocean"], "hc_vid":features_subsets_hc["hc_vid"]}
features_subsets_v = {"emo":features_subsets_hc["emo"], "pose":features_subsets_hc["pose_visible"], "ocean":features_subsets_hc["ocean"], "hc_vid":features_subsets_hc["hc_vid"], "nn":features_subsets["nn"]}


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # 1. Prepare your train/test dataframes & targets (replace placeholders)
    # ------------------------------------------------------------------
    # These must exist in the surrounding notebook / script
    #   x_dfs_seg, y_s_seg, features_subsets
    # Example:
    # X_train = x_dfs_seg["park_train"]
    # y_train = y_s_seg["park_train"]
    task = "parkinson_expl_at"
     # 1. Prepare train/test based on task
    if "parkinson" in task:
        X_train = x_dfs_seg["park_train"].copy()
        y_train = y_s_seg["park_train"].copy()
        X_test  = x_dfs_seg["park_test"].copy()
        y_test  = y_s_seg["park_test"].copy()
    elif "depression" in task:
        X_train = x_dfs_seg["depr_train"].copy()
        y_train = y_s_seg["depr_train"].copy()
        X_test  = x_dfs_seg["depr_test"].copy()
        y_test  = y_s_seg["depr_test"].copy()
    else:  # both
        # Concatenate Parkinson + Depression
        # Reset index or ignore_index so normalization works cleanly
        X_train = pd.concat(
            [x_dfs_seg["park_train"], x_dfs_seg["depr_train"]],
            ignore_index=True
        )
        y_train = pd.concat(
            [y_s_seg["park_train"], y_s_seg["depr_train"]],
            ignore_index=True
        )
        X_test  = pd.concat(
            [x_dfs_seg["park_test"], x_dfs_seg["depr_test"]],
            ignore_index=True
        )
        y_test  = pd.concat(
            [y_s_seg["park_test"], y_s_seg["depr_test"]],
            ignore_index=True
        )
    print(f"  → Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    # ------------------------------------------------------------------
    # 2. Normalise (scaling fitted **only** on training data!)
    # ------------------------------------------------------------------
    X_train_norm, X_test_norm, fitted_scaler = normalise_train_test(X_train, X_test)
    # ------------------------------------------------------------------
    # 3. Define classifiers & hyper‑parameter grids
    # ------------------------------------------------------------------
    models = {
        "RF": RandomForestClassifier(random_state=42),
        "LogReg": LogisticRegression(max_iter=1000, random_state=42),
        #"GradientBoosting": GradientBoostingClassifier(random_state=42),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        # "MLP": MLPClassifier(max_iter=1000, random_state=42),
        "SVM": SVC(probability=True, random_state=42),
    }

    feature_subsets_map = {key: [] for key in models.keys()}

    param_grids = {
        "RF": {
            "n_estimators": [10, 50, 100, 200],
            "max_features": ["sqrt", "log2"],
        },
        "LogReg": {"C": np.linspace(0.1, 1.0, 10)},
        "GradientBoosting": {
            "n_estimators": [10, 50, 100, 200],
            "learning_rate": [0.005, 0.01, 0.05, 0.1],
            "max_features": [100, 30, "sqrt", "log2"],
            "n_iter_no_change": [None, 10, 30, 50, 100],
        },
        "DecisionTree": {
            "max_depth": [3, 5, 10, 20, 50, None],
            "max_features": [200, 100, 50, 30, "sqrt", "log2"],
        },
        # "MLP": {
        #     "hidden_layer_sizes": [(50,), (100,), (200,), (500,)],
        #     "alpha": [0.0001, 0.0005, 0.001],
        #     "early_stopping": [False, True],
        # },
        "SVM": {
            "C": np.linspace(0.1, 1.0, 10),
            "kernel": ["linear", "rbf"],
        },
    }

    # ------------------------------------------------------------------
    # 4. Train & evaluate
    # ------------------------------------------------------------------
    df_results, trained_models, ensemble_members = train_evaluate_models_with_multiple_feature_subsets(
        models=models,
        X_train=X_train_norm,
        X_test=X_test_norm,
        y_train=y_train,
        y_test=y_test,
        feature_subsets_map=feature_subsets_map,
        feature_list=features_subsets_hc_at,
        param_grids=param_grids,
        task_type="classification",
        verbose=True,
        n_best_models=5,
        r=1,
    )

    print("\n🏁 Training finished – top results:\n", df_results.head(10))

     # 5. Persist the best model(s) & scaler, including task in filenames
    best_row = df_results.iloc[0]
    best_model_name = best_row["model"]

    # Create a directory to store models per task
    out_dir = f"models_{task}"
    os.makedirs(out_dir, exist_ok=True)

    if best_model_name in ensemble_members:
        roster = ensemble_members[best_model_name]  # list of base model names
        print(f"🏆 Ensemble '{best_model_name}' won. Saving its {len(roster)} members:")
        for name in roster:
            mdl  = trained_models[name]
            feats = df_results[df_results.model == name]["feature_subset"].values[0]
            # Filename includes task and model name
            fname = os.path.join(out_dir, f"{task}_{name}.pkl")
            with open(fname, "wb") as f:
                pickle.dump({
                    "model": mdl,
                    "scaler": fitted_scaler,
                    "features": feats,
                    "task": task
                }, f)
            print(f"   • {fname}")
        # additionally store the roster for inference
        roster_path = os.path.join(out_dir, f"{task}_ensemble_roster.pkl")
        with open(roster_path, "wb") as f:
            pickle.dump({"members": roster}, f)
        print(f"✅ Roster file '{roster_path}' written.")
    else:
        best_model = trained_models[best_model_name]
        best_features = best_row["feature_subset"]
        fname = os.path.join(out_dir, f"{task}_best_model.pkl")
        with open(fname, "wb") as f:
            pickle.dump({
                "model": best_model,
                "scaler": fitted_scaler,
                "features": best_features,
                "task": task
            }, f)
        print(f"✅ Best model '{best_model_name}' saved to {fname}")

    # 6. SHAP feature‑importance visualisation
    try:
        if best_model_name in ensemble_members:
            # Aggregated SHAP for ensemble
            print("Computing aggregated SHAP for ensemble members...")
            full_features = X_train_norm.columns.tolist()
            shap_series_list = []
            for name in roster:
                model_i = trained_models[name]
                feats_i = df_results[df_results.model == name]["feature_subset"].values[0]
                # Optionally sample if X_sel is large:
                X_sel = X_train_norm[feats_i]
                # e.g., sample subset:
                # if len(X_sel) > 1000:
                #     X_sel_sample = X_sel.sample(n=1000, random_state=0)
                # else:
                #     X_sel_sample = X_sel
                X_sel_sample = X_sel  # or sample
                explainer_i = shap.Explainer(model_i, X_sel_sample)
                shap_vals_i = explainer_i(X_sel_sample)
                shap_vals = shap_vals_i.values

                # Check for list of arrays (e.g., [array([..]), array([..]), ...])
                if isinstance(shap_vals, list):
                    try:
                        # Convert to 2D array
                        shap_array = np.array([np.ravel(val) for val in shap_vals])
                    except Exception:
                        raise ValueError("SHAP values could not be flattened to a 2D array.")
                else:
                    shap_array = shap_vals  # already a 2D array

                mean_abs_i = np.abs(shap_array).mean(axis=0)
                s = pd.Series(0.0, index=full_features)
                s.loc[feats_i] = mean_abs_i
                shap_series_list.append(s)
            shap_agg = pd.concat(shap_series_list, axis=1).mean(axis=1)
            shap_agg_sorted = shap_agg.sort_values(ascending=False)
            topN = min(30, len(shap_agg_sorted))
            top_features = shap_agg_sorted.iloc[:topN]
            plt.figure(figsize=(8, max(4, topN*0.3)))
            top_features.plot.barh()
            plt.gca().invert_yaxis()
            plt.xlabel("Mean(|SHAP value|) across ensemble")
            plt.title(f"{task.capitalize()} ensemble SHAP importance")
            plt.tight_layout()
            plot_path = os.path.join(out_dir, f"shap_{task}_ensemble_summary.png")
            plt.savefig(plot_path, dpi=300)
            plt.close()
            print(f"📊 Ensemble SHAP summary saved to {plot_path}")
        else:
            print(f"Computing SHAP for single best model '{best_model_name}'...")
            X_train_best = X_train_norm[best_features]
            X_sel_sample = X_train_best  # or sample if large
            explainer = shap.Explainer(best_model, X_sel_sample)
            shap_values = explainer(X_sel_sample)
            plt.figure()  # optional figsize
            shap.summary_plot(shap_values, X_sel_sample, show=False, max_display=30)
            plt.tight_layout()
            plot_path = os.path.join(out_dir, f"shap_{task}_best_model_summary.png")
            plt.savefig(plot_path, dpi=300)
            plt.close()
            print(f"📊 SHAP summary plot saved to {plot_path}")
    except Exception as exc:
        print("[!] Could not generate SHAP plot:", exc)