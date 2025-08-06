import pandas as pd
import os

def load_datasets(data_dirs):
    """
    Load CSVs from structured directory input:
    Example:
        data_dirs = {
            "depression": "depression/",
            "parkinson": "parkinson/"
        }
    Returns a dictionary of DataFrames for each split (train, test, devel)
    """
    datasets = {}
    for condition, path in data_dirs.items():
        for split in ["train", "test", "devel"]:
            file_path = os.path.join(path, f"{split}_w_fe_seg.csv")
            if os.path.exists(file_path):
                key = f"{condition}_{split}"
                df = pd.read_csv(file_path).dropna()
                if 'self-reported diagnosis' in df.columns:
                    df = df.rename(columns={'self-reported diagnosis': 'diagnosis'})
                datasets[key] = df
            else:
                print(f"File not found: {file_path}")
    return datasets


def split_features(x_dfs, keep_ids=False):
    """
    Removes unnecessary fields and extracts useful feature groups.
    """
    cleaned_dfs = {}
    for k, v in x_dfs.items():
        drop_cols = [c for c in v.columns if any(p in c for p in [
            "diagnosis", "channel_id", "duration", "frame_count", "fps", "Path", "Person ID", "if_public"
        ])]
        if not keep_ids:
            drop_cols += [c for c in v.columns if "video_id" in c]
        v_cleaned = v.drop(columns=[col for col in drop_cols if col in v.columns], errors='ignore')
        cleaned_dfs[k] = v_cleaned

    reference_df = list(cleaned_dfs.values())[0]  # Use any split as reference for feature keys

    features_subsets = {
        "id": [c for c in reference_df.columns if "video_id" in c],
        "nn": [c for c in reference_df.columns if "nn_" in c and "std" not in c],
        "hc_aud": [c for c in reference_df.columns if "hc_aud" in c and "mfcc" not in c],
        "hc_vid": [c for c in reference_df.columns if "hc_vid" in c],
        "hc_txt": [c for c in reference_df.columns if "hc_txt" in c],
    }

    # Range-based extraction
    def slice_cols(start_col, count):
        idx = reference_df.columns.get_loc(start_col)
        return reference_df.columns[idx:idx + count].tolist()

    if 'neutral_mean' in reference_df.columns:
        features_subsets["emo"] = slice_cols('neutral_mean', 14)
    if 'Openness' in reference_df.columns:
        features_subsets["ocean"] = slice_cols('Openness', 5)
    if 'nose_mean_x' in reference_df.columns:
        features_subsets["pose_visible"] = [c for c in slice_cols('nose_mean_x', 253) if "visibility" not in c]

    features_subsets["mfcc"] = [c for c in reference_df.columns if "mfcc" in c]

    # Flat lists for universal use
    not_required = ['nn', 'mfcc', 'all']
    required_fields = [
        col for name, cols in features_subsets.items()
        if name not in not_required
        for col in cols
    ]
    features_subsets["all"] = [c for c in required_fields if "video_id" not in c]
    features_subsets["all_hc"] = [c for c in features_subsets["all"] if "nn_" not in c and "mfcc" not in c]

    features_subsets_hc = {k: v for k, v in features_subsets.items() if k not in not_required}

    return cleaned_dfs, features_subsets, features_subsets_hc


def extract_labels(datasets):
    return {k: df["diagnosis"] for k, df in datasets.items() if "diagnosis" in df.columns}


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

    # If you want versions WITH video_id, call it with keep_ids=True
    x_dfs_seg_with_id, _, _ = split_features(x_dfs_seg, keep_ids=True)