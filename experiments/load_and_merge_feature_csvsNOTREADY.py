# Дооформить
depr_devel_w_fe_seg = pd.read_csv("depression/devel_w_fe_seg.csv").dropna()
depr_test_w_fe_seg =pd.read_csv("depression/test_w_fe_seg.csv").dropna()
depr_train_w_fe_seg = pd.read_csv("depression/train_w_fe_seg.csv").dropna()
depr_train_w_fe_seg.rename(columns={'self-reported diagnosis': 'diagnosis'}, inplace=True)
park_devel_w_fe_seg = pd.read_csv("parkinson/devel_w_fe_seg.csv").dropna()
park_test_w_fe_seg = pd.read_csv("parkinson/test_w_fe_seg.csv").dropna()
park_train_w_fe_seg = pd.read_csv("parkinson/train_w_fe_seg.csv").dropna()


def split_dfs_by_feats(x_dfs):
    x_dfs = {k: v.drop(["diagnosis_x", 'channel_id_x', 'channel_id_x', "duration_x", "duration_min_x", 
    "frame_count_x", "fps_x", "Path", "Person ID", "if_public_x"], axis=1) for k, v in x_dfs.items()}

    features_subsets = {
            "id": [c_nn for c_nn in x_dfs["depr_devel"].columns if "video_id" in c_nn],
            "nn": [c_nn for c_nn in x_dfs["depr_devel"].columns if "nn_" in c_nn and "std" not in c_nn],
            "hc_aud": [c_hc for c_hc in x_dfs["depr_devel"].columns if "hc_aud" in c_hc and "mfcc" not in c_hc],
            "hc_vid": [c_hc for c_hc in x_dfs["depr_devel"].columns if "hc_vid" in c_hc],
            "hc_txt": [c_hc for c_hc in x_dfs["depr_devel"].columns if "hc_txt" in c_hc],
            "emo": [i for i in x_dfs["depr_devel"].columns[x_dfs["depr_devel"].columns.get_loc('neutral_mean'):x_dfs["depr_devel"].columns.get_loc('neutral_mean')+14]],
            "ocean": [i for i in x_dfs["depr_devel"].columns[x_dfs["depr_devel"].columns.get_loc('Openness'):x_dfs["depr_devel"].columns.get_loc('Openness')+5]],
            "pose_visible": [i for i in x_dfs["depr_devel"].columns[x_dfs["depr_devel"].columns.get_loc( 'nose_mean_x'):x_dfs["depr_devel"].columns.get_loc( 'nose_mean_x')+253] if "visibility" not in i], # not in i and any(l in i for l in visible_list)
            "mfcc": [c for c in x_dfs["depr_devel"].columns if "mfcc" in c],
    }
    
    return x_dfs, features_subsets

x_dfs_seg = {
    "depr_devel": depr_devel_w_fe_seg,
    "depr_test": depr_test_w_fe_seg,
    "depr_train": depr_train_w_fe_seg,
    "park_devel": park_devel_w_fe_seg,
    "park_test": park_test_w_fe_seg,
    "park_train": park_train_w_fe_seg,
}

y_s_seg ={
    "depr_devel": depr_devel_w_fe_seg["diagnosis"],
    "depr_test": depr_test_w_fe_seg["diagnosis"],
    "depr_train": depr_train_w_fe_seg["diagnosis"],
    "park_devel": park_devel_w_fe_seg["diagnosis"],
    "park_test": park_test_w_fe_seg["diagnosis"],
    "park_train": park_train_w_fe_seg["diagnosis"]
}  

not_required_fields = ['nn', 'mfcc', 'all']

x_dfs, features_subsets = split_dfs_by_feats(x_dfs, False)

lis = [v for i, v in features_subsets.items()]
required_fields = [element for innerList in lis for element in innerList if "video_index" != element]

features_subsets["all"] = [i for i in required_fields if "video_id" not in i]
features_subsets["all_hc"] = [i for i in required_fields if "nn_" not in i and "mfcc" not in i and "video_id" not in i]
features_subsets_hc = {key:value for key, value in features_subsets.items() if key not in not_required_fields}

x_dfs = {k: v[required_fields] for k,v in x_dfs.items()}
x_dfs_seg_with_id, features_subsets_seg = split_dfs_by_feats(x_dfs_seg, True)
x_dfs_seg, features_subsets_seg = split_dfs_by_feats(x_dfs_seg, True)
features_subsets_seg["all"] = required_fields
features_subsets_seg["all_hc"] = [i for i in required_fields if "nn_" not in i and "mfcc" not in i]
features_subsets_seg_hc = {key:value for key, value in features_subsets_seg.items() if key not in not_required_fields}

x_dfs_seg_with_id = {k: v[required_fields] for k,v in x_dfs_seg_with_id.items()}
x_dfs_seg = {k: v[required_fields] for k,v in x_dfs_seg.items()}