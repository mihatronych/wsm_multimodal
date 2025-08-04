from pathlib import Path  
import pandas as pd
import numpy as np
import pandas as pd

def load_visual_features_wOCEAN(disease="depression", subset="devel", seg=True): # disease = "depression" / "parkinson"; subset = "devel", "train", "test"
    ds_w_fe_vid = pd.read_csv(f"{disease}/{subset}_w_fe_vid_seg.csv").dropna()
    ds_w_fe_vid = ds_w_fe_vid.set_axis(
        [c if "emo" not in c else f"{emo_dict[int(c[4])]}{c[5:]}" for c in ds_w_fe_vid.columns.tolist()], 
        axis=1)
    ds_w_fe_vid["Path"] = ds_w_fe_vid["seg_path"]
    ocean_ds = pd.read_csv(glob(f'{disease}/{subset}_labels/logs_seg/*.csv')[0])
    united_ds = pd.merge(ds_w_fe_vid, ocean_ds, on='Path', how='outer').dropna()
    united_ds.to_csv(f"{disease}/{subset}_w_fe_vid_ocean.csv", index=False)
    return united_ds