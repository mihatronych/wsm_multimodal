import time
import pandas as pd
from .blabla_features_list import blabla_features_list


proc = "" # blabla_proc
with DocumentProcessor('C:/Users/Mike/blabla/stanza_config/stanza_config.yaml', 'en') as doc_proc:
    proc = doc_proc

def extract_text_features_from_ds_seg_blabla(ds, ds_type="devel", disease="depression", modality="blabla"):
    features = []
    i = 0
    new_ds = ds
    for k, v_id in enumerate(list(ds["video_id"])):
        i += 1
        print(v_id, str(i))
        path = f"./{disease}/{ds_type}_labels/{v_id}"
        if glob(f'{path}/segments/*.json'):
            jss = glob(f'{path}/segments/*.json')
            for n, jsn in enumerate(jss):
                start_time = time.time()
                content = open(jsn).read()
                doc = doc_proc.analyze(content, 'json')
                try:
                    v_feat_df = ds.iloc[[n]]
                    v_feat_df["path"] = Path(glob(f'{path}/*.mp4')[0]).name
                    v_feat_df["seg_num"] = f"{n:03d}"
                    v_feat_df["seg_path"] = Path(jsn).name
                    blabla_features = doc.compute_features(blabla_features_list)
                    blabla_features = {k:[v] for k,v in blabla_features.items()}
                    blabla_features_df = pd.DataFrame.from_dict(blabla_features)
                    features_concat = pd.concat([v_feat_df, blabla_features_df], axis=1)
                    features.append(features_concat)
                    print("--- %s seconds ---" % (time.time() - start_time))
                except Exception as e:
                    print(e)
                    print("--- %s seconds ---" % (time.time() - start_time))
        else:
            new_ds = new_ds[new_ds["video_id"]!=v_id].reset_index(drop=True)
    
    concat_dif_samples_fe_df = pd.concat(features, axis=0, ignore_index=True)
    full_df = pd.concat([concat_dif_samples_fe_df], axis=1)
    full_df.to_csv(f"{disease}/{ds_type}_w_fe_{modality}_seg.csv", index=False)
    return full_df 