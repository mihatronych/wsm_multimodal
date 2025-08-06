from .video_build import _b5

def extract_video_features_from_ds_OAI_seg(ds, ds_type="devel", disease="depression", modality="vid"):
    features = []
    i = 0
    new_ds = ds
    for k, v_id in enumerate(list(ds["video_id"])):
        i += 1
        print(v_id, str(i))
        path = f"./{disease}/{ds_type}_labels/{v_id}"
        if glob(f'{path}/segments/*.mp4'):
            vs = glob(f'{path}/segments/*.mp4')
            for n, v in enumerate(vs):
                try:
                    hc_features, nn_features, emo_preds = _b5.get_visual_features(
                    path = v, # Путь к видеофайлу
                    reduction_fps = 5, # Понижение кадровой частоты
                    window = 10, # Размер окна сегмента сигнала (в кадрах)
                    step = 5, # Шаг сдвига окна сегмента сигнала (в кадрах)
                    lang = 'en',
                    out = False, # Отображение
                    runtime = True, # Подсчет времени выполнения
                    run = True # Блокировка выполнения
                    )
                    v_feat_df = ds.iloc[[k]]
                    v_feat_df["path"] = Path(glob(f'{path}/*.mp4')[0]).name
                    v_feat_df["seg_num"] = f"{n:03d}"
                    v_feat_df["seg_path"] = Path(v).name
                    hc_features_mean_w = np.mean(hc_features, axis=1)
                    hc_features_mean_v_mean = np.mean(hc_features_mean_w, axis=0).tolist()
                    hc_features_mean_v_std = np.std(hc_features_mean_w, axis=0).tolist()
                    nn_features_mean_w = np.mean(nn_features, axis=1)
                    nn_features_mean_v_mean = np.mean(nn_features_mean_w, axis=0).tolist()
                    emo_preds_mean_w = np.mean(emo_preds, axis=1)
                    emo_preds_mean_v_mean = np.mean(emo_preds_mean_w, axis=0).tolist()
                    emo_preds_mean_v_std = np.std(emo_preds_mean_w, axis=0).tolist()
                    emo_preds_mean_df = pd.DataFrame(columns=[f"emo_{str(e)}_mean" for e in range(len(emo_preds_mean_v_mean))], data=[emo_preds_mean_v_mean])
                    emo_preds_std_df = pd.DataFrame(columns=[f"emo_{str(e)}_std" for e in range(len(emo_preds_mean_v_std))], data=[emo_preds_mean_v_std])
                    hc_features_mean_df = pd.DataFrame(columns=[f"hc_vid_{str(e)}_mean" for e in range(len(hc_features_mean_v_mean))], data=[hc_features_mean_v_mean])
                    hc_features_std_df = pd.DataFrame(columns=[f"hc_vid_{str(e)}_std" for e in range(len(hc_features_mean_v_std))], data=[hc_features_mean_v_std])
                    nn_features_mean_df = pd.DataFrame(columns=[f"nn_{str(e)}_mean" for e in range(len(nn_features_mean_v_mean))], data=[nn_features_mean_v_mean])
                    features_concat = pd.concat([v_feat_df, emo_preds_mean_df, emo_preds_std_df, hc_features_mean_df, hc_features_std_df, nn_features_mean_df], axis=1)
                    features.append(features_concat)
                except KeyboardInterrupt:
                    return
                except:
                    continue
        else: 
            new_ds = new_ds[new_ds["video_id"]!=v_id].reset_index(drop=True)
    concat_dif_samples_fe_df = pd.concat(features, axis=0, ignore_index=True)
    concat_dif_samples_fe_df.reset_index(drop=True)
    concat_dif_samples_fe_df.to_csv(f"{disease}/{ds_type}_w_fe_{modality}_seg.csv", index=False)
    return concat_dif_samples_fe_df