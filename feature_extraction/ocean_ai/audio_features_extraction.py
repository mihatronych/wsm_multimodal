from .video_build import _b5
import opensmile
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
)
smile.feature_names

def extract_audio_features_from_ds_seg_OAI(ds, ds_type="devel", disease="depression", sr=16000, modality="aud"):
    features = []
    i = 0
    new_ds = ds
    for k, v_id in enumerate(list(ds["video_id"])):
        i += 1
        print(v_id, str(i))
        path = f"./{disease}/{ds_type}_labels/{v_id}"
        if glob(f'{path}/segments/*.wav'):
            auds = glob(f'{path}/segments/*.wav')
            for n, a in enumerate(auds):
                try:
                    hc_features, nn_features = _b5.get_acoustic_features(
                        path = a, # Путь к аудио или видеофайлу
                        sr = sr, # Частота дискретизации
                        window = 10, # Размер окна сегмента сигнала (в секундах)
                        step = 5, # Шаг сдвига окна сегмента сигнала (в секундах)
                        out = False, # Отображение
                        runtime = True, # Подсчет времени выполнения
                        run = True # Блокировка выполнения
                    )
                    v_feat_df = ds.iloc[[k]]
                    v_feat_df["path"] = Path(glob(f'{path}/*.mp4')[0]).name
                    v_feat_df["seg_num"] = f"{n:03d}"
                    v_feat_df["seg_path"] = Path(a).name
                    hc_features_mean_w = np.mean(hc_features, axis=1)
                    hc_features_mean_v_mean = np.mean(hc_features_mean_w, axis=0).tolist()
                    hc_features_mean_v_std = np.std(hc_features_mean_w, axis=0).tolist()
                    nn_features_z = np.mean(nn_features, axis=0)
                    nn_features_mean_w = np.mean(nn_features_z, axis=1)
                    nn_features_mean_v_mean = np.mean(nn_features_mean_w, axis=1).tolist()
                    nn_features_mean_v_std = np.mean(nn_features_mean_w, axis=1).tolist()
                    hc_features_mean_df = pd.DataFrame(columns=[f"hc_{smile.feature_names[e]}_mean" for e in range(len(smile.feature_names))], data=[hc_features_mean_v_mean])
                    hc_features_std_df = pd.DataFrame(columns=[f"hc_{smile.feature_names[e]}_std" for e in range(len(smile.feature_names))], data=[hc_features_mean_v_std])
                    nn_features_mean_df = pd.DataFrame(columns=[f"mfcc_{str(e)}_mean" for e in range(len(nn_features_mean_v_mean))], data=[nn_features_mean_v_mean])
                    nn_features_std_df = pd.DataFrame(columns=[f"mfcc_{str(e)}_std" for e in range(len(nn_features_mean_v_std))], data=[nn_features_mean_v_std])
                    features_concat = pd.concat([v_feat_df, hc_features_mean_df, hc_features_std_df, nn_features_mean_df, nn_features_std_df], axis=1)
                    features.append(features_concat)
                except KeyboardInterrupt:
                    return
                except Exception as e:
                    print(e)
                    continue
        else: 
            new_ds = new_ds[new_ds["video_id"]!=v_id].reset_index(drop=True)
    concat_dif_samples_fe_df = pd.concat(features, axis=0, ignore_index=True)
    concat_dif_samples_fe_df.reset_index(drop=True)
    concat_dif_samples_fe_df.to_csv(f"{disease}/{ds_type}_w_fe_{modality}_seg.csv", index=False)
    return concat_dif_samples_fe_df