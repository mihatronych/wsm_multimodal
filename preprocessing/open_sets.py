import pandas as pd
from pathlib import Path
import os
import subprocess
from glob import glob
import cv2# Check
from datetime import timedelta

def with_opencv(filename): 
    video = cv2.VideoCapture(filename)
    if not video.isOpened():
        print("Error: Could not open video.")
        return
        
    duration = video.get(cv2.CAP_PROP_POS_MSEC)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)

    video_fps = video.get(cv2.CAP_PROP_FPS)

    return duration, frame_count, video_fps


def check_videos_in_sets(wsm_sets):
    sets_dfs_dict = {"depression": {}, "parkinson": {}}
    for set_csv_path in wsm_sets:
        path = Path(set_csv_path)
        dirname = path.parent
        set_dir = path.stem
        set_df = pd.read_csv(set_csv_path)
        video_ids = list(set_df["video_id"])
        if_public_ar = []
        durations = []
        durations_min = []
        frame_counts = []
        frame_per_secs = []
        for v_id in video_ids:
            path_glob = glob(f"./{dirname}/{set_dir}/{v_id}/*.mp4")
            if path_glob:
                if_public_ar.append(1)
                duration, frame_count, fps = with_opencv(path_glob[0])
                durations.append(frame_count / fps)
                durations_min.append(frame_count / fps / 60)
                frame_counts.append(frame_count)
                frame_per_secs.append(fps)
                # frame_per_secs.append(frame_count / duration)
            else:
                if_public_ar.append(0)
                durations.append(0)
                durations_min.append(0)
                frame_counts.append(0)
                frame_per_secs.append(0)
                
        set_df["if_public"] = if_public_ar
        set_df["duration"] = durations
        set_df["duration_min"] = durations_min
        set_df["frame_count"] = frame_counts
        set_df["fps"] = frame_per_secs
        set_df.to_csv(f"{dirname}/{set_dir}.csv", index=False)
        sets_dfs_dict[str(dirname)][str(set_dir)] = set_df
    return sets_dfs_dict