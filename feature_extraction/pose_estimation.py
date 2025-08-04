import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path

def summarize_pose_with_labels_and_confidence(
    df: pd.DataFrame,
    video_path_column: str = 'video_path',
    src_path=".",
    dimensions: int = 3,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Extract 1 FPS pose data from videos using MediaPipe and return
    per-video summary with joint-wise mean/std for coordinates and confidence.

    Args:
        df (pd.DataFrame): DataFrame with video paths.
        video_path_column (str): Column with video file paths.
        dimensions (int): 2 or 3 for coordinate dimensions.
        verbose (bool): If True, show warnings.

    Returns:
        pd.DataFrame: Summary DataFrame with labeled features.
    """
    
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False)
    landmark_names = list(mp_pose.PoseLandmark)

    coord_labels = ['x', 'y']
    if dimensions == 3:
        coord_labels.append('z')
    coord_labels.append('visibility')

    summaries = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        video_name = row[video_path_column]
        try:
            video_path = [str(path) for path in Path(src_path).rglob(video_name)][0]
            cap = cv2.VideoCapture(video_path)
        except KeyboardInterrupt:
                    return
        except:
            continue
        
        if not cap.isOpened():
            if verbose:
                print(f"!!! Cannot open video: {video_path}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            if verbose:
                print(f"!!! Invalid FPS for video: {video_path}")
            continue

        frame_interval = int(fps)
        frame_id = 0
        keypoints_list = []

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            if frame_id % frame_interval == 0:
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)

                if results.pose_landmarks:
                    coords = []
                    for lm in results.pose_landmarks.landmark:
                        joint_values = [lm.x, lm.y]
                        if dimensions == 3:
                            joint_values.append(lm.z)
                        joint_values.append(lm.visibility)
                        coords.extend(joint_values)
                    keypoints_list.append(coords)

            frame_id += 1

        cap.release()

        if keypoints_list:
            arr = np.array(keypoints_list)
            feature_dict = {'video_index': idx, 'Path': video_name}

            for j, joint in enumerate(landmark_names):
                for d, dim in enumerate(coord_labels):
                    col_index = j * len(coord_labels) + d
                    mean_val = np.mean(arr[:, col_index])
                    std_val = np.std(arr[:, col_index])
                    joint_label = joint.name.lower()
                    feature_dict[f'{joint_label}_mean_{dim}'] = mean_val
                    feature_dict[f'{joint_label}_std_{dim}'] = std_val

            summaries.append(feature_dict)

    summary_df = pd.DataFrame(summaries)
    united_ds = pd.merge(df, summary_df, on='Path', how='outer').dropna()
    return united_ds