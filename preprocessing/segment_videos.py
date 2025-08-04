import os
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def process_video(
    video_path: Path,
    output_subdir: Path,
    segment_length_sec: int = 60,
    fps: int = 5,
    audio_rate: int = 16000
):
    video_name = video_path.stem
    output_folder = output_subdir / "segments"
    output_folder.mkdir(parents=True, exist_ok=True)

    output_pattern = output_folder / f"{video_name}_%03d.mp4"

    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-r", str(fps),
        "-ar", str(audio_rate),
        "-f", "segment",
        "-segment_time", str(segment_length_sec),
        "-reset_timestamps", "1",
        str(output_pattern),
        "-y"
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        return f"Done: {video_path}"
    except subprocess.CalledProcessError:
        return f"Error: {video_path}"

def split_videos_recursive(
    root_dir: str,
    segment_length_sec: int = 60,
    fps: int = 5,
    audio_rate: int = 16000,
    max_workers: int = 4
):
    root = Path(root_dir)
    video_files = list(root.rglob("*.mp4"))  # recursively search for .mp4

    if not video_files:
        print("No videos found.")
        return

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                process_video,
                video_path,
                video_path.parent,
                segment_length_sec,
                fps,
                audio_rate
            )
            for video_path in video_files
        ]

        for f in tqdm(as_completed(futures), total=len(futures), desc="📦 Processing videos"):
            print(f.result())

# ▶️ Example usage:
if __name__ == "__main__":
    split_videos_recursive(
        root_dir="videos",      # Path where each video is in its own folder
        segment_length_sec=60,
        fps=5,
        audio_rate=16000,
        max_workers=4
    )