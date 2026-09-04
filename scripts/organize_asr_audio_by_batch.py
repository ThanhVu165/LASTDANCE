"""Organize audio files into batch folders for Kaggle upload.

Audio files organized as: batch-01/, batch-02/, ..., batch-09/
Each folder contains FLAC files for corresponding batch.
"""

import shutil
from pathlib import Path

import pandas as pd

from offline.asr_production import DEFAULT_ASR_BATCH_IDS


def main():
    """Organize audio files by batch."""
    # Paths
    audio_dir = Path("F:\\LASTDANCE-DATA\\data\\audio")
    frames_csv = Path("C:\\LASTDANCE\\data\\index\\frames.csv")
    
    # Load frames.csv to get video_id -> global index mapping
    frames = pd.read_csv(frames_csv)
    
    # Create mapping: video_id -> batch_id
    # Batch IDs are assigned based on global order of unique video_ids
    unique_videos = frames["video_id"].unique()
    video_to_batch = {}
    
    batch_size = 97  # Batch 01 has 97 videos
    for i, video_id in enumerate(unique_videos):
        batch_index = min(i // batch_size, 8)  # 0-8 for batch-01 to batch-09
        batch_id = DEFAULT_ASR_BATCH_IDS[batch_index]
        video_to_batch[video_id] = batch_id
    
    # Create batch folders
    batch_dirs = {}
    for batch_id in DEFAULT_ASR_BATCH_IDS:
        batch_dir = audio_dir.parent.parent / "batch-audio" / batch_id
        batch_dir.mkdir(parents=True, exist_ok=True)
        batch_dirs[batch_id] = batch_dir
    
    # Organize audio files
    print("Organizing audio files by batch...")
    for audio_file in audio_dir.glob("*.flac"):
        # Extract video_id from filename (e.g., L21_V001.flac -> L21_V001)
        video_id = audio_file.stem
        
        if video_id not in video_to_batch:
            print(f"Warning: {video_id} not found in frames.csv, skipping")
            continue
        
        batch_id = video_to_batch[video_id]
        dest_path = batch_dirs[batch_id] / audio_file.name
        
        # Copy file to batch folder
        shutil.copy2(audio_file, dest_path)
        print(f"* {audio_file.name} -> {batch_id}/")
    
    # Print summary
    print("\n=== Summary ===")
    for batch_id in DEFAULT_ASR_BATCH_IDS:
        batch_dir = batch_dirs[batch_id]
        count = len(list(batch_dir.glob("*.flac")))
        print(f"{batch_id}: {count} files")
    
    print(f"\n* Audio organized into: F:\\LASTDANCE-DATA\\batch-audio/")
    print("Ready to upload to Kaggle Dataset.")


if __name__ == "__main__":
    main()
