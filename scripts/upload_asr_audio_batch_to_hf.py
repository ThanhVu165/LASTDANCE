"""Upload organized audio batches to HuggingFace Dataset."""

from pathlib import Path
from huggingface_hub import HfApi, CommitOperationAdd
import sys

def upload_batch_to_hf(batch_dir: Path, repo_id: str = "Vu165/lastdance-asr"):
    """Upload a single batch folder to HF."""
    api = HfApi()
    batch_id = batch_dir.name
    
    print(f"Uploading {batch_id}...")
    
    audio_files = sorted(batch_dir.glob("*.flac"))
    print(f"  Found {len(audio_files)} audio files")
    
    # Create operations for all files in this batch
    operations = []
    for audio_file in audio_files:
        operations.append(
            CommitOperationAdd(
                path_in_repo=f"asr/audio/{batch_id}/{audio_file.name}",
                path_or_fileobj=audio_file,
            )
        )
    
    # Upload batch
    try:
        api.create_commit(
            repo_id=repo_id,
            repo_type="dataset",
            operations=operations,
            commit_message=f"Add {batch_id} audio files"
        )
        print(f"[OK] {batch_id} uploaded successfully")
        return True
    except Exception as e:
        error_msg = str(e)
        # Skip if no files modified (already uploaded)
        if "No files have been modified" in error_msg:
            print(f"[SKIP] {batch_id} already uploaded")
            return True
        print(f"[FAIL] {batch_id} upload failed: {error_msg}")
        return False

def main():
    """Upload all batches sequentially."""
    batch_audio_dir = Path("F:\\LASTDANCE-DATA\\batch-audio")
    
    if not batch_audio_dir.exists():
        print(f"Error: {batch_audio_dir} not found")
        sys.exit(1)
    
    batch_dirs = sorted([d for d in batch_audio_dir.iterdir() if d.is_dir()])
    print(f"Found {len(batch_dirs)} batch folders\n")
    
    success_count = 0
    for batch_dir in batch_dirs:
        if upload_batch_to_hf(batch_dir):
            success_count += 1
        print()
    
    print(f"=== Summary ===")
    print(f"Successfully uploaded: {success_count}/{len(batch_dirs)} batches")
    
    if success_count == len(batch_dirs):
        print(f"\nAll batches uploaded to: Vu165/lastdance-asr")
        print(f"Audio organized as: asr/audio/batch-01/ ... asr/audio/batch-09/")

if __name__ == "__main__":
    main()
