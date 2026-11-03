"""
src/data/download_stocknet.py
Automated acquisition of the official StockNet dataset from GitHub.
Clones or downloads repository into data/raw/stocknet-dataset.
"""
import os
import subprocess
import sys
import shutil
import urllib.request
import zipfile
from pathlib import Path

STOCKNET_REPO_URL = "https://github.com/yumoxu/stocknet-dataset.git"
STOCKNET_ZIP_URL = "https://github.com/yumoxu/stocknet-dataset/archive/refs/heads/master.zip"
TARGET_DIR = Path("data/raw/stocknet-dataset")

def acquire_stocknet(target_dir: Path = TARGET_DIR):
    target_dir = Path(target_dir)
    target_dir.parent.mkdir(parents=True, exist_ok=True)

    # Check if already present and has expected folders
    price_dir = target_dir / "price" / "preprocessed"
    tweet_dir = target_dir / "tweet" / "preprocessed"
    if price_dir.exists() and tweet_dir.exists():
        print(f"[StockNet] Found existing dataset at {target_dir}")
        price_files = list(price_dir.glob("*.txt"))
        tweet_stocks = [d for d in tweet_dir.iterdir() if d.is_dir()]
        print(f"[StockNet] Price preprocessed files: {len(price_files)}, Tweet stock directories: {len(tweet_stocks)}")
        return True

    print(f"[StockNet] Downloading official dataset into {target_dir}...")

    # Method 1: Git clone with depth 1
    git_bin = shutil.which("git")
    if git_bin:
        print("[StockNet] Attempting shallow git clone (--depth 1)...")
        cmd = [git_bin, "clone", "--depth", "1", STOCKNET_REPO_URL, str(target_dir)]
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("[StockNet] Git clone completed successfully.")
            return True
        except subprocess.CalledProcessError as e:
            print(f"[StockNet] Git clone failed: {e.stderr}. Falling back to ZIP download...")

    # Method 2: Fallback ZIP download
    zip_path = target_dir.parent / "stocknet-dataset-master.zip"
    print(f"[StockNet] Downloading zip archive from {STOCKNET_ZIP_URL}...")
    try:
        urllib.request.urlretrieve(STOCKNET_ZIP_URL, zip_path)
        print(f"[StockNet] Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir.parent)
        
        extracted_folder = target_dir.parent / "stocknet-dataset-master"
        if extracted_folder.exists():
            if target_dir.exists():
                shutil.rmtree(target_dir)
            extracted_folder.rename(target_dir)
        
        if zip_path.exists():
            zip_path.unlink()
        print("[StockNet] Zip extraction complete.")
        return True
    except Exception as e:
        print(f"[StockNet] Failed to download zip: {e}")
        return False

if __name__ == "__main__":
    success = acquire_stocknet()
    sys.exit(0 if success else 1)
