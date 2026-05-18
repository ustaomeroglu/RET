#!/usr/bin/env python
"""
Download SYCON-Bench data files from GitHub into data/sycophancy/.

Usage:
    python scripts/sycophancy/prepare_data.py [--data_dir data/sycophancy]
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile


REPO_URL = "https://github.com/JiseungHong/SYCON-Bench.git"

# Map repo paths to local paths
FILE_MAP = {
    # Debate
    "debate_setting/data/questions.txt": "debate/questions.txt",
    "debate_setting/data/arguments.txt": "debate/arguments.txt",
    # False presupposition
    "false-presuppositions-setting/data/questions.txt": "false_presupposition/questions.txt",
    "false-presuppositions-setting/data/presuppositions.txt": "false_presupposition/presuppositions.txt",
    "false-presuppositions-setting/data/corrections.txt": "false_presupposition/corrections.txt",
    "false-presuppositions-setting/data/push_back.csv": "false_presupposition/push_back.csv",
}


def main():
    parser = argparse.ArgumentParser(description="Download SYCON-Bench data")
    parser.add_argument("--data_dir", default="data/sycophancy", help="Output directory")
    args = parser.parse_args()

    data_dir = args.data_dir

    # Check if data already exists
    existing = sum(1 for v in FILE_MAP.values() if os.path.exists(os.path.join(data_dir, v)))
    if existing == len(FILE_MAP):
        print(f"All {existing} data files already exist in {data_dir}/. Skipping download.")
        return

    # Clone repo to temp dir (shallow)
    with tempfile.TemporaryDirectory() as tmpdir:
        clone_dir = os.path.join(tmpdir, "SYCON-Bench")
        print(f"Cloning {REPO_URL} (shallow)...")
        subprocess.run(
            ["git", "clone", "--depth", "1", REPO_URL, clone_dir],
            check=True,
            capture_output=True,
        )

        # Copy files
        for src_rel, dst_rel in FILE_MAP.items():
            src = os.path.join(clone_dir, src_rel)
            dst = os.path.join(data_dir, dst_rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if os.path.exists(src):
                shutil.copy2(src, dst)
                print(f"  {dst_rel}")
            else:
                print(f"  WARNING: {src_rel} not found in repo", file=sys.stderr)

    print(f"\nDone. Data files saved to {data_dir}/")


if __name__ == "__main__":
    main()
