#!/usr/bin/env python3
"""
upload_to_hf.py

Simple helper to convert a CSV to a Hugging Face dataset and optionally push it to the Hub.

Usage examples:
  Dry-run (no push):
    python upload_to_hf.py --csv ulasan_google_play.csv --dry-run

  Push to Hugging Face (requires token via --token or HF_TOKEN env var):
    python upload_to_hf.py --csv ulasan_google_play.csv --repo-id username/ulasan-google-play --token <HF_TOKEN>

The script saves a local copy when dry-run is used.
"""
import argparse
import os
import sys
import pandas as pd
from datasets import Dataset


def parse_args():
    p = argparse.ArgumentParser(description="Convert CSV to Hugging Face dataset and push to Hub")
    p.add_argument("--csv", required=True, help="Path to CSV file")
    p.add_argument("--repo-id", help="Hugging Face repo id (e.g. username/dataset-name)")
    p.add_argument("--token", help="Hugging Face token (or set HF_TOKEN env var)")
    p.add_argument("--private", action="store_true", help="Create dataset as private")
    p.add_argument("--dry-run", action="store_true", dest="dry_run", help="Do not push to hub; save locally")
    p.add_argument("--out", help="Output folder for dry-run (default: dataset_local)")
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.csv):
        print(f"CSV not found: {args.csv}")
        sys.exit(1)

    df = pd.read_csv(args.csv)
    print(f"Loaded CSV: {args.csv} — {len(df)} rows")
    print("Columns:", df.columns.tolist())
    print("Sample rows:\n", df.head().to_string())

    # Convert to Hugging Face Dataset
    df = df.reset_index(drop=True)
    ds = Dataset.from_pandas(df)

    if args.dry_run:
        out = args.out or "dataset_local"
        print(f"Dry-run: saving dataset to disk at '{out}'")
        ds.save_to_disk(out)
        print("Done.")
        return

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        print("Error: no Hugging Face token provided. Use --token or set HF_TOKEN environment variable.")
        sys.exit(1)

    repo_id = args.repo_id
    if not repo_id:
        repo_id = input("Enter Hugging Face repo id (username/repo): ").strip()
        if not repo_id:
            print("Error: repo id required to push")
            sys.exit(1)

    print(f"Pushing dataset to Hub repo: {repo_id} (private={args.private})")
    ds.push_to_hub(repo_id=repo_id, token=token, private=args.private)
    print("Push complete.")


if __name__ == "__main__":
    main()
