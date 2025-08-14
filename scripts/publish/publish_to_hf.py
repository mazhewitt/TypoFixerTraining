#!/usr/bin/env python3
"""Publish the CoreML models to Hugging Face Hub as a standalone repo.

Requires:
  pip install huggingface_hub
  export HUGGINGFACE_HUB_TOKEN=...  (write token)

Usage:
  python scripts/publish/publish_to_hf.py --repo-id <namespace/name> \
      --source models/qwen-typo-fixer-ane --private false --create
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder, HfFolder


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", required=True, help="e.g. mazhewitt/qwen-typo-fixer-coreml")
    p.add_argument("--source", default="models/qwen-typo-fixer-ane", help="Folder with .mlpackage & tokenizer")
    p.add_argument("--private", default="false", choices=["true", "false"], help="Create as private repo")
    p.add_argument("--create", action="store_true", help="Create repo if missing")
    p.add_argument("--hf-mirror", default=None, help="Optional alternative hub URL")
    args = p.parse_args()

    source = Path(args.source).resolve()
    if not source.exists():
        raise SystemExit(f"Source not found: {source}")

    # Use cached token from `huggingface-cli login` if available; env var optional
    token = HfFolder.get_token()

    api = HfApi(endpoint=args.hf_mirror) if args.hf_mirror else HfApi()

    if args.create:
        create_repo(args.repo_id, private=(args.private == "true"), exist_ok=True, token=token)

    # Upload folder; preserves structure and large files
    upload_folder(
        repo_id=args.repo_id,
        folder_path=str(source),
        path_in_repo=".",
        repo_type="model",
    token=token,
        commit_message="Add CoreML models and tokenizer",
        ignore_patterns=["*.DS_Store", "__pycache__/*"],
    )

    print(f"Uploaded {source} to hf://{args.repo_id}")


if __name__ == "__main__":
    main()
