#!/usr/bin/env bash
set -euo pipefail

# Publish CoreML models to a standalone Git repo with Git LFS
# Usage:
#   ./scripts/publish/publish_coreml_repo.sh <target_dir> <remote_git_url>
# Example:
#   ./scripts/publish/publish_coreml_repo.sh ./publish/qwen-typo-fixer-coreml git@github.com:mazhewitt/qwen-typo-fixer-coreml.git

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <target_dir> <remote_git_url>" >&2
  exit 1
fi

TARGET_DIR="$1"
REMOTE_URL="$2"

SRC_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
MODELS_DIR="$SRC_DIR/models/qwen-typo-fixer-ane"

if [[ ! -d "$MODELS_DIR" ]]; then
  echo "Models directory not found: $MODELS_DIR" >&2
  exit 1
fi

mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR"

# Init repo
if [[ ! -d .git ]]; then
  git init
fi

# Git LFS
if ! command -v git-lfs >/dev/null 2>&1; then
  echo "git-lfs not found. Install with: brew install git-lfs" >&2
  exit 1
fi

git lfs install

cat > .gitattributes <<'EOF'
# Track all CoreML package contents with LFS
*.mlpackage/** filter=lfs diff=lfs merge=lfs -text
*.mlmodelc/** filter=lfs diff=lfs merge=lfs -text
*.mlmodel filter=lfs diff=lfs merge=lfs -text
EOF

echo "# Qwen Typo Fixer — CoreML (ANE) Models" > README.md
printf "\nThis repository contains CoreML artifacts for the Qwen typo-fixer.\n\n" >> README.md

# Copy artifacts
rsync -av --delete \
  --include '*/' \
  --include '*.mlpackage/***' \
  --include 'meta.yaml' \
  --include 'tokenizer.json' \
  --include 'tokenizer_config.json' \
  --include 'vocab.json' \
  --include 'merges.txt' \
  --exclude '*' \
  "$MODELS_DIR"/ ./

# Commit & push
git add .gitattributes README.md *.mlpackage meta.yaml tokenizer.json tokenizer_config.json vocab.json merges.txt 2>/dev/null || true

if ! git diff --cached --quiet; then
  git commit -m "Add CoreML models and tokenizer"
fi

git branch -M main || true

git remote remove origin 2>/dev/null || true

git remote add origin "$REMOTE_URL"

git push -u origin main

echo "Done. Pushed to $REMOTE_URL"
