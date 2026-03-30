#!/usr/bin/env bash
# Remove oversized sentiment artifacts from Git history so GitHub accepts push.
# Run from repo root: bash scripts/strip_large_files_from_git_history.sh
#
# Keeps files on disk; only rewrites Git history. Afterward:
#   git add .gitignore
#   git commit -m "chore: ignore large sentiment model weights and outputs"   # if needed
#   git push --force-with-lease origin "$(git branch --show-current)"
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BRANCH="$(git branch --show-current)"

if command -v git-filter-repo >/dev/null 2>&1; then
  echo "Using git-filter-repo (only rewriting refs/heads/$BRANCH)..."
  # git-filter-repo removes 'origin' — re-add with: git remote add origin <your-url>
  git filter-repo --force --refs "refs/heads/$BRANCH" \
    --path "3a_Sentiment_Analysis/models/final/german-nlp-group/electra-base-german-uncased/pytorch_model.bin" --invert-paths \
    --path "3a_Sentiment_Analysis/outputs/emotion_full_results.csv" --invert-paths \
    --path "Sentiment_Analysis/models/final/german-nlp-group/electra-base-german-uncased/pytorch_model.bin" --invert-paths \
    --path "Sentiment_Analysis/outputs/emotion_full_results.csv" --invert-paths
else
  echo "git-filter-repo not found. Install:  brew install git-filter-repo   or   pip install git-filter-repo"
  echo "Falling back to git filter-branch (slower)..."
  git filter-branch -f --index-filter \
    'git rm --cached --ignore-unmatch -f \
      3a_Sentiment_Analysis/models/final/german-nlp-group/electra-base-german-uncased/pytorch_model.bin \
      3a_Sentiment_Analysis/outputs/emotion_full_results.csv \
      Sentiment_Analysis/models/final/german-nlp-group/electra-base-german-uncased/pytorch_model.bin \
      Sentiment_Analysis/outputs/emotion_full_results.csv' \
    --prune-empty "$BRANCH"
fi

git reflog expire --expire=now --all
git gc --prune=now --aggressive
echo "Done. Verify with: git log --oneline -3"
if ! git remote get-url origin >/dev/null 2>&1; then
  echo "No 'origin' remote (git-filter-repo removes it). Run: git remote add origin <repo-url>"
fi
echo "Then: git push --force-with-lease origin $BRANCH"
