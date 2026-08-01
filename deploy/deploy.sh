#!/usr/bin/env bash
# Publishes the project to GitHub. macOS / Linux.
#   ./deploy/deploy.sh <github-username> [repo] [branch]
set -euo pipefail
cd "$(dirname "$0")/.."

USER_NAME="${1:?Usage: ./deploy/deploy.sh <github-username> [repo] [branch]}"
REPO="${2:-quant-backtest-studio}"
BRANCH="${3:-main}"

command -v git >/dev/null || { echo "Git was not found."; exit 1; }
[ -f .streamlit/secrets.toml ] && echo "secrets.toml detected: ignored by .gitignore, not published."

[ -d .git ] || { git init -q; git branch -M "$BRANCH"; }
git add -A
git diff --cached --quiet && echo "Nothing new to publish." || git commit -q -m "Quant Backtest Studio"

URL="https://github.com/$USER_NAME/$REPO.git"
git remote get-url origin >/dev/null 2>&1 && git remote set-url origin "$URL" || git remote add origin "$URL"

echo "Pushing to $URL ..."
git push -u origin "$BRANCH"

cat <<MSG

Repository published. Three clicks left:
  1. Open https://share.streamlit.io and sign in with GitHub
  2. New app -> repo '$REPO', branch '$BRANCH', file 'app.py'
  3. Deploy

Password (optional): Settings -> Secrets -> password = "..."
MSG
