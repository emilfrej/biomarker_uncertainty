#!/usr/bin/env bash
# Clone (or update) external repos this project depends on into the project root.
# Add one line per repo: "<github url> [folder name]". The folder defaults to the repo name.
set -euo pipefail

REPOS=(
  "https://github.com/KCGallagher/AT_Model_Comparison"
  "https://github.com/MathOnco/AT_costOfResistance_LVModel"
  "https://github.com/ap-browning/robust-adaptive-therapy"
)

cd "$(dirname "$0")"

for entry in "${REPOS[@]}"; do
  read -r url dir <<< "$entry"
  dir="${dir:-$(basename "$url" .git)}"
  if [ -d "$dir/.git" ]; then
    echo "updating $dir"
    git -C "$dir" pull --ff-only
  else
    echo "cloning $url -> $dir"
    git clone "$url" "$dir"
  fi
done
