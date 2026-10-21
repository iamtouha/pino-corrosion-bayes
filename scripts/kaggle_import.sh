#!/usr/bin/env bash
# Merge Step-5 results from a Kaggle notebook version into this repo.
#   ./scripts/kaggle_import.sh <downloaded output .zip | extracted folder> [laptop.yaml]
# Finished runs are copied without overwriting local files. When all 36 runs (+ lam_r
# selections) are present, continue locally with:
#   ./scripts/launch_detached.sh laptop.yaml study bench nuts zenodo capacity
# (`study` then only rebuilds summary.md, since every run is already finished.)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
IN="${1:?path to the Kaggle output zip or folder}"; CFG="${2:-laptop.yaml}"; TAG="$(basename "$CFG" .yaml)"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
if [[ -f "$IN" ]]; then
  python3 -c "import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$IN" "$TMP"
  SRC="$TMP"
else
  SRC="$IN"
fi
D="$(find "$SRC" -type d -path "*results/$TAG/step5" | head -1)"
[[ -n "$D" ]] || { echo "no results/$TAG/step5 in $IN"; exit 1; }
mkdir -p "results/$TAG/step5"
cp -rn "$D/." "results/$TAG/step5/"
if [[ -d "$(dirname "$(dirname "$(dirname "$D")")")/logs" ]]; then
  mkdir -p "logs/$TAG/kaggle"; cp -n "$(dirname "$(dirname "$(dirname "$D")")")"/logs/* "logs/$TAG/kaggle/" 2>/dev/null || true
fi
n=$(ls "results/$TAG/step5/runs/"*.json 2>/dev/null | wc -l)
echo "results/$TAG/step5 now has $n finished runs:"
ls "results/$TAG/step5/runs/" | sed 's/\.json$//' | tr '\n' ' '; echo
