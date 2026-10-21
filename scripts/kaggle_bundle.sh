#!/usr/bin/env bash
# Build the folder to upload as a Kaggle dataset for running Step 5 (training) on Kaggle.
#   ./scripts/kaggle_bundle.sh [laptop.yaml]
# Output: kaggle/upload/  (~2.3 GB) with
#   A.h5 B.h5 C.h5  - training / test sets (hard links, no extra disk)
#   U.h5            - unlabelled set, parameters only (the sampler regenerates the fields)
#   vdic/*.json     - filter width etc. (step 5 evaluates after H)
#   pino-code.tar.gz - pino/, experiments/step5_study.py, configs/
#   step5_prev.tar.gz - finished step-5 results, if any (resume)
# Then upload kaggle/upload as a new dataset (web UI, or `kaggle datasets create -p kaggle/upload
# --dir-mode tar` after editing dataset-metadata.json) and attach it to kaggle/pino_step5.ipynb.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
CFG="${1:-laptop.yaml}"; TAG="$(basename "$CFG" .yaml)"
OUT="${OUT:-kaggle/upload}"
DATA="$(python3 - "$CFG" <<'PY'
import sys, yaml
d = "data"
for f in ("configs/base.yaml", "configs/" + sys.argv[1]):
    d = ((yaml.safe_load(open(f)) or {}).get("dataset") or {}).get("dir", d)
print(d)
PY
)"
echo "config $CFG, data in $DATA"
rm -rf "$OUT"; mkdir -p "$OUT/vdic"
for s in A B C; do ln "$DATA/$s.h5" "$OUT/$s.h5" 2>/dev/null || cp "$DATA/$s.h5" "$OUT/$s.h5"; done
cp "$DATA"/vdic/*.json "$OUT/vdic/"
./docker/run.sh python3 -c '
import sys, h5py
with h5py.File(sys.argv[1], "r") as f, h5py.File(sys.argv[2], "w") as g:
    for k in f:
        if k != "t_tilde":
            g.create_dataset(k, data=f[k][:])
    g.attrs.update(dict(f.attrs))
print("slim U.h5 written")' "$DATA/U.h5" "$OUT/U.h5"
test -s "$OUT/U.h5"
tar czf "$OUT/pino-code.tar.gz" --exclude='__pycache__' pino experiments/step5_study.py configs
if compgen -G "results/$TAG/step5/runs/*.json" > /dev/null; then
  tar czf "$OUT/step5_prev.tar.gz" "results/$TAG/step5"
  echo "included $(ls results/$TAG/step5/runs/*.json | wc -l) finished runs"
fi
cat > "$OUT/dataset-metadata.json" <<EOF
{"title": "pino-step5", "id": "YOUR_KAGGLE_USERNAME/pino-step5", "licenses": [{"name": "other"}]}
EOF
du -shL "$OUT"; ls -la "$OUT"
