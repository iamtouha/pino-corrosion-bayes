#!/usr/bin/env bash
# Build kaggle/upload_t3/ (~0.6 GB) for the surrogate-only part of T3 (follow-up studies, README) on Kaggle:
#   D_pit.h5, step-7 data.npz, the four PINO checkpoints, vdic/*.json, code, finished T3 posteriors.
# Upload the folder as a dataset "pino-t3", attach it to kaggle/pino_t3.ipynb (GPU T4 x2).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
OUT="${OUT:-kaggle/upload_t3}"
rm -rf "$OUT"; mkdir -p "$OUT/vdic" "$OUT/ckpt"
ln data/D_pit.h5 "$OUT/D_pit.h5" 2>/dev/null || cp data/D_pit.h5 "$OUT/D_pit.h5"
cp results/laptop/step7/pit/data.npz "$OUT/data.npz"
for f in pino_A_N100_s0 pino_A_N2000_s0 pino_B_N100_s0 pino_B_N2000_s0; do cp "results/laptop/step5/ckpt/$f.pt" "$OUT/ckpt/"; done
cp data/vdic/*.json "$OUT/vdic/"
tar czf "$OUT/pino-code.tar.gz" --exclude='__pycache__' pino experiments/__init__.py experiments/step7_benchmark.py \
  experiments/improve_t3_regime.py configs
if compgen -G "results/laptop/improve/t3/posteriors/*.npz" > /dev/null; then
  tar czf "$OUT/t3_prev.tar.gz" results/laptop/improve/t3
  echo "included $(ls results/laptop/improve/t3/posteriors/*.npz | wc -l) finished posteriors"
fi
cat > "$OUT/dataset-metadata.json" <<EOT
{"title": "pino-t3", "id": "YOUR_KAGGLE_USERNAME/pino-t3", "licenses": [{"name": "other"}]}
EOT
du -shL "$OUT"; ls -la "$OUT"
