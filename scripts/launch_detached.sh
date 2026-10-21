#!/usr/bin/env bash
# Start the pipeline in a detached container that survives closing the terminal.
#   ./scripts/launch_detached.sh laptop.yaml            # all stages
#   ./scripts/launch_detached.sh laptop.yaml study bench
# Follow:  tail -f logs/laptop/pipeline.log      Stop:  docker stop pino-<cfg>
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CFG="${1:-laptop.yaml}"; shift || true
NAME="pino-$(basename "$CFG" .yaml)"
GPU_ARGS=()
if [[ "${GPU_MODE:-auto}" == "nvidia" ]]; then GPU_ARGS=(--gpus all)
elif [[ -e /dev/dxg ]]; then GPU_ARGS=(--device /dev/dxg -v /usr/lib/wsl:/usr/lib/wsl:ro); fi
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" "${GPU_ARGS[@]}" --shm-size=2g \
  --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$ROOT":/work -w /work \
  pino-surrogate:latest bash scripts/pipeline.sh "$CFG" "$@"
echo "started container $NAME; logs in logs/$(basename "$CFG" .yaml)/"
