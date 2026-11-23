#!/usr/bin/env bash
# Start an improvement job (follow-up studies, README) in a detached container pino-improve-<task>.
#   ./scripts/launch_improve.sh t1 python3 experiments/improve_t1_particles.py --config laptop.yaml --config improve.yaml
# Log: logs/laptop/improve_<task>.log     Stop: docker stop pino-improve-<task>
# Only the container with exactly this name is replaced; nothing else is touched.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TASK="$1"; shift
NAME="pino-improve-${TASK}"
LOG="logs/laptop/improve_${TASK}.log"
GPU_ARGS=()
if [[ "${GPU_MODE:-auto}" == "nvidia" ]]; then GPU_ARGS=(--gpus all)
elif [[ -e /dev/dxg && "${GPU_MODE:-auto}" != "none" ]]; then GPU_ARGS=(--device /dev/dxg -v /usr/lib/wsl:/usr/lib/wsl:ro); fi
if docker inspect "$NAME" >/dev/null 2>&1; then
  if [[ "$(docker inspect -f '{{.State.Running}}' "$NAME")" == "true" ]]; then
    echo "$NAME is still running; stop it first (docker stop $NAME)" >&2; exit 1
  fi
  docker rm "$NAME" >/dev/null
fi
mkdir -p "$ROOT/logs/laptop"
docker run -d --name "$NAME" "${GPU_ARGS[@]}" --shm-size=2g \
  --user "$(id -u):$(id -g)" -e HOME=/tmp -e PYTHONUNBUFFERED=1 -v "$ROOT":/work -w /work \
  pino-surrogate:latest bash -c "echo \"=== [\$(date '+%F %T')] start: $*\" >> $LOG; $* >> $LOG 2>&1; rc=\$?; echo \"=== [\$(date '+%F %T')] exit \$rc\" >> $LOG"
echo "started $NAME; log: $LOG"
