#!/usr/bin/env bash
# Run a command inside the project container with the repo mounted at /work.
# GPU: on WSL2 (no nvidia-container-toolkit needed) we pass /dev/dxg + the WSL driver libs.
# On a normal Linux host with nvidia-container-toolkit, set GPU_MODE=nvidia.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="${IMAGE:-pino-surrogate:latest}"
GPU_ARGS=()
if [[ "${GPU_MODE:-auto}" == "nvidia" ]]; then
  GPU_ARGS=(--gpus all)
elif [[ -e /dev/dxg && "${GPU_MODE:-auto}" != "none" ]]; then
  GPU_ARGS=(--device /dev/dxg -v /usr/lib/wsl:/usr/lib/wsl:ro)
fi
TTY=(); [[ -t 0 && -t 1 ]] && TTY=(-it)
exec docker run --rm "${TTY[@]}" "${GPU_ARGS[@]}" \
  --shm-size=2g --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$ROOT":/work -w /work "$IMAGE" "$@"
