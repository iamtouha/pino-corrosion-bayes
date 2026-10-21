#!/usr/bin/env bash
# Run the whole pipeline inside the container, stage by stage, logging to logs/<cfg>/.
# Usage (host):  ./docker/run.sh bash scripts/pipeline.sh laptop.yaml [stage ...]
# Stages: verify bending dataset vdic tests study bench nuts zenodo capacity   (default: all)
# Every stage is resumable; rerunning skips finished work.
set -uo pipefail
CFG="${1:-laptop.yaml}"; shift || true
STAGES=("$@"); [[ ${#STAGES[@]} -eq 0 ]] && STAGES=(verify bending dataset vdic tests study bench nuts zenodo capacity)
TAG="$(basename "$CFG" .yaml)"
LOG="logs/$TAG"; mkdir -p "$LOG"
run() {  # name, command...
  local name="$1"; shift
  echo "=== [$(date '+%F %T')] $name: $*" | tee -a "$LOG/pipeline.log"
  "$@" > "$LOG/$name.log" 2>&1
  local rc=$?
  if [[ $rc -eq 0 ]]; then
    echo "=== [$(date '+%F %T')] $name OK" | tee -a "$LOG/pipeline.log"
  else
    echo "=== [$(date '+%F %T')] $name FAILED (exit $rc), see $LOG/$name.log" | tee -a "$LOG/pipeline.log"
  fi
}
BEND_ARGS=(); [[ "$TAG" == "quick" ]] && BEND_ARGS=(--quick)
for s in "${STAGES[@]}"; do
  case "$s" in
    verify)   run step1_verify   python3 experiments/step1_verify.py --config "$CFG" ;;
    bending)  run step1b_bending python3 experiments/step1b_bending.py --config "$CFG" "${BEND_ARGS[@]}" ;;
    dataset)  run step2_dataset  python3 experiments/step2_dataset.py --config "$CFG" ;;
    vdic)     run step3_vdic     python3 experiments/step3_vdic.py --config "$CFG" ;;
    tests)    run tests          python3 -m pytest -q tests -p no:cacheprovider ;;
    study)    run step5_study    python3 experiments/step5_study.py --config "$CFG" ;;
    bench)    run step7_bench    python3 experiments/step7_benchmark.py --config "$CFG" ;;
    nuts)     run step6_nuts     python3 experiments/step6_nuts.py --config "$CFG" ;;
    zenodo)   run step8a_zenodo  python3 experiments/step8a_zenodo.py --config "$CFG" ;;
    capacity) run step8c_capacity python3 experiments/step8c_capacity.py --config "$CFG" ;;
    *) echo "unknown stage $s" ;;
  esac
done
echo "=== [$(date '+%F %T')] pipeline finished" | tee -a "$LOG/pipeline.log"
