#!/usr/bin/env bash
set -u

# Single-window Bench2Drive closed-loop runner for the DiffusionDrive
# baseline-basic checkpoint. It is designed for Slurm jobs where tmux and
# attaching a second shell to the allocation are unavailable.

START_IDX=${START_IDX:-0}
END_IDX=${END_IDX:-219}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-2}
CARLA_START_WAIT=${CARLA_START_WAIT:-25}
CARLA_READY_TIMEOUT=${CARLA_READY_TIMEOUT:-120}
CARLA_READY_INTERVAL=${CARLA_READY_INTERVAL:-5}
ROUTE_CLEANUP_WAIT=${ROUTE_CLEANUP_WAIT:-10}
EVAL_TIMEOUT=${EVAL_TIMEOUT:-300}
PYTHON_BIN=${PYTHON_BIN:-python}

CARLA_ROOT=${CARLA_ROOT:-/share/home/u19666033/syb/carla_0_9_15}
WORK_DIR=${WORK_DIR:-/share/home/u19666033/ltr/carla_garage}
SCENARIO_RUNNER_ROOT=${SCENARIO_RUNNER_ROOT:-${WORK_DIR}/scenario_runner}
LEADERBOARD_ROOT=${LEADERBOARD_ROOT:-${WORK_DIR}/leaderboard}

RUN_ID=${RUN_ID:-origlike_ddp4_bs64x4_lr6e-4_ep100_fs5_spatial_imgenc0p5}
OUT=${OUT:-/share/home/u19666033/ltr/dd_logs/full_baseline_basic/${RUN_ID}/closed_loop_bench2drive}
CKPT=${CKPT:-/share/home/u19666033/ltr/dd_logs/full_baseline_basic/${RUN_ID}/latest.pth}

ANCHOR_PATH=${ANCHOR_PATH:-/share/home/u19666033/ltr/4-0-0-1910-tracked_clusters_anchor.npy}
BACKBONE_PATH=${BACKBONE_PATH:-/share/home/u19666033/ltr/pytorch_model.bin}

ROUTE_DIR=${ROUTE_DIR:-${WORK_DIR}/leaderboard/data/bench2drive_split}
AGENT_PATH=${AGENT_PATH:-${WORK_DIR}/team_code/diffusiondrive_agent.py}
AGENT_CONFIG=${AGENT_CONFIG:-${WORK_DIR}/team_code/diffusiondrive_agent.py}

PORT_BASE=${PORT_BASE:-2050}
STREAMING_PORT_BASE=${STREAMING_PORT_BASE:-2150}
TM_PORT_BASE=${TM_PORT_BASE:-8050}
ATTEMPT_PORT_STRIDE=${ATTEMPT_PORT_STRIDE:-100}

export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla:${SCENARIO_RUNNER_ROOT}:${LEADERBOARD_ROOT}:${PYTHONPATH:-}"

export DIFFUSIONDRIVE_CHECKPOINT=${DIFFUSIONDRIVE_CHECKPOINT:-${CKPT}}
export DIFFUSIONDRIVE_ANCHOR_PATH=${DIFFUSIONDRIVE_ANCHOR_PATH:-${ANCHOR_PATH}}
export DIFFUSIONDRIVE_BACKBONE_PATH=${DIFFUSIONDRIVE_BACKBONE_PATH:-${BACKBONE_PATH}}
export STOP_CONTROL=${STOP_CONTROL:-0}
export DIFFUSIONDRIVE_SPATIAL_PID=${DIFFUSIONDRIVE_SPATIAL_PID:-1}
export DIFFUSIONDRIVE_SPATIAL_PID_SPEED_FAST=${DIFFUSIONDRIVE_SPATIAL_PID_SPEED_FAST:-}
export DIFFUSIONDRIVE_SPATIAL_PID_SPEED_SLOW=${DIFFUSIONDRIVE_SPATIAL_PID_SPEED_SLOW:-}
export DIFFUSIONDRIVE_SPATIAL_PID_TURN_THRESHOLD=${DIFFUSIONDRIVE_SPATIAL_PID_TURN_THRESHOLD:-}
export DIFFUSIONDRIVE_SPATIAL_PID_SHARP_TURN_THRESHOLD=${DIFFUSIONDRIVE_SPATIAL_PID_SHARP_TURN_THRESHOLD:-}
export DIFFUSIONDRIVE_COMMAND_DELAY=${DIFFUSIONDRIVE_COMMAND_DELAY:-0}
export DIFFUSIONDRIVE_LOW_SPEED_STEER=${DIFFUSIONDRIVE_LOW_SPEED_STEER:-0}
export DIFFUSIONDRIVE_JPEG_ARTIFACT=${DIFFUSIONDRIVE_JPEG_ARTIFACT:-1}
export DIFFUSIONDRIVE_IMAGE_NORMALIZATION=${DIFFUSIONDRIVE_IMAGE_NORMALIZATION:-none}

mkdir -p "${OUT}/results" "${OUT}/logs" "${OUT}/carla_logs" "${OUT}/monitor"

result_is_complete() {
  "${PYTHON_BIN}" - "$1" <<'PY'
import json
import sys

path = sys.argv[1]
try:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
except Exception:
    sys.exit(1)

checkpoint = data.get("_checkpoint", {})
progress = checkpoint.get("progress", [])
records = checkpoint.get("records", [])

if len(progress) < 2 or progress[0] < progress[1]:
    sys.exit(1)
if not records:
    sys.exit(1)

rerun_status = {
    "Failed - Agent couldn't be set up",
    "Failed - Simulation crashed",
    "Failed - Agent crashed",
}

for record in records:
    if record.get("status", "") in rerun_status:
        sys.exit(1)

sys.exit(0)
PY
}

wait_for_carla_ready() {
  "${PYTHON_BIN}" - "$1" "$2" "$3" <<'PY'
import sys
import time

import carla

port = int(sys.argv[1])
timeout = float(sys.argv[2])
interval = float(sys.argv[3])
deadline = time.time() + timeout
last_error = None

while time.time() < deadline:
    try:
        client = carla.Client("localhost", port)
        client.set_timeout(min(5.0, max(1.0, interval)))
        world = client.get_world()
        settings = carla.WorldSettings(
            synchronous_mode=True,
            fixed_delta_seconds=0.05,
            deterministic_ragdolls=True,
            spectator_as_ego=False,
        )
        world.apply_settings(settings)
        print(f"CARLA RPC ready on port {port}", flush=True)
        sys.exit(0)
    except Exception as exc:
        last_error = exc
        print(f"Waiting for CARLA RPC on port {port}: {exc}", flush=True)
        time.sleep(interval)

print(f"CARLA RPC not ready on port {port} after {timeout}s: {last_error}", flush=True)
sys.exit(1)
PY
}

cleanup_route_processes() {
  if [ "${EVAL_PID:-}" != "" ]; then
    kill -TERM -"${EVAL_PID}" 2>/dev/null || true
    sleep 2
    kill -KILL -"${EVAL_PID}" 2>/dev/null || true
    wait "${EVAL_PID}" 2>/dev/null || true
    EVAL_PID=""
  fi

  if [ "${CARLA_PID:-}" != "" ]; then
    kill -TERM -"${CARLA_PID}" 2>/dev/null || true
    sleep 5
    kill -KILL -"${CARLA_PID}" 2>/dev/null || true
    wait "${CARLA_PID}" 2>/dev/null || true
    CARLA_PID=""
  fi

  pkill -9 -f "CarlaUE4-Linux-Shipping" 2>/dev/null || true
  pkill -9 -f "CarlaUE4.sh" 2>/dev/null || true
}

start_monitor() {
  (
    while true; do
      echo "===== $(date) host=$(hostname) ====="
      nvidia-smi || true
      echo
      ps -u "$USER" -o pid,ppid,stat,etime,rss,cmd \
        | grep -E 'CarlaUE4|CarlaUE4-Linux-Shipping|leaderboard_evaluator_local|python' \
        | grep -v grep || true
      echo
      free -h || true
      echo
      sleep 10
    done
  ) > "${OUT}/monitor/node_monitor.log" 2>&1 &
  MONITOR_PID=$!
}

stop_all() {
  cleanup_route_processes
  if [ "${MONITOR_PID:-}" != "" ]; then
    kill "${MONITOR_PID}" 2>/dev/null || true
  fi
}

trap stop_all EXIT

echo "Closed-loop output: ${OUT}"
echo "Checkpoint: ${DIFFUSIONDRIVE_CHECKPOINT}"
echo "Routes: ${START_IDX}..${END_IDX}"
echo "Max attempts per route: ${MAX_ATTEMPTS}"
echo "CARLA readiness timeout: ${CARLA_READY_TIMEOUT}s"

start_monitor
cd "${WORK_DIR}" || exit 1

for IDX in $(seq "${START_IDX}" "${END_IDX}"); do
  ROUTE_NAME=$(printf "bench2drive_%02d" "${IDX}")
  ROUTE_FILE="${ROUTE_DIR}/${ROUTE_NAME}.xml"
  RESULT_JSON="${OUT}/results/${ROUTE_NAME}.json"
  LIVE_TXT="${OUT}/results/${ROUTE_NAME}_live.txt"

  if [ ! -f "${ROUTE_FILE}" ]; then
    echo "Missing route file: ${ROUTE_FILE}"
    continue
  fi

  if [ -f "${RESULT_JSON}" ]; then
    if result_is_complete "${RESULT_JSON}"; then
      echo "Skip finished ${ROUTE_NAME}: ${RESULT_JSON}"
      continue
    fi

    echo "Remove incomplete/crashed result: ${RESULT_JSON}"
    rm -f "${RESULT_JSON}" "${LIVE_TXT}"
  fi

  ATTEMPT=1
  while [ "${ATTEMPT}" -le "${MAX_ATTEMPTS}" ]; do
    PORT=$((PORT_BASE + IDX + ATTEMPT * ATTEMPT_PORT_STRIDE))
    STREAMING_PORT=$((STREAMING_PORT_BASE + IDX + ATTEMPT * ATTEMPT_PORT_STRIDE))
    TM_PORT=$((TM_PORT_BASE + IDX + ATTEMPT * ATTEMPT_PORT_STRIDE))

    CARLA_LOG="${OUT}/carla_logs/${ROUTE_NAME}_attempt${ATTEMPT}_carla.log"
    EVAL_LOG="${OUT}/logs/${ROUTE_NAME}_attempt${ATTEMPT}_eval.log"

    echo "===== ${ROUTE_NAME} attempt=${ATTEMPT} port=${PORT} streaming=${STREAMING_PORT} tm=${TM_PORT} ====="

    cleanup_route_processes
    sleep "${ROUTE_CLEANUP_WAIT}"

    setsid "${CARLA_ROOT}/CarlaUE4.sh" \
      -carla-rpc-port="${PORT}" \
      -carla-streaming-port="${STREAMING_PORT}" \
      -nosound -RenderOffScreen -carla-primary-port=0 -graphicsadapter=0 \
      > "${CARLA_LOG}" 2>&1 &
    CARLA_PID=$!

    sleep "${CARLA_START_WAIT}"

    if ! kill -0 "${CARLA_PID}" 2>/dev/null; then
      echo "CARLA died before evaluator: ${ROUTE_NAME} attempt=${ATTEMPT}"
      tail -n 80 "${CARLA_LOG}" || true
      ATTEMPT=$((ATTEMPT + 1))
      continue
    fi

    if ! wait_for_carla_ready "${PORT}" "${CARLA_READY_TIMEOUT}" "${CARLA_READY_INTERVAL}"; then
      echo "CARLA process is alive but RPC is not ready: ${ROUTE_NAME} attempt=${ATTEMPT}"
      tail -n 120 "${CARLA_LOG}" || true
      cleanup_route_processes
      sleep "${ROUTE_CLEANUP_WAIT}"
      ATTEMPT=$((ATTEMPT + 1))
      continue
    fi

    setsid bash -o pipefail -c "
      python -u leaderboard/leaderboard/leaderboard_evaluator_local.py \
        --host localhost \
        --port ${PORT} \
        --traffic-manager-port ${TM_PORT} \
        --traffic-manager-seed 100 \
        --debug 0 \
        --timeout ${EVAL_TIMEOUT} \
        --routes ${ROUTE_FILE} \
        --repetitions 1 \
        --agent ${AGENT_PATH} \
        --agent-config ${AGENT_CONFIG} \
        --track SENSORS \
        --checkpoint ${RESULT_JSON} \
        --debug-checkpoint ${LIVE_TXT} \
        2>&1 | tee ${EVAL_LOG}
    " &
    EVAL_PID=$!

    while kill -0 "${EVAL_PID}" 2>/dev/null; do
      if ! kill -0 "${CARLA_PID}" 2>/dev/null; then
        echo "CARLA crashed while evaluator is running: ${ROUTE_NAME} attempt=${ATTEMPT}"
        tail -n 80 "${CARLA_LOG}" || true
        kill -TERM -"${EVAL_PID}" 2>/dev/null || true
        sleep 5
        kill -KILL -"${EVAL_PID}" 2>/dev/null || true
        break
      fi
      sleep 5
    done

    wait "${EVAL_PID}" 2>/dev/null
    EVAL_CODE=$?
    EVAL_PID=""

    cleanup_route_processes
    sleep "${ROUTE_CLEANUP_WAIT}"

    if [ -f "${RESULT_JSON}" ] && result_is_complete "${RESULT_JSON}"; then
      echo "Finished ${ROUTE_NAME}"
      break
    fi

    echo "Route did not finish cleanly: ${ROUTE_NAME} attempt=${ATTEMPT} eval_code=${EVAL_CODE}"
    tail -n 60 "${CARLA_LOG}" || true
    tail -n 60 "${EVAL_LOG}" || true

    rm -f "${RESULT_JSON}" "${LIVE_TXT}"
    ATTEMPT=$((ATTEMPT + 1))
  done

  if [ "${ATTEMPT}" -gt "${MAX_ATTEMPTS}" ]; then
    echo "Give up ${ROUTE_NAME} after ${MAX_ATTEMPTS} attempts"
  fi
done

echo "All requested routes processed: ${START_IDX}..${END_IDX}"
echo "Monitor log: ${OUT}/monitor/node_monitor.log"
