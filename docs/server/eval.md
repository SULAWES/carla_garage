### 全量eval

下面是“完整 scenarios 开环测试”：遍历 /share/home/u19666033/djy/carla_dataset 下所有 scenario，每个 scenario 输出一个 CSV，然后汇总全局和
分场景指标。

默认每个 scenario 采 1024 个样本，和之前六场景口径一致。若你想每个 scenario 全量评估，把命令里的 --max-samples ${MAX_SAMPLES} 删除即可。

cd /share/home/u19666033/ltr/carla_garage
conda activate ltr_garage_2

export RUN_ID=origlike_ddp4_bs64x4_lr6e-4_ep100_fs5_spatial_imgenc0p5
export CKPT=/share/home/u19666033/ltr/dd_logs/full_baseline_basic/${RUN_ID}/latest.pth
export DATA_ROOT=/share/home/u19666033/djy/carla_dataset
export OUT=/share/home/u19666033/ltr/dd_logs/full_baseline_basic/${RUN_ID}/open_loop_all_scenarios
export MAX_SAMPLES=1024

mkdir -p ${OUT}/csv ${OUT}/logs

for SCENE_DIR in $(find ${DATA_ROOT} -mindepth 1 -maxdepth 1 -type d | sort); do
    SCENE=$(basename ${SCENE_DIR})
    CSV=${OUT}/csv/${RUN_ID}_${SCENE}.csv
    LOG=${OUT}/logs/${RUN_ID}_${SCENE}.log

    if [ -f "${CSV}" ]; then
        echo "Skip existing ${CSV}"
        continue
    fi

    echo "===== Open-loop eval: ${SCENE} ====="

    python tools/inspect_diffusiondrive_eval_errors.py \
        --root-dir ${SCENE_DIR} \
        --route-glob "*" \
        --checkpoint ${CKPT} \
        --top-k 50 \
        --max-samples ${MAX_SAMPLES} \
        --frame-sampling 5 \
        --batch-size 16 \
        --num-workers 6 \
        --device cuda:0 \
        --anchor-path /share/home/u19666033/ltr/4-0-0-1910-tracked_clusters_anchor.npy \
        --backbone-path /share/home/u19666033/ltr/pytorch_model.bin \
        --output-csv ${CSV} \
        2>&1 | tee ${LOG}
done

汇总所有 CSV：

python - <<'PY'
import csv
import statistics
from pathlib import Path

run_id = "origlike_ddp4_bs64x4_lr6e-4_ep100_fs5_spatial_imgenc0p5"
root = Path(f"/share/home/u19666033/ltr/dd_logs/full_baseline_basic/{run_id}/open_loop_all_scenarios")
csv_dir = root / "csv"

def pct(values, p):
    values = sorted(values)
    return values[round((len(values) - 1) * p)] if values else 0.0

def summarize(rows):
    out = {"n": len(rows)}
    for key in ("l1", "ade", "fde"):
        vals = [float(r[key]) for r in rows]
        out[f"{key}_mean"] = sum(vals) / len(vals) if vals else 0.0
        out[f"{key}_median"] = statistics.median(vals) if vals else 0.0
        out[f"{key}_p90"] = pct(vals, 0.90)
        out[f"{key}_p95"] = pct(vals, 0.95)
        out[f"{key}_p99"] = pct(vals, 0.99)
        out[f"{key}_max"] = max(vals) if vals else 0.0
    l1 = [float(r["l1"]) for r in rows]
    out["l1_gt1"] = sum(v > 1 for v in l1)
    out["l1_gt2"] = sum(v > 2 for v in l1)
    out["l1_gt4"] = sum(v > 4 for v in l1)
    return out

all_rows = []
by_scene = {}

for path in sorted(csv_dir.glob("*.csv")):
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        continue
    scene = rows[0].get("scenario") or path.stem.split("_")[-1]
    by_scene.setdefault(scene, []).extend(rows)
    all_rows.extend(rows)

fields = [
    "scope", "n",
    "l1_mean", "l1_median", "l1_p90", "l1_p95", "l1_p99", "l1_max", "l1_gt1", "l1_gt2", "l1_gt4",
    "ade_mean", "ade_median", "ade_p90", "ade_p95", "ade_p99", "ade_max",
    "fde_mean", "fde_median", "fde_p90", "fde_p95", "fde_p99", "fde_max",
]

summary_path = root / "summary_all_scenarios.csv"
with summary_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()

    agg = summarize(all_rows)
    agg["scope"] = "ALL"
    writer.writerow(agg)

    for scene, rows in sorted(by_scene.items()):
        row = summarize(rows)
        row["scope"] = scene
        writer.writerow(row)

print(f"Wrote {summary_path}")
print("ALL:", summarize(all_rows))
PY

结果文件：

${OUT}/csv/*.csv
${OUT}/logs/*.log
${OUT}/summary_all_scenarios.csv

如果你要“每个 scenario 全量样本”而不是每类最多 1024，改循环中的评估命令，删掉这一行：

--max-samples ${MAX_SAMPLES} \


二、闭环 smoke 测试，单窗口版

闭环先不要直接跑 220 条，先跑 1 条 route。单窗口没有 tmux，就把 CARLA 放后台，evaluator 在前台跑，退出时自动 kill CARLA。

cd /share/home/u19666033/ltr/carla_garage
conda activate ltr_garage_2

export RUN_ID=origlike_ddp4_bs64x4_lr6e-4_ep100_fs5_spatial_imgenc0p5
export CKPT=/share/home/u19666033/ltr/dd_logs/full_baseline_basic/${RUN_ID}/latest.pth
export OUT=/share/home/u19666033/ltr/dd_logs/full_baseline_basic/${RUN_ID}/closed_loop_smoke
mkdir -p ${OUT}

export CARLA_ROOT=/share/home/u19666033/syb/carla_0_9_15
export WORK_DIR=/share/home/u19666033/ltr/carla_garage
export SCENARIO_RUNNER_ROOT=${WORK_DIR}/scenario_runner
export LEADERBOARD_ROOT=${WORK_DIR}/leaderboard
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla:${SCENARIO_RUNNER_ROOT}:${LEADERBOARD_ROOT}:${PYTHONPATH}"

export DIFFUSIONDRIVE_CHECKPOINT=${CKPT}
export DIFFUSIONDRIVE_ANCHOR_PATH=/share/home/u19666033/ltr/4-0-0-1910-tracked_clusters_anchor.npy
export DIFFUSIONDRIVE_BACKBONE_PATH=/share/home/u19666033/ltr/pytorch_model.bin

export STOP_CONTROL=0
export DIFFUSIONDRIVE_SPATIAL_PID=1
export DIFFUSIONDRIVE_COMMAND_DELAY=0
export DIFFUSIONDRIVE_JPEG_ARTIFACT=1
export DIFFUSIONDRIVE_IMAGE_NORMALIZATION=none
export DIFFUSIONDRIVE_DEBUG_CONTROL=1
export DIFFUSIONDRIVE_DEBUG_INTERVAL=20

export PORT=2000
export STREAMING_PORT=2001
export ROUTE=leaderboard/data/bench2drive_split/bench2drive_00.xml

${CARLA_ROOT}/CarlaUE4.sh \
-carla-rpc-port=${PORT} \
-carla-streaming-port=${STREAMING_PORT} \
-nosound \
-RenderOffScreen \
-carla-primary-port=0 \
-graphicsadapter=0 \
> ${OUT}/carla.log 2>&1 &

CARLA_PID=$!
trap 'kill ${CARLA_PID} 2>/dev/null; wait ${CARLA_PID} 2>/dev/null' EXIT

sleep 180

python -u leaderboard/leaderboard/leaderboard_evaluator_local.py \
--routes ${ROUTE} \
--repetitions 1 \
--track SENSORS \
--checkpoint ${OUT}/bench2drive_00.json \
--debug-checkpoint ${OUT}/bench2drive_00_live.txt \
--agent team_code/diffusiondrive_agent.py \
--agent-config "" \
--debug 0 \
--traffic-manager-seed 0 \
--resume 1 \
--port ${PORT} \
--timeout 900 \
2>&1 | tee ${OUT}/bench2drive_00_eval.log

看这几个点：

cat ${OUT}/bench2drive_00.json
tail -n 100 ${OUT}/bench2drive_00_eval.log
tail -n 100 ${OUT}/carla.log

如果 agent setup、模型加载、传感器、route 都没问题，再跑完整闭环。

三、完整闭环测试，单窗口顺序版

这个会很久，但最稳，不需要 tmux。每条 route 重启一次 CARLA，结果逐条写到 ${OUT}/results。

cd /share/home/u19666033/ltr/carla_garage
conda activate ltr_garage_2

export RUN_ID=origlike_ddp4_bs64x4_lr6e-4_ep100_fs5_spatial_imgenc0p5
export CKPT=/share/home/u19666033/ltr/dd_logs/full_baseline_basic/${RUN_ID}/latest.pth
export OUT=/share/home/u19666033/ltr/dd_logs/full_baseline_basic/${RUN_ID}/closed_loop_bench2drive
mkdir -p ${OUT}/results ${OUT}/logs ${OUT}/carla_logs

export CARLA_ROOT=/share/home/u19666033/syb/carla_0_9_15
export WORK_DIR=/share/home/u19666033/ltr/carla_garage
export SCENARIO_RUNNER_ROOT=${WORK_DIR}/scenario_runner
export LEADERBOARD_ROOT=${WORK_DIR}/leaderboard
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla:${SCENARIO_RUNNER_ROOT}:${LEADERBOARD_ROOT}:${PYTHONPATH}"

export DIFFUSIONDRIVE_CHECKPOINT=${CKPT}
export DIFFUSIONDRIVE_ANCHOR_PATH=/share/home/u19666033/ltr/4-0-0-1910-tracked_clusters_anchor.npy
export DIFFUSIONDRIVE_BACKBONE_PATH=/share/home/u19666033/ltr/pytorch_model.bin

export STOP_CONTROL=0
export DIFFUSIONDRIVE_SPATIAL_PID=1
export DIFFUSIONDRIVE_COMMAND_DELAY=0
export DIFFUSIONDRIVE_JPEG_ARTIFACT=1
export DIFFUSIONDRIVE_IMAGE_NORMALIZATION=none
export DIFFUSIONDRIVE_DEBUG_CONTROL=0

for IDX in $(seq 0 219); do
    ROUTE_NAME=$(printf "bench2drive_%02d" ${IDX})
    ROUTE_XML=leaderboard/data/bench2drive_split/${ROUTE_NAME}.xml
    RESULT_JSON=${OUT}/results/${ROUTE_NAME}.json

    if [ -f "${RESULT_JSON}" ]; then
        echo "Skip existing ${RESULT_JSON}"
        continue
    fi

    PORT=$(comm -23 <(seq 2000 2099 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)
    STREAMING_PORT=$(comm -23 <(seq 2100 2199 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)

    echo "===== ${ROUTE_NAME} port=${PORT} streaming=${STREAMING_PORT} ====="

    ${CARLA_ROOT}/CarlaUE4.sh \
        -carla-rpc-port=${PORT} \
        -carla-streaming-port=${STREAMING_PORT} \
        -nosound \
        -RenderOffScreen \
        -carla-primary-port=0 \
        -graphicsadapter=0 \
        > ${OUT}/carla_logs/${ROUTE_NAME}_carla.log 2>&1 &

    CARLA_PID=$!
    sleep 180

    timeout 6h python -u leaderboard/leaderboard/leaderboard_evaluator_local.py \
        --routes ${ROUTE_XML} \
        --repetitions 1 \
        --track SENSORS \
        --checkpoint ${RESULT_JSON} \
        --debug-checkpoint ${OUT}/results/${ROUTE_NAME}_live.txt \
        --agent team_code/diffusiondrive_agent.py \
        --agent-config "" \
        --debug 0 \
        --traffic-manager-seed 0 \
        --resume 1 \
        --port ${PORT} \
        --timeout 900 \
        2>&1 | tee ${OUT}/logs/${ROUTE_NAME}_eval.log

    kill ${CARLA_PID} 2>/dev/null
    wait ${CARLA_PID} 2>/dev/null
    sleep 10
done

跑完后汇总：

python tools/result_parser.py \
--xml leaderboard/data/bench2drive220.xml \
--results ${OUT}/results

如果你只想先做较小闭环验证，把循环范围改成：

for IDX in $(seq 0 9); do

或者先跑 longest6：

ROUTE_XML=leaderboard/data/longest6_split/longest6_00.xml

tail -n 200 ${OUT}/monitor/node_monitor.log


注意，下面的脚本是从5开始的

mkdir -p ${OUT}/monitor

(
    while true; do
        echo "===== $(date) host=$(hostname) ====="
        nvidia-smi || true
        echo
        ps -u $USER -o pid,ppid,stat,etime,rss,cmd | grep -E 'CarlaUE4|CarlaUE4-Linux-Shipping|leaderboard_evaluator_local|python' | grep -v
        grep || true
        echo
        free -h || true
        echo
        sleep 10
    done
) > ${OUT}/monitor/node_monitor.log 2>&1 &

MONITOR_PID=$!
trap 'kill ${MONITOR_PID} 2>/dev/null || true' EXIT

result_is_complete() {
python - "$1" <<'PY'
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
    status = record.get("status", "")
    if status in rerun_status:
        sys.exit(1)

sys.exit(0)
PY
}


for IDX in $(seq 1 219); do
    ROUTE_NAME=$(printf "bench2drive_%02d" ${IDX})
    ROUTE_XML=leaderboard/data/bench2drive_split/${ROUTE_NAME}.xml
    RESULT_JSON=${OUT}/results/${ROUTE_NAME}.json

    if [ -f "${RESULT_JSON}" ]; then
        if result_is_complete "${RESULT_JSON}"; then
            echo "Skip finished ${RESULT_JSON}"
            continue
        else
            echo "Remove incomplete/crashed ${RESULT_JSON}"
            rm -f "${RESULT_JSON}"
            rm -f ${OUT}/results/${ROUTE_NAME}_live.txt
        fi
    fi


    for ATTEMPT in 1 2 3; do
        PORT=$(comm -23 <(seq 2000 2099 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)
        STREAMING_PORT=$(comm -23 <(seq 2100 2199 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)

        echo "===== ${ROUTE_NAME} attempt=${ATTEMPT} port=${PORT} streaming=${STREAMING_PORT} ====="

        setsid ${CARLA_ROOT}/CarlaUE4.sh \
            -carla-rpc-port=${PORT} \
            -carla-streaming-port=${STREAMING_PORT} \
            -nosound \
            -RenderOffScreen \
            -carla-primary-port=0 \
            -graphicsadapter=0 \
            > ${OUT}/carla_logs/${ROUTE_NAME}_attempt${ATTEMPT}_carla.log 2>&1 &

        CARLA_PID=$!
        sleep 180

        if ! kill -0 ${CARLA_PID} 2>/dev/null; then
            echo "CARLA crashed before evaluator, retry ${ROUTE_NAME}"
            tail -n 50 ${OUT}/carla_logs/${ROUTE_NAME}_attempt${ATTEMPT}_carla.log
            sleep 20
            continue
        fi

        timeout 6h python -u leaderboard/leaderboard/leaderboard_evaluator_local.py \
            --routes ${ROUTE_XML} \
            --repetitions 1 \
            --track SENSORS \
            --checkpoint ${RESULT_JSON} \
            --debug-checkpoint ${OUT}/results/${ROUTE_NAME}_live.txt \
            --agent team_code/diffusiondrive_agent.py \
            --agent-config "" \
            --debug 0 \
            --traffic-manager-seed 0 \
            --resume 1 \
            --port ${PORT} \
            --timeout 900 \
            2>&1 | tee ${OUT}/logs/${ROUTE_NAME}_attempt${ATTEMPT}_eval.log

        EVAL_CODE=${PIPESTATUS[0]}

        kill -TERM -${CARLA_PID} 2>/dev/null || true
        sleep 5
        kill -KILL -${CARLA_PID} 2>/dev/null || true
        pkill -9 -f "carla-rpc-port=${PORT}" || true
        pkill -9 -f "CarlaUE4-Linux-Shipping" || true
        wait ${CARLA_PID} 2>/dev/null || true
        sleep 10

        if [ ${EVAL_CODE} -eq 0 ] && [ -f "${RESULT_JSON}" ]; then
            echo "Finished ${ROUTE_NAME}"
            break
        fi

        echo "Evaluator failed for ${ROUTE_NAME}, retry"
        rm -f ${RESULT_JSON}
        sleep 30
    done
done

### OOM解决

最可能原因：前面某些 CARLA 进程没有被彻底杀掉，显存被残留的 CarlaUE4-Linux-Shipping 占住了。你当前脚本里 CARLA_PID=$! 拿到的可能只是
CarlaUE4.sh shell 进程，杀它不一定能稳定杀掉真正的 UE4 子进程。

先处理当前状态

在作业节点上执行：

nvidia-smi
ps -u $USER -o pid,ppid,stat,etime,rss,cmd | grep -E 'CarlaUE4|leaderboard_evaluator|python -u leaderboard' | grep -v grep

如果看到残留，清掉：

pkill -9 -f CarlaUE4-Linux-Shipping || true
pkill -9 -f CarlaUE4.sh || true
pkill -9 -f leaderboard_evaluator_local.py || true
sleep 10
nvidia-smi

删除失败结果，否则会被 skip：

OUT=/share/home/u19666033/ltr/dd_logs/full_baseline_basic/origlike_ddp4_bs64x4_lr6e-4_ep100_fs5_spatial_imgenc0p5/closed_loop_bench2drive

for IDX in 12 13 14 15 16; do
    ROUTE_NAME=$(printf "bench2drive_%02d" ${IDX})
    rm -f ${OUT}/results/${ROUTE_NAME}.json
    rm -f ${OUT}/results/${ROUTE_NAME}_live.txt
done

然后从 12 继续跑。

把脚本里的 kill 改稳

启动 CARLA 时用 setsid 新建进程组：

setsid ${CARLA_ROOT}/CarlaUE4.sh \
    -carla-rpc-port=${PORT} \
    -carla-streaming-port=${STREAMING_PORT} \
    -nosound \
    -RenderOffScreen \
    -carla-primary-port=0 \
    -graphicsadapter=0 \
    > ${OUT}/carla_logs/${ROUTE_NAME}_attempt${ATTEMPT}_carla.log 2>&1 &

CARLA_PID=$!

结束时不要只 kill ${CARLA_PID}，改成杀整个进程组，并兜底 pkill：

kill -TERM -${CARLA_PID} 2>/dev/null || true
sleep 5
kill -KILL -${CARLA_PID} 2>/dev/null || true
pkill -9 -f "carla-rpc-port=${PORT}" || true
pkill -9 -f "CarlaUE4-Linux-Shipping" || true
wait ${CARLA_PID} 2>/dev/null || true
sleep 10

不要把失败 json 当成功

你现在的 skip 逻辑只看文件存在不够。建议改成只跳过真正成功的结果。简单一点：先不要自动 skip Failed - Agent couldn't be set up 的 json。可
以在每个 route 开头加：

if [ -f "${RESULT_JSON}" ]; then
    STATUS=$(python -c 'import json,sys; d=json.load(open(sys.argv[1])); rs=d.get("_checkpoint",{}).get("records",[]);
    print(rs[0].get("status","") if rs else "")' "${RESULT_JSON}")
    if [ "${STATUS}" = "Completed" ]; then
        echo "Skip completed ${RESULT_JSON}"
        continue
    fi
    echo "Remove incomplete/failed ${RESULT_JSON}: ${STATUS}"
    rm -f "${RESULT_JSON}"
fi

有些正常结束可能不是严格 Completed，如果你发现成功 route 的 status 名称不同，就把条件改成排除：

if echo "${STATUS}" | grep -q "Failed"; then rm -f "${RESULT_JSON}"; fi

如果仍然 OOM

优先级：

1. 换一个干净节点重新从 12 跑。
2. 每条 route 后强制 pkill -9 -f CarlaUE4-Linux-Shipping。
3. 不要同时跑多个闭环 evaluator。
4. 关闭 debug 和保存：

    export DIFFUSIONDRIVE_DEBUG_CONTROL=0
    export SAVE_PATH=

5. 如果 GPU 本身显存太小或 CARLA 占用太高，再考虑把模型推理降显存，但从你“前 11 条能跑”看，首要问题还是残留进程/显存累积。

结论：先清进程、删失败 json、把 CARLA 启动改成 setsid + 杀进程组，再从 bench2drive_12 继续。


对，闭环里 status=Failed 不一定等于“这条 route 没跑完”。它可能是正常完成但发生了 route failure / infraction / blocked / timeout，这种结果
应该保留，不能重跑。真正要重跑的是“agent 没 setup / simulation crashed / agent crashed / 进度没完成 / json 损坏”。

所以 skip 逻辑不要按 status == Completed，而应该按：

progress 已完成 + 不是 setup/crash 类失败

用这个函数判断最稳。

推荐 skip 判断

在闭环脚本前面加：

result_is_complete() {
python - "$1" <<'PY'
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
    status = record.get("status", "")
    if status in rerun_status:
        sys.exit(1)

sys.exit(0)
PY
}

然后 route 开头这样写：

if [ -f "${RESULT_JSON}" ]; then
if result_is_complete "${RESULT_JSON}"; then
    echo "Skip finished ${RESULT_JSON}"
    continue
else
    echo "Remove incomplete/crashed ${RESULT_JSON}"
    rm -f "${RESULT_JSON}"
    rm -f ${OUT}/results/${ROUTE_NAME}_live.txt
fi
fi

这样：

- Completed：skip
- 普通 Failed：skip，因为它可能是有效闭环失败结果
- Failed - Agent couldn't be set up：删除重跑
- Failed - Simulation crashed：删除重跑
- Failed - Agent crashed：删除重跑
- progress 没完成：删除重跑
- json 损坏：删除重跑

你现在这几个 OOM 的结果需要删

因为是：

Failed - Agent couldn't be set up

所以应该删除重跑：

OUT=/share/home/u19666033/ltr/dd_logs/full_baseline_basic/origlike_ddp4_bs64x4_lr6e-4_ep100_fs5_spatial_imgenc0p5/closed_loop_bench2drive

for IDX in 12 13 14 16; do
ROUTE_NAME=$(printf "bench2drive_%02d" ${IDX})
rm -f ${OUT}/results/${ROUTE_NAME}.json
rm -f ${OUT}/results/${ROUTE_NAME}_live.txt
done

bench2drive_15 你要看有没有结果文件；如果没有，本来就会继续跑。


cat > run_baseline_basic_closed_loop.sh <<'BASH'
#!/usr/bin/env bash
set -u

START_IDX=${START_IDX:-15}
END_IDX=${END_IDX:-219}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-2}

export CARLA_ROOT=/share/home/u19666033/syb/carla_0_9_15
export WORK_DIR=/share/home/u19666033/ltr/carla_garage
export SCENARIO_RUNNER_ROOT=${WORK_DIR}/scenario_runner
export LEADERBOARD_ROOT=${WORK_DIR}/leaderboard
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla:${SCENARIO_RUNNER_ROOT}:${LEADERBOARD_ROOT}:${PYTHONPATH:-}"

export RUN_ID=origlike_ddp4_bs64x4_lr6e-4_ep100_fs5_spatial_imgenc0p5
export OUT=/share/home/u19666033/ltr/dd_logs/full_baseline_basic/${RUN_ID}/closed_loop_bench2drive
export CKPT=/share/home/u19666033/ltr/dd_logs/full_baseline_basic/${RUN_ID}/latest.pth

export DIFFUSIONDRIVE_CHECKPOINT=${CKPT}
export DIFFUSIONDRIVE_ANCHOR_PATH=/share/home/u19666033/ltr/4-0-0-1910-tracked_clusters_anchor.npy
export DIFFUSIONDRIVE_BACKBONE_PATH=/share/home/u19666033/ltr/pytorch_model.bin
export STOP_CONTROL=0
export DIFFUSIONDRIVE_SPATIAL_PID=1
export DIFFUSIONDRIVE_COMMAND_DELAY=0
export DIFFUSIONDRIVE_JPEG_ARTIFACT=1
export DIFFUSIONDRIVE_IMAGE_NORMALIZATION=none

mkdir -p ${OUT}/results ${OUT}/logs ${OUT}/carla_logs ${OUT}/monitor

result_is_complete() {
    python - "$1" <<'PY'
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

cleanup_route_processes() {
    if [ "${EVAL_PID:-}" != "" ]; then
        kill -TERM -${EVAL_PID} 2>/dev/null || true
        sleep 2
        kill -KILL -${EVAL_PID} 2>/dev/null || true
        wait ${EVAL_PID} 2>/dev/null || true
    fi

    if [ "${CARLA_PID:-}" != "" ]; then
        kill -TERM -${CARLA_PID} 2>/dev/null || true
        sleep 5
        kill -KILL -${CARLA_PID} 2>/dev/null || true
        wait ${CARLA_PID} 2>/dev/null || true
    fi

    pkill -9 -f "CarlaUE4-Linux-Shipping" 2>/dev/null || true
    pkill -9 -f "CarlaUE4.sh" 2>/dev/null || true
}

(
    while true; do
        echo "===== $(date) host=$(hostname) ====="
        nvidia-smi || true
        echo
        ps -u $USER -o pid,ppid,stat,etime,rss,cmd | grep -E 'CarlaUE4|CarlaUE4-Linux-Shipping|leaderboard_evaluator_local|python' | grep -v
        grep || true
        echo
        free -h || true
        echo
        sleep 10
    done
) > ${OUT}/monitor/node_monitor.log 2>&1 &
MONITOR_PID=$!

trap 'cleanup_route_processes; kill ${MONITOR_PID} 2>/dev/null || true' EXIT

cd ${WORK_DIR}

for IDX in $(seq ${START_IDX} ${END_IDX}); do
    ROUTE_NAME=$(printf "bench2drive_%02d" ${IDX})
    ROUTE_FILE=${WORK_DIR}/leaderboard/data/bench2drive_split/${ROUTE_NAME}.xml
    RESULT_JSON=${OUT}/results/${ROUTE_NAME}.json
    LIVE_TXT=${OUT}/results/${ROUTE_NAME}_live.txt

    if [ ! -f "${ROUTE_FILE}" ]; then
        echo "Missing route file: ${ROUTE_FILE}"
        continue
    fi

    if [ -f "${RESULT_JSON}" ]; then
        if result_is_complete "${RESULT_JSON}"; then
        echo "Skip finished ${ROUTE_NAME}: ${RESULT_JSON}"
        continue
        else
        echo "Remove incomplete/crashed result: ${RESULT_JSON}"
        rm -f "${RESULT_JSON}" "${LIVE_TXT}"
        fi
    fi

    ATTEMPT=1
    while [ ${ATTEMPT} -le ${MAX_ATTEMPTS} ]; do
        PORT=$((2050 + IDX + ATTEMPT * 100))
        STREAMING_PORT=$((2150 + IDX + ATTEMPT * 100))
        TM_PORT=$((8050 + IDX + ATTEMPT * 100))

        CARLA_LOG=${OUT}/carla_logs/${ROUTE_NAME}_attempt${ATTEMPT}_carla.log
        EVAL_LOG=${OUT}/logs/${ROUTE_NAME}_attempt${ATTEMPT}_eval.log

        echo "===== ${ROUTE_NAME} attempt=${ATTEMPT} port=${PORT} streaming=${STREAMING_PORT} tm=${TM_PORT} ====="

        cleanup_route_processes
        sleep 8

        setsid ${CARLA_ROOT}/CarlaUE4.sh \
        -carla-rpc-port=${PORT} \
        -carla-streaming-port=${STREAMING_PORT} \
        -nosound -RenderOffScreen -carla-primary-port=0 -graphicsadapter=0 \
        > "${CARLA_LOG}" 2>&1 &
        CARLA_PID=$!

        sleep 25

        if ! kill -0 ${CARLA_PID} 2>/dev/null; then
        echo "CARLA died before evaluator: ${ROUTE_NAME} attempt=${ATTEMPT}"
        tail -n 80 "${CARLA_LOG}" || true
        ATTEMPT=$((ATTEMPT + 1))
        continue
        fi

        setsid bash -c "
        python -u leaderboard/leaderboard/leaderboard_evaluator_local.py \
            --host localhost \
            --port ${PORT} \
            --traffic-manager-port ${TM_PORT} \
            --traffic-manager-seed 100 \
            --debug 0 \
            --timeout 300 \
            --routes ${ROUTE_FILE} \
            --repetitions 1 \
            --agent ${WORK_DIR}/team_code/diffusiondrive_agent.py \
            --agent-config ${WORK_DIR}/team_code/diffusiondrive_agent.py \
            --track SENSORS \
            --checkpoint ${RESULT_JSON} \
            --debug-checkpoint ${LIVE_TXT} \
            2>&1 | tee ${EVAL_LOG}
        " &
        EVAL_PID=$!

        while kill -0 ${EVAL_PID} 2>/dev/null; do
        if ! kill -0 ${CARLA_PID} 2>/dev/null; then
            echo "CARLA crashed while evaluator is running: ${ROUTE_NAME} attempt=${ATTEMPT}"
            tail -n 80 "${CARLA_LOG}" || true
            kill -TERM -${EVAL_PID} 2>/dev/null || true
            sleep 5
            kill -KILL -${EVAL_PID} 2>/dev/null || true
            break
        fi
        sleep 5
        done

        wait ${EVAL_PID} 2>/dev/null
        EVAL_CODE=$?

        cleanup_route_processes
        sleep 10

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

    if [ ${ATTEMPT} -gt ${MAX_ATTEMPTS} ]; then
        echo "Give up ${ROUTE_NAME} after ${MAX_ATTEMPTS} attempts"
    fi
done

echo "All requested routes processed: ${START_IDX}..${END_IDX}"
echo "Monitor log: ${OUT}/monitor/node_monitor.log"
BASH

chmod +x run_baseline_basic_closed_loop.sh

START_IDX=15 END_IDX=219 MAX_ATTEMPTS=2 bash run_baseline_basic_closed_loop.sh

中途想看状态，不用新 shell，等当前前台输出间隙直接看日志路径即可；如果卡住很久，可以 Ctrl+Z 后执行：

tail -n 200 ${OUT}/monitor/node_monitor.log
fg

这个版本的关键点是：CARLA 一旦 Exit 139，脚本会发现 CARLA_PID 已死，杀掉 evaluator，不再空等 --timeout。
已写成脚本并设为可执行：

carla_garage/tools/run_baseline_basic_closed_loop.sh

在作业节点单窗口里这样跑：

cd /share/home/u19666033/ltr/carla_garage
START_IDX=15 END_IDX=219 MAX_ATTEMPTS=2 bash tools/run_baseline_basic_closed_loop.sh

默认会使用当前 baseline-basic：

RUN_ID=origlike_ddp4_bs64x4_lr6e-4_ep100_fs5_spatial_imgenc0p5
STOP_CONTROL=0
DIFFUSIONDRIVE_SPATIAL_PID=1

监控日志会写到：

/share/home/u19666033/ltr/dd_logs/full_baseline_basic/${RUN_ID}/closed_loop_bench2drive/monitor/node_monitor.log


### 文件版

已写成脚本并设为可执行：

carla_garage/tools/run_baseline_basic_closed_loop.sh

在作业节点单窗口里这样跑：

cd /share/home/u19666033/ltr/carla_garage
START_IDX=15 END_IDX=219 MAX_ATTEMPTS=5 bash tools/run_baseline_basic_closed_loop.sh

默认会使用当前 baseline-basic：

RUN_ID=origlike_ddp4_bs64x4_lr6e-4_ep100_fs5_spatial_imgenc0p5
STOP_CONTROL=0
DIFFUSIONDRIVE_SPATIAL_PID=1

监控日志会写到：

/share/home/u19666033/ltr/dd_logs/full_baseline_basic/${RUN_ID}/closed_loop_bench2drive/monitor/node_monitor.log

### 文件修正

你这个现象的核心是：CarlaUE4 进程存在不等于 CARLA RPC 已经可用。之前脚本只检查 PID，所以 CARLA 半启动/卡住时 evaluator 还是会进去等
get_world().apply_settings()，最后报 300s timeout。

我已经更新了脚本：

carla_garage/tools/run_baseline_basic_closed_loop.sh

改了两点：

- evaluator 前新增 CARLA RPC readiness probe：主动调用 carla.Client(...).get_world().apply_settings(...)，确认 simulator 真能响应后才启动
evaluator。

- evaluator 管道加了 pipefail，避免 Python traceback 但 tee 成功导致 eval_code=0 的误判。

运行建议：

cd /share/home/u19666033/ltr/carla_garage

START_IDX=60 \
END_IDX=219 \
MAX_ATTEMPTS=5 \
CARLA_START_WAIT=30 \
CARLA_READY_TIMEOUT=180 \
CARLA_READY_INTERVAL=5 \
EVAL_TIMEOUT=300 \
bash tools/run_baseline_basic_closed_loop.sh

如果之后看到：

CARLA process is alive but RPC is not ready

说明 CARLA 进程还在，但 RPC 端口不可用/初始化卡死，脚本会杀掉并 retry，不会再白等 evaluator 的 300 秒。

另外这行：

FUnixPlatformMisc::RequestExit

多半是脚本清理时终止 CARLA 产生的，不一定是原始崩溃原因。真正要看的是它前面有没有 RenderThread timed out、Signal 11、Segmentation fault，
或者 readiness probe 一直连不上。


### A/B 测试

for IDX in 24 0 94 50 139 9 15 212 115 185 153 203 105 19 101 4 167 194 102 43; do
    START_IDX=${IDX} END_IDX=${IDX} \
    OUT=/share/home/u19666033/ltr/dd_logs/full_baseline_basic/ablation_20routes/current \
    MAX_ATTEMPTS=5 \
    bash tools/run_baseline_basic_closed_loop.sh
done

run20 () {
    EXP=$1
    shift
    for IDX in ${ROUTES}; do
      env \
        OUT=${BASE_OUT}/${EXP} \
        START_IDX=${IDX} \
        END_IDX=${IDX} \
        MAX_ATTEMPTS=2 \
        CARLA_READY_TIMEOUT=180 \
        "$@" \
        bash tools/run_baseline_basic_closed_loop.sh
    done

    python tools/result_parser.py \
      --xml leaderboard/data/bench2drive220.xml \
      --results ${BASE_OUT}/${EXP}/results
}

通用跑法
在远端 carla_garage 下先定义：

cd /share/home/u19666033/ltr/carla_garage

ROUTES="24 0 94 50 139 9 15 212 115 185 153 203 105 19 101 4 167 194 102 43"
BASE_OUT=/share/home/u19666033/ltr/dd_logs/full_baseline_basic/ablation_20routes

run20 () {
    EXP=$1
    shift
    for IDX in ${ROUTES}; do
        env \
        OUT=${BASE_OUT}/${EXP} \
        START_IDX=${IDX} \
        END_IDX=${IDX} \
        MAX_ATTEMPTS=2 \
        CARLA_READY_TIMEOUT=180 \
        "$@" \
        bash tools/run_baseline_basic_closed_loop.sh
    done

    python tools/result_parser.py \
        --xml leaderboard/data/bench2drive220.xml \
        --results ${BASE_OUT}/${EXP}/results
}

每个实验用一个新的 OUT，否则脚本会 skip 已完成 JSON。

1. 基准复跑
先复跑一次当前 baseline，确认 20-route 小集稳定性。

run20 A0_baseline

用途：后面所有实验都和 A0_baseline/results/results.csv 比。

2. command delay
这个现在不用改代码，直接测：

run20 A1_command_delay DIFFUSIONDRIVE_COMMAND_DELAY=1

重点看：bench2drive_94/50/139/09/15/212 的 route deviation 和 RC 是否改善。
如果 deviation 下降但低速/碰撞变差，要谨慎。

3. stop sign privileged ablation
这个不是 sensor-only 主结果，只用来判断 stop sign 规则损失占比。

run20 A2_stop_control STOP_CONTROL=1

重点看：bench2drive_101/102/194/203/15/43 的 stop infractions、DS、collision。
如果分数明显上升，说明 sensor-only stop sign 能力后面必须补。

4. spatial PID fallback
测一下旧 time-index 控制逻辑是不是完全不可用：

run20 A3_time_index_pid DIFFUSIONDRIVE_SPATIAL_PID=0

重点看：平均速度、min speed、route deviation。
我预期它大概率不如 spatial PID，但值得作为 sanity check。

5. debug 日志
这个不一定要跑满 20 条，先跑 5-10 条代表 route：

ROUTES_DEBUG="94 139 115 19 167 194"

for IDX in ${ROUTES_DEBUG}; do
    env \
        OUT=${BASE_OUT}/D0_debug \
        START_IDX=${IDX} \
        END_IDX=${IDX} \
        MAX_ATTEMPTS=1 \
        DIFFUSIONDRIVE_DEBUG_CONTROL=1 \
        DIFFUSIONDRIVE_DEBUG_INTERVAL=5 \
        bash tools/run_baseline_basic_closed_loop.sh
done

看 logs/*_eval.log 里的：

desired_speed
turn_ratio
aim_index
steer/throttle/brake
stuck
force_move
cmd_used/cmd_current/cmd_delayed

这一步主要定位问题，不以分数为主。

6. 低速允许转向
这个当前还没有开关，需要先改代码加一个 env，例如 DIFFUSIONDRIVE_LOW_SPEED_STEER=1。加完后测：

run20 A4_low_speed_steer DIFFUSIONDRIVE_LOW_SPEED_STEER=1

重点 route：09 15 212 105 194 102 43。
重点指标：route deviation、outside_route_lanes、collision。
目标是减少低速起步直冲导致的偏航。

7. 提前 stuck recovery
当前 stuck_threshold=1100 基本太晚，也需要先加 env 覆盖，例如 DIFFUSIONDRIVE_STUCK_THRESHOLD。

run20 A5_stuck120 DIFFUSIONDRIVE_STUCK_THRESHOLD=120
run20 A6_stuck170 DIFFUSIONDRIVE_STUCK_THRESHOLD=170
run20 A7_stuck300 DIFFUSIONDRIVE_STUCK_THRESHOLD=300

重点 route：115 185 153 203 105 19 101 4 167。
重点指标：vehicle_blocked、route_timeout、min_speed，同时必须看 collision 是否上升。
如果 blocked 降了但 collision 大涨，阈值太激进。

8. spatial PID 速度参数
也建议先加 env 覆盖，例如：

DIFFUSIONDRIVE_SPATIAL_PID_SPEED_FAST
DIFFUSIONDRIVE_SPATIAL_PID_SPEED_SLOW
DIFFUSIONDRIVE_SPATIAL_PID_TURN_THRESHOLD
DIFFUSIONDRIVE_SPATIAL_PID_SHARP_TURN_THRESHOLD

然后测：

run20 A8_pid_7_3 \
    DIFFUSIONDRIVE_SPATIAL_PID_SPEED_FAST=7.0 \
    DIFFUSIONDRIVE_SPATIAL_PID_SPEED_SLOW=3.0

run20 A9_pid_6_2p5 \
    DIFFUSIONDRIVE_SPATIAL_PID_SPEED_FAST=6.0 \
    DIFFUSIONDRIVE_SPATIAL_PID_SPEED_SLOW=2.5

重点 route：139 19 167 43 50 94。
目标是减少 min speed，但不能明显增加 collision / route deviation。

9. route guard / route blend
这个需要实现后再测，建议 env：

DIFFUSIONDRIVE_ROUTE_GUARD=1
DIFFUSIONDRIVE_ROUTE_BLEND_ALPHA=0.3

测试：

run20 A10_route_guard \
    DIFFUSIONDRIVE_ROUTE_GUARD=1 \
    DIFFUSIONDRIVE_ROUTE_BLEND_ALPHA=0.3

重点 route：94 50 139 09 15 212。
这是最针对 deviated=69 的实验。目标是 route_dev 明显下降，RC 上升，且正常 route 24/00 不退化。

比较结果
每个实验跑完后看：

for EXP in A0_baseline A1_command_delay A2_stop_control A3_time_index_pid A4_low_speed_steer A5_stuck120 A8_pid_7_3 A10_route_guard; do
    echo "===== ${EXP} ====="
    sed -n '1,18p' ${BASE_OUT}/${EXP}/results/results.csv
done

我的建议顺序是：

1. A0_baseline
2. D0_debug
3. A1_command_delay
4. A2_stop_control
5. 先改代码加 env 开关
6. A4_low_speed_steer
7. A5/A6/A7 stuck threshold
8. A8/A9 PID speed
9. A10 route guard

最优先验证的是 low_speed_steer + stuck_threshold + route_guard，这三项最像是在修现在的执行层失败。

已加好低速转向开关，默认不改变 baseline 行为。

改动点：

- carla_garage/team_code/diffusiondrive_agent.py:87 新增 DIFFUSIONDRIVE_LOW_SPEED_STEER。
- 默认 0：保持原逻辑，speed < 0.01 时清零横向 angle。
- 设置 1：低速但未 brake 时保留预测轨迹对应的转向。
- brake 激活时仍然清零 angle，避免停车/急刹状态下乱打方向。
- debug 日志新增 angle、raw_angle、angle_reset、low_speed_steer。
- carla_garage/tools/run_baseline_basic_closed_loop.sh:47 已导出默认值，远端脚本可直接传 env。
- 已同步更新非 server 文档和根目录 AGENTS.md；docs/server/* 没有动。

验证通过：

python -m py_compile team_code/diffusiondrive_agent.py
bash -n tools/run_baseline_basic_closed_loop.sh

测试这个开关时用：

run20 A4_low_speed_steer DIFFUSIONDRIVE_LOW_SPEED_STEER=1

建议先重点看 bench2drive_09/15/50/94/139/212 的 route_dev / RC / collision，确认低速起步保留转向是否真的减少偏航。

你同步这两个文件到远端后，建议重新跑：

run20 A5_stuck120_real DIFFUSIONDRIVE_STUCK_THRESHOLD=120
run20 A6_stuck170_real DIFFUSIONDRIVE_STUCK_THRESHOLD=170
run20 A7_stuck300_real DIFFUSIONDRIVE_STUCK_THRESHOLD=300

同时可以继续跑 spatial PID：

run20 A8_pid_6_2p5 \
    DIFFUSIONDRIVE_SPATIAL_PID_SPEED_FAST=6.0 \
    DIFFUSIONDRIVE_SPATIAL_PID_SPEED_SLOW=2.5

run20 A10_pid_6p5_2p75 \
    DIFFUSIONDRIVE_SPATIAL_PID_SPEED_FAST=6.5 \
    DIFFUSIONDRIVE_SPATIAL_PID_SPEED_SLOW=2.75

run20 A11_pid_7_2p5 \
    DIFFUSIONDRIVE_SPATIAL_PID_SPEED_FAST=7.0 \
    DIFFUSIONDRIVE_SPATIAL_PID_SPEED_SLOW=2.5

run20 A12_pid_7_2p5_turn_0p20_0p45 \
    DIFFUSIONDRIVE_SPATIAL_PID_SPEED_FAST=7.0 \
    DIFFUSIONDRIVE_SPATIAL_PID_SPEED_SLOW=2.5 \
    DIFFUSIONDRIVE_SPATIAL_PID_TURN_THRESHOLD=0.20 \
    DIFFUSIONDRIVE_SPATIAL_PID_SHARP_TURN_THRESHOLD=0.45
