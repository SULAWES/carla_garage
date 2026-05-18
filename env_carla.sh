#!/usr/bin/env bash

export CARLA_ROOT="/home/HeavenlySU/sitp_workspace/carla_garage/carla"
export WORK_DIR="/home/HeavenlySU/sitp_workspace/carla_garage"
export SCENARIO_RUNNER_ROOT="${WORK_DIR}/scenario_runner"
export LEADERBOARD_ROOT="${WORK_DIR}/leaderboard"

export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla:${SCENARIO_RUNNER_ROOT}:${LEADERBOARD_ROOT}:${PYTHONPATH}"
