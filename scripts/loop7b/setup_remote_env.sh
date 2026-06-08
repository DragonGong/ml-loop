#!/usr/bin/env bash
# 中文注释：在远程机器上准备 ml-loop 7B 实验环境。
set -euo pipefail

CONDA_BIN="${CONDA_BIN:-$HOME/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-ml-loop-py312}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
APPWORLD_ROOT="${APPWORLD_ROOT:-/data/appworld}"
HF_HOME="${HF_HOME:-$HOME/dragongong/.cache/huggingface}"
SKIP_PROJECT_INSTALL="${SKIP_PROJECT_INSTALL:-0}"
SKIP_APPWORLD_INSTALL="${SKIP_APPWORLD_INSTALL:-0}"
SKIP_APPWORLD_DOWNLOAD="${SKIP_APPWORLD_DOWNLOAD:-0}"

if [[ ! -x "$CONDA_BIN" ]]; then
  echo "Missing conda executable: $CONDA_BIN" >&2
  exit 1
fi

mkdir -p "$HF_HOME"

if ! "$CONDA_BIN" env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
  "$CONDA_BIN" create -y -n "$CONDA_ENV" "python=$PYTHON_VERSION" pip
fi

"$CONDA_BIN" run -n "$CONDA_ENV" python -m pip install -U pip poetry

if [[ "$SKIP_PROJECT_INSTALL" != "1" ]]; then
  POETRY_VIRTUALENVS_CREATE=false \
  HF_HOME="$HF_HOME" \
  TRANSFORMERS_CACHE="$HF_HOME/transformers" \
  HF_DATASETS_CACHE="$HF_HOME/datasets" \
    "$CONDA_BIN" run -n "$CONDA_ENV" poetry install
fi

if [[ "$SKIP_APPWORLD_INSTALL" != "1" ]]; then
  if [[ ! -x appworld-env/bin/appworld ]]; then
    "$CONDA_BIN" run -n "$CONDA_ENV" python -m virtualenv appworld-env
    appworld-env/bin/pip install -U pip
    appworld-env/bin/pip install click==8.2.1 appworld
    appworld-env/bin/appworld install
  fi

  mkdir -p "$APPWORLD_ROOT"
  if [[ "$SKIP_APPWORLD_DOWNLOAD" != "1" ]]; then
    appworld-env/bin/appworld download data --root "$APPWORLD_ROOT"
  fi
fi

echo "Conda env: $CONDA_ENV"
"$CONDA_BIN" run -n "$CONDA_ENV" python --version
"$CONDA_BIN" run -n "$CONDA_ENV" poetry --version
if [[ -x appworld-env/bin/appworld ]]; then
  echo "AppWorld env: appworld-env"
fi
echo "APPWORLD_ROOT: $APPWORLD_ROOT"
