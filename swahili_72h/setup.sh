#!/usr/bin/env bash
# File management/environment assembly only: no GPU work on the login node.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIF="${SIF:-$HOME/containers/anaconda-2025-12-2.sif}"
BASEPY="${BASEPY:-$HOME/.conda/envs/ld4lg-blackwell/bin/python}"
ENV="${SW_ENV:-/datasets/hbutt/swahili72/env}"
test -f "$SIF"; test -x "$BASEPY"
apptainer exec --bind "$HOME:$HOME,/datasets:/datasets" "$SIF" "$BASEPY" -c 'import sys; assert sys.version_info[:2] == (3,10), sys.version'
apptainer exec --bind "$HOME:$HOME,/datasets:/datasets" "$SIF" "$BASEPY" -m venv --system-site-packages "$ENV"
apptainer exec --bind "$HOME:$HOME,/datasets:/datasets" "$SIF" "$ENV/bin/python" -m pip install --no-index --no-deps --find-links "$ROOT/wheelhouse" -r "$ROOT/requirements.lock"
apptainer exec --bind "$HOME:$HOME,/datasets:/datasets" "$SIF" "$ENV/bin/python" -c 'import importlib.metadata as m; assert m.version("torch")=="2.7.1+cu128", m.version("torch"); print("Offline environment assembled; GPU tests run inside the one allocation.")'
