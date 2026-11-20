#!/usr/bin/env bash
# Usage (from the repo root): pixi run -e <env> bash prototype/pool-probe-envs/run.sh <env>
# Builds the probe in its own dir for that env and writes out-<env>.txt next to this script.
set -uo pipefail
ENV_NAME=$1
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=${REPO:-$PWD}
B=${BUILD_ROOT:-$REPO/build}/pool-probe-$ENV_NAME
OUT=$HERE/out-$ENV_NAME.txt
{
  echo "env=$ENV_NAME"
  echo "-- packages"
  (cd "$REPO" && pixi list -e "$ENV_NAME" 2>/dev/null |
    grep -E "^(librmm|libcudf|cuda-version|cuda-cudart-dev|cuda-nvcc|cuda-cccl|cuda-cccl_linux-64|rapids-cccl|gxx_linux-64|cmake) " |
    awk '{print $1, $2, $3}')
  echo "-- configure + build (repo -Werror flags on probe.cpp)"
  if cmake -S "$HERE" -B "$B" -G Ninja -DCMAKE_BUILD_TYPE=Release -DREPO_INCLUDE="$REPO/include" >"$B.configure.log" 2>&1 &&
    cmake --build "$B" >"$B.build.log" 2>&1; then
    echo "build=ok"
  else
    echo "build=FAILED"
    grep -E "error|Error" "$B.configure.log" "$B.build.log" | head -40
  fi
  echo "-- which CCCL copy probe.cpp includes"
  CMD=$(python3 -c "import json,sys;print([c['command'] for c in json.load(open(sys.argv[1])) if c['file'].endswith('probe.cpp')][0])" "$B/compile_commands.json")
  (cd "$B" && eval "${CMD/ -o / -H -fsyntax-only -o }" 2>&1 |
    grep -oE "[^ ]*/(shared_pinned_memory_pool\.h|shared_resource\.h|std/__cccl/version\.h)$" | sort -u |
    sed "s|$REPO/.pixi/envs/$ENV_NAME/||")
  echo "-- CCCL copies present in the env"
  (cd "$REPO/.pixi/envs/$ENV_NAME" && find . -path "*cuda/std/__cccl/version.h" | while read -r f; do
    echo "$f $(grep -m1 -E 'define CCCL_VERSION ' "$f" | awk '{print $3}') shared_pinned_pool_header=$([ -f "$(dirname "$f")/../../__memory_pool/shared_pinned_memory_pool.h" ] && echo yes || echo no)"
  done)
  echo "-- run"
  [ -x "$B/pool_probe" ] && "$B/pool_probe" 2>&1
} >"$OUT"
echo "wrote $OUT"
