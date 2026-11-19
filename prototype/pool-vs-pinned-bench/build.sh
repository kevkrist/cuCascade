#!/usr/bin/env bash
# Build against an existing release build of cuCascade: REPO=<checkout with build/release> ./build.sh
set -euo pipefail
REPO=${REPO:-/home/kkristensen/Code/cuCascade}
E=$REPO/.pixi/envs/default
$E/bin/x86_64-conda-linux-gnu-c++ -O3 -DNDEBUG -std=gnu++20 -DCCCL_DISABLE_PDL \
  -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP \
  -I$REPO/include -I$E/include/rapids -isystem $E/include \
  -I$E/targets/x86_64-linux/include -I$E/targets/x86_64-linux/include/cccl \
  bench.cpp -o bench -Wl,-rpath,$REPO/build/release:$E/lib \
  $REPO/build/release/libcucascade.so $REPO/build/release/libcucascade_topology_discovery.so \
  $E/lib/librmm.so $E/lib/librapids_logger.so $E/lib/libcudart_static.a -lrt -ldl -lpthread
