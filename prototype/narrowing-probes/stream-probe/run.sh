#!/bin/bash
# usage: run.sh <tag> ; runs all scenarios, always releases the hog
cd "$(dirname "$0")"
HOG=""
cleanup() { [ -n "$HOG" ] && kill "$HOG" 2>/dev/null; wait 2>/dev/null; }
trap cleanup EXIT INT TERM
smi() { timeout 20 nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader; }
one() { echo "## $*"; env "$@"; echo "exit=$?"; }
{
  echo "#### listed={1}"; one LISTED=1 timeout 180 ./build/stream_probe
  echo "#### listed={1} raw cudaStreamCreate"; one LISTED=1 timeout 180 ./build/stream_probe raw
  echo "#### listed={1} no phase B (destroy from main, nothing current)"; one LISTED=1 timeout 180 ./build/stream_probe nophaseb
} > out-listed1.txt 2>&1
{
  echo "#### listed={2} (A400)"; one LISTED=2 timeout 180 ./build/stream_probe
  echo "#### listed={2} raw"; one LISTED=2 timeout 180 ./build/stream_probe raw
} > out-listed2.txt 2>&1
hog() { # dev mib
  timeout 120 ./build/hog "$1" "$2" 100 & HOG=$!
  sleep 6; smi
}
unhog() { kill "$HOG" 2>/dev/null; wait "$HOG" 2>/dev/null; HOG=""; sleep 2; smi; }
{
  echo "#### Q4: CUDA dev 0 (unlisted) nearly full, listed={1}"
  hog 0 96700
  one LISTED=1 timeout 60 ./build/stream_probe nophaseb
  echo "--- same, with phase B (callers make full device 0 current: expected to fail in cudaSetDevice)"
  one LISTED=1 timeout 60 ./build/stream_probe
  unhog
} > out-q4-dev0-full.txt 2>&1
{
  echo "#### Q5: CUDA dev 1 (listed, stream device) nearly full, listed={1}"
  hog 1 96700
  one LISTED=1 timeout 60 ./build/stream_probe nophaseb
  echo "--- raw stream creation"
  one LISTED=1 timeout 60 ./build/stream_probe raw nophaseb
  unhog
} > out-q5-dev1-full.txt 2>&1
