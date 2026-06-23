#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="$ROOT/build"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --target hist_eq_test -j

BIN=./hist_eq_test
echo "W H channels clahe tile_w tile_h avg_ms"
configs=("1024 768 1 0 64 64" "1024 768 1 1 64 64" "2048 1024 1 0 64 64" "2048 1024 1 1 64 64" "1024 768 3 1 64 64")
for cfg in "${configs[@]}"; do
  read -r W H C CLAHE TW TH <<< "$cfg"
  out=$($BIN $W $H $C $CLAHE $TW $TH |& tail -n1)
  # parse avg from output
  avg=$(echo "$out" | sed -n 's/.*avg=\([0-9.]*\) ms.*/\1/p')
  echo "$W $H $C $CLAHE $TW $TH $avg"
done
