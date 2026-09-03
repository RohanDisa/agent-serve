#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/bench/results"
mkdir -p "$OUT"
"$ROOT/build/stream" --threads "$(nproc)" --meg 32 > "$OUT/stream.json"
"$ROOT/build/bench_matmul" --threads "$(nproc)" > "$OUT/gemm.json"
if [[ -n "${TINFERNO_MODEL:-}" ]]; then
  for st in v0 v1 v2 v3 v4 v5; do
    "$ROOT/build/bench_e2e" --model "$TINFERNO_MODEL" --stage "$st" --n-predict 64 \
      --runs 10 --warmup 3 --out "$OUT/e2e_$st.json" | tee "$OUT/e2e_$st.json"
  done
fi
python3 "$ROOT/bench/roofline.py" "$OUT"
