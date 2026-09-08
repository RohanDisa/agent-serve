# tinferno

A from-scratch C++17 Llama inference engine for CPU. The point is not to beat llama.cpp. The point is to measure, stage by stage, what actually moves prefill and decode, and to score decode against a bandwidth ceiling that matches the workload.

Prefill and decode are separate functions (`Model::prefill`, `Model::decode`). One binary, `--stage v0` through `v5`.

## Results

Measured on an AMD Ryzen 5 4600H (6c/12t, 45 W) with 2×8 GB DDR4-3200, theoretical peak 51.2 GB/s. Decode is scored against a 6-thread **read-only** STREAM ceiling of **19.42 GB/s** (gcc 11.4, WSL2). STREAM triad is lower and is the wrong roof: decode streams weights in and writes almost nothing back. F32 GEMM microbench: **83 GFLOP/s**.

Full tables, STREAM gcc vs MSVC, and JSON: [`RESULTS.md`](RESULTS.md), [`bench/results/`](bench/results/).

Same code, stage v2 (AVX2, 6 pinned threads), short prompt, 32 generated tokens:

| Model | Decode tok/s | Prefill tok/s | GB/s | % of 19.42 GB/s roof |
|---|---|---|---|---|
| Qwen2.5 0.5B Q8_0 | 20.0 | 42.0 | 13.4 | 69% |
| Llama 3.2 1B Q8_0 | 10.7 | 22.3 | 14.1 | 73% |
| TinyLlama 1.1B Q8_0 | 9.3 | 22.3 | 10.8 | 56% |
| Qwen2.5 1.5B Q8_0 | 6.8 | 13.7 | 12.8 | 66% |

Decode tok/s tracks 1 / weight size. Nothing sits on the STREAM line — the engine still pays for attention, RMSNorm, softmax, sampling, and thread barriers.

Llama 3.2 1B Instruct Q8_0, same machine:

| Stage | Decode tok/s | Prefill tok/s | % of 14.79 tok/s ceiling |
|---|---|---|---|
| v0 scalar, 1 thread | 1.16 | 1.41 | 8% |
| v1 scalar, 6 threads | 4.43 | 5.96 | 30% |
| v2 AVX2, 6 threads | **10.74** | 22.28 | **73%** |
| v4 + blocking | 10.41 | 12.09 | 70% |
| v5 mixed Q4 GGUF + head-major KV | 4.63 | 8.76 | 18% of the Q4 ceiling |

v0 → v2 is **9.3×** decode. SIMD and extra threads move prefill more than decode (scalar→AVX2: 2.4× decode, 3.7× prefill; 1→6 threads: 2.8× decode, 4.0× prefill). Blocking did nothing for decode and hurt 6-token prefill: there is no GEMM reuse at M=6.

The unsloth Llama 3.2 1B “Q4_0” GGUF is not a Q4_0 working set. `token_embd` (tied lm_head) stays Q6_K; two `ffn_down` tensors are Q4_1. Decode dropped below Q8_0.

## Build

C++17, CMake 3.16+. Linux / macOS:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Windows (MSVC 2022 x64):

```powershell
scripts/build.ps1
```

Binaries: `build/tinferno` or `build/Release/tinferno.exe`. CI builds and runs the unit tests on Ubuntu.

## Run

```bash
huggingface-cli download unsloth/Llama-3.2-1B-Instruct-GGUF \
  Llama-3.2-1B-Instruct-Q8_0.gguf --local-dir models
```

```bash
./build/tinferno --model models/Llama-3.2-1B-Instruct-Q8_0.gguf \
  --prompt "Write a haiku about memory bandwidth." \
  --stage v2 --n-predict 64 --temp 0 --pin --profile
```

Useful flags: `--threads N`, `--kv-layout seq|head`, `--tokens 1,2,3` (skip the tokenizer), `--max-seq N`.

Loads GGUF F32 / F16 / Q8_0 / Q4_0. Qwen2 is accepted as a Llama-style decoder with Q/K/V bias. Mixed files fall back to scalar dequant for Q4_1 and Q6_K.

## Stages

`--stage` is a preset, not a different binary.

| Stage | Threads | SIMD | Blocking | KV layout |
|---|---|---|---|---|
| v0 | 1 | off | off | seq-major |
| v1 | all | off | off | seq-major |
| v2 / v3 | all | AVX2 | off | seq-major |
| v4 | all | AVX2 | on | seq-major |
| v5 | all | AVX2 | on | head-major |

v2 and v3 are the same code path. Quantization comes from the GGUF, not from the stage flag.

Correctness: temperature-0 generations must match. CI runs that gate on a tiny synthetic GGUF so it does not need a 1B download.

## Measurement

Published numbers are medians. Protocol:

- `--pin`; SMT left on; STREAM pins OpenMP threads to physical cores
- warmup discarded; report median and IQR, never the best run
- temperature 0, seed 42
- prefill and decode timed separately; TTFT is prefill of the prompt
- decode latency p50 / p95 / p99
- machine, compiler, and flags next to the table in `RESULTS.md`

Decode ceiling:

```
tok/s_ceiling = STREAM_read_6t_GB/s × 1e9 / weight_bytes
```

Ceilings (STREAM, GEMM, e2e) are JSON under `bench/results/`. Regenerating STREAM (gcc/WSL2 + MSVC, triad + read-only):

```powershell
python scripts/run_stream.py
python bench/roofline.py bench/results
```

