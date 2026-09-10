# tinferno

A from-scratch C++17 Llama inference engine for CPU, built to measure what actually moves prefill and decode and to score decode against a bandwidth ceiling derived from the workload rather than a datasheet.

Prefill and decode are separate code paths (`Model::prefill`, `Model::decode`), kept apart because they sit on opposite sides of the roofline: prefill amortizes weight reads across many tokens, decode reads every weight for every token.

## Results

Measured on an AMD Ryzen 5 4600H (6c/12t, 45 W) with 2x8 GB DDR4-3200, theoretical peak 51.2 GB/s. Decode is scored against a 6-thread read-only STREAM ceiling of **19.42 GB/s** (gcc 11.4, WSL2). Triad is the wrong roof here: decode streams weights in and writes almost nothing back, so a read-plus-write benchmark understates the available read bandwidth. F32 GEMM microbenchmark: 83 GFLOP/s.

Full tables, gcc and MSVC STREAM figures, and raw JSON in [`RESULTS.md`](RESULTS.md) and [`bench/results/`](bench/results/).

**Four models, same binary, AVX2 with 6 pinned threads, 32 generated tokens:**

| Model | Decode tok/s | Prefill tok/s | Achieved GB/s | Percent of roof |
|---|---|---|---|---|
| Qwen2.5 0.5B Q8_0 | 20.0 | 42.0 | 13.4 | 69% |
| Llama 3.2 1B Q8_0 | 10.7 | 22.3 | 14.1 | 73% |
| TinyLlama 1.1B Q8_0 | 9.3 | 22.3 | 10.8 | 56% |
| Qwen2.5 1.5B Q8_0 | 6.8 | 13.7 | 12.8 | 66% |

Decode throughput tracks 1 / weight size across a 3x range in model size while achieved bandwidth stays in a narrow 10.8 to 14.1 GB/s band. That relationship, rather than any single figure, is what identifies decode as bandwidth-bound on this machine.

None of the four reach the STREAM line. The engine still pays for attention, RMSNorm, softmax, sampling, and a thread barrier on every matmul.

**Stage progression, Llama 3.2 1B Instruct Q8_0:**

| Stage | Decode tok/s | Prefill tok/s | Percent of 14.79 tok/s ceiling |
|---|---|---|---|
| v0 scalar, 1 thread | 1.16 | 1.41 | 8% |
| v1 scalar, 6 threads | 4.43 | 5.96 | 30% |
| v2 AVX2, 6 threads | **10.74** | 22.28 | **73%** |
| v4 + cache blocking | 10.41 | 12.09 | 70% |
| v5 mixed Q4 GGUF + head-major KV | 4.63 | 8.76 | 18% of the Q4 ceiling |

v0 to v2 is a **9.3x** decode improvement.

Both threading and SIMD move prefill more than decode: scalar to AVX2 gives 2.4x on decode against 3.7x on prefill, and 1 to 6 threads gives 2.8x against 4.0x. Faster per-thread compute saturates the memory system with fewer threads, so decode runs out of bandwidth headroom while prefill keeps scaling.

Cache blocking did nothing for decode and hurt short-prompt prefill. At M=6 there is no GEMM reuse for a blocked kernel to exploit.

## The Q4 result

The unsloth Llama 3.2 1B file labelled Q4_0 decoded slower than Q8_0 despite being 40 percent smaller, at 18 percent of its own ceiling against 73 percent for Q8_0. Achieved bandwidth fell from 14.1 GB/s to 3.5 GB/s.

Inspecting the file explains it. `token_embd`, which is the tied lm_head and therefore read in full on every token, stays at Q6_K, and two `ffn_down` tensors are Q4_1. Both fall back to a scalar dequant path in this engine, which moves the inner loop from bandwidth-bound to ALU-bound.

The underlying effect is documented in the llama.cpp community: K-quants carry more per-block work than the legacy formats, and mixed files leave the output tensor at higher precision by convention. What this measurement adds is the bandwidth number. The tok/s drop alone could be explained several ways; the collapse from 14.1 to 3.5 GB/s while every other configuration clusters above 10 identifies it specifically as the kernel leaving the memory-bound regime.

## Build

C++17, CMake 3.16 or newer.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Windows, MSVC 2022 x64:

```powershell
scripts/build.ps1
```

Binaries land at `build/tinferno` or `build/Release/tinferno.exe`. CI builds and runs the unit tests on Ubuntu.

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

Other flags: `--threads N`, `--kv-layout seq|head`, `--tokens 1,2,3` to skip the tokenizer, `--max-seq N`.

Loads GGUF F32, F16, Q8_0, and Q4_0. Qwen2 is handled as a Llama-style decoder with Q/K/V bias. Mixed files fall back to scalar dequant for Q4_1 and Q6_K.

## Stages

`--stage` selects a preset within one binary, not a separate build.

| Stage | Threads | SIMD | Blocking | KV layout |
|---|---|---|---|---|
| v0 | 1 | off | off | seq-major |
| v1 | all | off | off | seq-major |
| v2 / v3 | all | AVX2 | off | seq-major |
| v4 | all | AVX2 | on | seq-major |
| v5 | all | AVX2 | on | head-major |

v2 and v3 share a code path. Quantization comes from the GGUF file, not from the stage flag.

Correctness gate: temperature-0 generations must match across stages. CI runs this on a small synthetic GGUF so it does not require a 1B download.

## Measurement

All published figures are medians.

- Threads pinned with `--pin`, SMT left enabled; STREAM pins OpenMP threads to physical cores
- Warmup discarded, median and interquartile range reported, never best-of-N
- Temperature 0, seed 42
- Prefill and decode timed separately; time to first token is prefill of the prompt
- Decode latency reported at p50, p95, and p99
- Machine, compiler, and flags recorded alongside every table in `RESULTS.md`

Decode ceiling:

```
tok/s_ceiling = STREAM_read_6t_GB/s * 1e9 / weight_bytes
```

STREAM, GEMM, and end-to-end ceilings are stored as JSON under `bench/results/`. To regenerate STREAM across both toolchains and both kernel types:

```powershell
python scripts/run_stream.py
python bench/roofline.py bench/results
```

Prefill figures come from a short prompt, where prefill and decode are difficult to distinguish. Prefill claims at realistic prompt lengths need a rerun at 512 tokens or more.
