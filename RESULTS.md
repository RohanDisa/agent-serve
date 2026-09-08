# Results

Numbers below are medians from the JSON in `bench/results/`. Do not copy a throughput figure that is not in those files. Engine tok/s were not rerun; only the STREAM ceiling and the percentages derived from it changed.

## Hardware

Ryzen 5 4600H (**45 W**, not the 15 W 4600U), 2x8GB Micron 4ATF1G64HZ DDR4-3200 dual-channel at rated speed, theoretical peak 51.2 GB/s. Decode ceiling is STREAM **read-only** at 6 threads: **19.42 GB/s** (gcc 11.4 / WSL2). STREAM triad is kept for reference and is not the decode roof.

| | |
|---|---|
| CPU | AMD Ryzen 5 4600H, 6 physical cores / 12 threads, 3.0 GHz base, 45 W |
| RAM | 2 × 8 GB Micron 4ATF1G64HZ DDR4-3200, dual-channel, ConfiguredClockSpeed 3200 |
| Theoretical DRAM peak | 51.2 GB/s (3200 MT/s × 8 B × 2 channels) |
| STREAM read-only (decode ceiling) | **19.42 GB/s** median at 6 threads (IQR 18.54–19.75), gcc `-O3 -march=native -fopenmp`, N=`STREAM_ARRAY_SIZE=100000000` (800e6 bytes/array) |
| STREAM triad (reference) | gcc 11.10 GB/s / MSVC 13.83 GB/s at 6 threads. Retired decode ceiling: MSVC triad 14.10 GB/s |
| Peak F32 GEMM | **82.86 GFLOP/s** median, M=256 N=K=2048, `gemm.json` |
| OS | Windows 10.0.26200. STREAM gcc runs in WSL2 Ubuntu 22.04 (12 vCPUs). Engine benches are native Windows/MSVC |
| STREAM compilers | gcc 11.4.0 (`-O3 -march=native -fopenmp -DSTREAM_ARRAY_SIZE=100000000`); MSVC 19.31.31104.0 (`/O2 /openmp /arch:AVX2 /DSTREAM_ARRAY_SIZE=100000000`) |
| SMT | left on. STREAM pins OpenMP threads to even logical CPUs (physical cores 0,2,4,6,8,10) via `sched_setaffinity` / `SetThreadAffinityMask` |

Linux `perf` was not used.

### Why the old 14.10 GB/s roof was wrong

Llama 3.2 1B Q8 v2 decode landed at **14.11 GB/s**, exactly 100% of the previous MSVC triad ceiling. That cannot be a correctly measured roof: the engine does everything STREAM triad does plus attention, RMSNorm, softmax, sampling, and per-matmul barriers. Three checks:

1. **MSVC vs gcc.** `/O2 /openmp` is not `-O3 -march=native -fopenmp`. Rebuilt STREAM in WSL2 with gcc 11.4. gcc **did not** raise triad: 6-thread gcc triad is 11.10 GB/s vs MSVC 13.83 GB/s. 1-thread is 8.78 (gcc) vs 10.34 (MSVC), not the 18–24 GB/s a Zen 2 core can move on a well-vectorized loop. WSL2's write path looks worse than native MSVC; this comparison does not support "MSVC under-measured triad." Both 1-thread numbers are compiler-or-platform limited relative to that Zen 2 expectation.

2. **Triad is the wrong kernel for decode.** STREAM triad is `a[i] = b[i] + scalar*c[i]`: two reads plus a write, and the write incurs read-for-ownership, so it moves about three cache lines per two useful. Decode is read-dominated (weights stream in; almost nothing is written back). A sum-reduction over the same arrays at 6 threads measured **19.42 GB/s** gcc / **17.48 GB/s** MSVC, 38% and 24% above the retired 14.10 triad ceiling, in the 20–40% band a pure-read kernel typically sits above triad. Copy and Scale are in `stream.json` for reference. **The decode ceiling is the 6-thread read-only figure.** Scoring a read-only workload against triad systematically overstates hit rate; that is what produced the 100% row.

3. **TDP.** The 4600H is a 45 W part. 15 W belongs to the 4600U. Low STREAM vs the 51.2 GB/s datasheet peak is not explained by a 15 W limit, and that claim is removed.

Authoritative ceiling used below: gcc/WSL2 6-thread read-only **19.42 GB/s** (`gb_s_median` in `stream.json`). JSON records both toolchains, flags, thread count, array size, pinning, and kernel kind (triad vs read-only).

### STREAM vs thread count

OpenMP, physical-core pin (`tid*2`), 10 runs after 3 warmup, N=1e8 doubles/array. JSON: `bench/results/stream.json`.

**gcc 11.4 / WSL2** (`-O3 -march=native -fopenmp`):

| Threads | Triad GB/s (IQR) | Read-only GB/s (IQR) | Copy | Scale |
|---|---|---|---|---|
| 1 | 8.78 (7.24–10.45) | 7.46 (7.28–7.57) | 7.83 | 11.14 |
| 2 | 7.50 (6.95–7.86) | 13.81 (13.31–14.36) | 6.30 | 7.00 |
| 4 | 11.58 (11.35–11.73) | 16.91 (16.46–17.37) | 10.42 | 10.51 |
| 6 | 11.10 (10.65–11.97) | **19.42** (18.54–19.75) | 10.07 | 9.34 |

**MSVC 19.31** (`/O2 /openmp /arch:AVX2`), native Windows:

| Threads | Triad GB/s (IQR) | Read-only GB/s (IQR) | Copy | Scale |
|---|---|---|---|---|
| 1 | 10.34 (9.61–10.41) | 7.32 (7.18–7.44) | 8.06 | 8.14 |
| 2 | 11.91 (11.43–12.26) | 8.91 (8.07–10.29) | 9.26 | 9.52 |
| 4 | 12.17 (11.49–12.46) | 17.06 (16.35–17.42) | 10.70 | 10.85 |
| 6 | 13.83 (13.61–14.06) | 17.48 (15.47–18.24) | 11.98 | 12.09 |

1-thread read (~7.5 GB/s on both) is a reduction, not a saturated memory controller; it is not used as a roof. 6-thread read is. 19.42 / 51.2 = 38% of datasheet peak. That is the measured read-only roof on this laptop (WSL2 gcc), not a 15 W TDP story. An earlier 22.6 GB/s figure used nontemporal stores and 64 MiB arrays and is retired.

## Methodology (engine runs; unchanged)

- Prompt: `"The capital of France is"` (5–11 tokens depending on tokenizer)
- Temperature 0, seed 42, `--pin`, `--max-seq 2048`, 32 generated tokens
- Llama 3.2 1B Q8 stages: v0 is 1 warmup + 3 runs; everything else is 2 warmup + 5 runs
- Short-prompt prefill is **not** a large GEMM

GGUFs:

| File | Source |
|---|---|
| `Llama-3.2-1B-Instruct-Q8_0.gguf` / `Q4_0.gguf` | `unsloth/Llama-3.2-1B-Instruct-GGUF` (HF weights `9213176726f574b556790deb65791e0c5aa438b6`) |
| `qwen2.5-0.5b-instruct-q8_0.gguf` | `Qwen/Qwen2.5-0.5B-Instruct-GGUF` |
| `qwen2.5-1.5b-instruct-q8_0.gguf` | `Qwen/Qwen2.5-1.5B-Instruct-GGUF` |
| `tinyllama-1.1b-chat-v1.0.Q8_0.gguf` | `TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF` |

## Analytical ceiling

```
decode_tok_s_ceiling = STREAM_read_6t_GB_s * 1e9 / weight_bytes = 19.4187e9 / weight_bytes
```

| Model | Weight bytes | Ceiling tok/s |
|---|---|---|
| Llama 3.2 1B Q8_0 | 1.313 GB | **14.79** |
| Llama 3.2 1B Q4_0 (mixed) | 0.765 GB | 25.38 |
| Qwen2.5 0.5B Q8_0 | 0.670 GB | 28.99 |
| TinyLlama 1.1B Q8_0 | 1.169 GB | 16.61 |
| Qwen2.5 1.5B Q8_0 | 1.889 GB | 10.28 |

## Four Q8_0 models, same code, stage v2, 6 threads

| Model | Decode tok/s | Prefill tok/s | Achieved GB/s | % of STREAM read | % of own ceiling |
|---|---|---|---|---|---|
| Qwen2.5 0.5B | 19.97 | 41.99 | 13.38 | 69% | 69% |
| Llama 3.2 1B | 10.74 | 22.28 | 14.11 | 73% | 73% |
| TinyLlama 1.1B | 9.27 | 22.29 | 10.84 | 56% | 56% |
| Qwen2.5 1.5B | 6.78 | 13.69 | 12.80 | 66% | 66% |

Decode tok/s still tracks 1/weight_size. Against the read-only roof, all four models land **below** 100%, as they must. Llama 3.2 1B is the highest at 73% (14.11 / 19.42 GB/s). Nothing is on the STREAM line. TinyLlama remains the low outlier (56%); smaller vocab (32k) changes the lm_head mix.

JSON: `e2e_qwen25_05b_q8_v2.json`, `e2e_llama32_1b_q8_v2.json`, `e2e_tinyllama_11b_q8_v2.json`, `e2e_qwen25_15b_q8_v2.json`.

## Stage-by-stage — Llama 3.2 1B Instruct Q8_0

v0 here is **scalar Q8_0**, not F32.

| Stage | Decode tok/s | Prefill tok/s | Decode GB/s | % of 14.79 ceiling | JSON |
|---|---|---|---|---|---|
| v0 scalar, 1 thread | 1.16 | 1.41 | 1.53 | 8% | `e2e_llama32_1b_q8_v0.json` |
| v1 6 threads, scalar | 4.43 | 5.96 | 5.82 | 30% | `e2e_llama32_1b_q8_v1.json` |
| v2 6 threads, AVX2 | 10.74 | 22.28 | 14.11 | **73%** | `e2e_llama32_1b_q8_v2.json` |
| v4 + blocking | 10.41 | 12.09 | 13.67 | 70% | `e2e_llama32_1b_q8_v4.json` |
| v5 Q4_0 file + head-major KV | 4.63 | 8.76 | 3.54 | 18% of the *Q4* ceiling | `e2e_llama32_1b_q4_v5.json` |

Deltas worth keeping:

- v0 → v1 decode **3.8×** on 6 threads (not 6×).
- v1 → v2 SIMD: decode **2.4×**, prefill **3.7×**.
- v2 is **73%** of the read-only roof, not on it. The remaining gap is the non-STREAM work (attention, RMSNorm, softmax, sampling, barriers) plus whatever the Q8_0 kernel leaves on the table. Further SIMD/threading can still raise decode tok/s without a fused quant kernel; it is just no longer a free 9×.
- v2 → v4: decode flat (70% vs 73%). Prefill **got worse** (22 → 12 tok/s) at M=6.
- v5 Q4_0 was **slower** than Q8_0 (18% of its own ceiling). Mixed GGUF: `token_embd` is Q6_K, two `ffn_down` tensors are Q4_1, scalar dequant fallback.

v0 → v2 decode speedup: **9.3×** (8% → 73% of ceiling).

## Thread scaling — Llama 3.2 1B Q8_0, stage v2

| Threads | Decode tok/s | Prefill tok/s | Decode GB/s | % of 14.79 ceiling |
|---|---|---|---|---|
| 1 | 3.82 | 5.63 | 5.01 | 26% |
| 2 | 5.08 | 7.88 | 6.67 | 34% |
| 4 | 8.02 | 14.85 | 10.53 | 54% |
| 6 | 10.74 | 22.28 | 14.11 | 73% |

Decode 1→6 is **2.8×**. Prefill 1→6 is **4.0×**. Plot: `bench/results/threads.png`.

## Operator breakdown

Every e2e file has matmul at **99%** of timed ops. The prompt is 5–11 tokens, so attention is noise.

## Context length

Not run.

## Latency (Llama 3.2 1B Q8, v2, 6 threads)

p50 / p95 / p99 decode token latency: **88 / 134 / 168 ms**. TTFT median **269 ms** on a 6-token prompt.
