# CPU Inference

A from-scratch C++17 Llama inference engine for CPU. It measures what actually moves prefill and decode performance, and scores decode against a bandwidth ceiling derived from the workload rather than from a datasheet.

## Design

Prefill and decode are separate code paths (`Model::prefill` and `Model::decode`) because they sit on opposite sides of the roofline:

| Phase | Behaviour | Bound by |
| --- | --- | --- |
| Prefill | Weight reads are amortized across many tokens | Compute |
| Decode | Every weight is read for every token | Memory bandwidth |

## Key findings

- **Decode is bandwidth-bound.** Across a 3x range in model size, decode throughput tracks 1 / weight size while achieved bandwidth stays within 10.8 to 14.1 GB/s.
- **Best result: 73% of the bandwidth ceiling** (Llama 3.2 1B Q8_0, AVX2, 6 threads), a 9.3x decode improvement over the scalar single-thread baseline.
- **SIMD and threading help prefill more than decode**, because faster compute saturates memory with fewer threads.
- **Cache blocking did not help** decode and hurt short-prompt prefill.
- **The Q4 file decoded slower than Q8_0**, because mixed-precision tensors fall back to a scalar path and leave the memory-bound regime.

## Results

### Test system

| Item | Value |
| --- | --- |
| CPU | AMD Ryzen 5 4600H (6 cores / 12 threads, 45 W) |
| Memory | 2x8 GB DDR4-3200, 51.2 GB/s theoretical peak |
| Decode ceiling | **19.42 GB/s** (6-thread read-only STREAM, gcc 11.4, WSL2) |
| F32 GEMM | 83 GFLOP/s |

The ceiling uses a read-only STREAM kernel, not Triad. Decode streams weights in and writes almost nothing back, so a read-plus-write benchmark would understate the available bandwidth.

Full tables, gcc and MSVC STREAM figures, and raw JSON are in [`RESULTS.md`](RESULTS.md) and [`bench/results/`](bench/results/).

### Model comparison

Same binary, AVX2, 6 pinned threads, 32 generated tokens.

| Model | Decode tok/s | Prefill tok/s | Achieved GB/s | % of ceiling |
| --- | --- | --- | --- | --- |
| Llama 3.2 1B Q8_0 | 10.7 | 22.3 | 14.1 | 73% |
| Qwen2.5 0.5B Q8_0 | 20.0 | 42.0 | 13.4 | 69% |
| Qwen2.5 1.5B Q8_0 | 6.8 | 13.7 | 12.8 | 66% |
| TinyLlama 1.1B Q8_0 | 9.3 | 22.3 | 10.8 | 56% |

No model reaches the STREAM ceiling. Each decode step still pays for attention, RMSNorm, softmax, sampling, and a thread barrier on every matmul.

### Optimization stages

Llama 3.2 1B Instruct Q8_0. The decode ceiling for this model is 14.79 tok/s.

| Stage | Decode tok/s | Prefill tok/s | % of ceiling |
| --- | --- | --- | --- |
| v0: scalar, 1 thread | 1.16 | 1.41 | 8% |
| v1: scalar, 6 threads | 4.43 | 5.96 | 30% |
| v2: AVX2, 6 threads | **10.74** | 22.28 | **73%** |
| v4: + cache blocking | 10.41 | 12.09 | 70% |
| v5: mixed Q4 GGUF + head-major KV | 4.63 | 8.76 | 18% (of Q4 ceiling) |

**Scaling by optimization:**

| Change | Decode speedup | Prefill speedup |
| --- | --- | --- |
| 1 to 6 threads | 2.8x | 4.0x |
| Scalar to AVX2 | 2.4x | 3.7x |

**Cache blocking:** no decode gain, and a prefill regression on short prompts. At M=6 there is no GEMM reuse for a blocked kernel to exploit.

## The Q4 anomaly

The unsloth Llama 3.2 1B file labelled Q4_0 is 40% smaller than Q8_0 but decoded slower.

| Metric | Q8_0 | Q4 file |
| --- | --- | --- |
| % of own ceiling | 73% | 18% |
| Achieved bandwidth | 14.1 GB/s | 3.5 GB/s |

**Cause.** The file is not uniformly Q4_0:

- `token_embd` (the tied lm_head, read in full every token) stays at Q6_K.
- Two `ffn_down` tensors are Q4_1.

Both formats use a scalar dequant path in this engine, which moves the inner loop from bandwidth-bound to ALU-bound.

**What this adds.** The llama.cpp community already documents that K-quants carry more per-block work and that mixed files keep the output tensor at higher precision. A tok/s drop alone has several possible explanations. The bandwidth collapse from 14.1 to 3.5 GB/s, while every other configuration stays above 10 GB/s, pins it specifically on the kernel leaving the memory-bound regime.

## Build

Requires C++17 and CMake 3.16 or newer.

**Linux / macOS:**

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

**Windows (MSVC 2022 x64):**

```powershell
scripts/build.ps1
```

Binaries are written to `build/tinferno` or `build/Release/tinferno.exe`. CI builds and runs the unit tests on Ubuntu.

## Run

Download a model:

```bash
huggingface-cli download unsloth/Llama-3.2-1B-Instruct-GGUF \
  Llama-3.2-1B-Instruct-Q8_0.gguf --local-dir models
```

Generate:

```bash
./build/tinferno --model models/Llama-3.2-1B-Instruct-Q8_0.gguf \
  --prompt "Write a haiku about memory bandwidth." \
  --stage v2 --n-predict 64 --temp 0 --pin --profile
```

### Flags

| Flag | Purpose |
| --- | --- |
| `--model PATH` | GGUF model file |
| `--prompt TEXT` | Input prompt |
| `--stage v0..v5` | Optimization preset (see [Stages](#stages)) |
| `--n-predict N` | Tokens to generate |
| `--temp T` | Sampling temperature |
| `--threads N` | Thread count |
| `--pin` | Pin threads to cores |
| `--profile` | Print timing breakdown |
| `--kv-layout seq\|head` | KV cache layout |
| `--tokens 1,2,3` | Supply token IDs directly, skipping the tokenizer |
| `--max-seq N` | Maximum sequence length |

### Supported formats

| Format | Path |
| --- | --- |
| F32, F16, Q8_0, Q4_0 | Native kernels |
| Q4_1, Q6_K | Scalar dequant fallback (mixed files) |

Qwen2 is handled as a Llama-style decoder with Q/K/V bias.

## Stages

`--stage` selects a preset within a single binary, not a separate build.

| Stage | Threads | SIMD | Cache blocking | KV layout |
| --- | --- | --- | --- | --- |
| v0 | 1 | off | off | seq-major |
| v1 | all | off | off | seq-major |
| v2 / v3 | all | AVX2 | off | seq-major |
| v4 | all | AVX2 | on | seq-major |
| v5 | all | AVX2 | on | head-major |

- v2 and v3 share a code path.
- Quantization comes from the GGUF file, not from the stage flag.
- **Correctness gate:** temperature-0 output must match across all stages. CI checks this on a small synthetic GGUF, so no 1B download is needed.

## Measurement methodology

All published figures are medians.

- Threads pinned with `--pin`, SMT left enabled. STREAM pins OpenMP threads to physical cores.
- Warmup runs discarded. Median and interquartile range reported, never best-of-N.
- Temperature 0, seed 42.
- Prefill and decode timed separately. Time to first token equals prompt prefill time.
- Decode latency reported at p50, p95, and p99.
- Machine, compiler, and flags recorded alongside every table in `RESULTS.md`.

### Decode ceiling

```
tok/s_ceiling = STREAM_read_6t_GB/s * 1e9 / weight_bytes
```

STREAM, GEMM, and end-to-end ceilings are stored as JSON under `bench/results/`. To regenerate STREAM across both toolchains and kernel types:

```powershell
python scripts/run_stream.py
python bench/roofline.py bench/results
```

## Limitations

- **Prefill figures use a short prompt**, where prefill and decode behave similarly. Claims about prefill at realistic lengths need a rerun at 512 tokens or more.
- Results come from a single machine (Ryzen 5 4600H, DDR4-3200).
- Mixed-precision GGUF files fall back to scalar kernels for Q4_1 and Q6_K.
