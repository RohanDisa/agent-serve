# Run e2e benches. Writes UTF-8 JSON under bench/results.
import json, os, subprocess, sys, time
from pathlib import Path

root = Path(__file__).resolve().parents[1]
bin_dir = root / "build" / "Release"
exe = bin_dir / "bench_e2e.exe"
if not exe.exists():
    exe = root / "build" / "bench_e2e"
out_dir = root / "bench" / "results"
out_dir.mkdir(parents=True, exist_ok=True)

jobs = [
    # name, model, stage, threads, n_predict, warmup, runs
    ("llama32_1b_q8_v0", "models/Llama-3.2-1B-Instruct-Q8_0.gguf", "v0", 1, 32, 1, 3),
    ("llama32_1b_q8_v1", "models/Llama-3.2-1B-Instruct-Q8_0.gguf", "v1", 6, 32, 2, 5),
    ("llama32_1b_q8_v2", "models/Llama-3.2-1B-Instruct-Q8_0.gguf", "v2", 6, 32, 2, 5),
    ("llama32_1b_q8_v4", "models/Llama-3.2-1B-Instruct-Q8_0.gguf", "v4", 6, 32, 2, 5),
    ("llama32_1b_q4_v5", "models/Llama-3.2-1B-Instruct-Q4_0.gguf", "v5", 6, 32, 2, 5),
    ("qwen25_05b_q8_v2", "models/qwen2.5-0.5b-instruct-q8_0.gguf", "v2", 6, 32, 2, 5),
    ("tinyllama_11b_q8_v2", "models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf", "v2", 6, 32, 2, 5),
    ("qwen25_15b_q8_v2", "models/qwen2.5-1.5b-instruct-q8_0.gguf", "v2", 6, 32, 2, 5),
    ("llama32_1b_q8_v2_t1", "models/Llama-3.2-1B-Instruct-Q8_0.gguf", "v2", 1, 32, 2, 5),
    ("llama32_1b_q8_v2_t2", "models/Llama-3.2-1B-Instruct-Q8_0.gguf", "v2", 2, 32, 2, 5),
    ("llama32_1b_q8_v2_t4", "models/Llama-3.2-1B-Instruct-Q8_0.gguf", "v2", 4, 32, 2, 5),
]

prompt = "The capital of France is"

def run(name, model, stage, threads, n_predict, warmup, runs):
    json_path = out_dir / f"e2e_{name}.json"
    cmd = [
        str(exe),
        "--model", str(root / model),
        "--stage", stage,
        "--threads", str(threads),
        "--n-predict", str(n_predict),
        "--warmup", str(warmup),
        "--runs", str(runs),
        "--max-seq", "2048",
        "--prompt", prompt,
        "--pin",
        "--out", str(json_path),
    ]
    print(f"\n=== {name} ===", flush=True)
    print(" ".join(cmd), flush=True)
    t0 = time.time()
    p = subprocess.run(cmd, cwd=str(root))
    dt = time.time() - t0
    print(f"exit={p.returncode} elapsed={dt:.1f}s", flush=True)
    if p.returncode != 0:
        raise SystemExit(f"bench failed: {name}")
    if json_path.exists():
        data = json.loads(json_path.read_text(encoding="utf-8", errors="replace"))
        data["job"] = name
        data["model_file"] = model
        json_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        print(
            f"  decode {data.get('decode_tok_s_median'):.2f} tok/s  "
            f"prefill {data.get('prefill_tok_s_median'):.2f} tok/s",
            flush=True,
        )

def main():
    if not exe.exists():
        raise SystemExit(f"missing {exe}")
    only = sys.argv[1:]
    for job in jobs:
        if only and job[0] not in only:
            continue
        run(*job)

if __name__ == "__main__":
    main()
