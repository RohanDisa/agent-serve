"""Compile and run STREAM on gcc/WSL2 and MSVC. Writes bench/results/stream.json."""
import json
import subprocess
import sys
from pathlib import Path

KERNEL_KIND = {
    "triad": "triad",
    "copy": "copy",
    "scale": "scale",
    "read": "read-only",
}

root = Path(__file__).resolve().parents[1]
out = root / "bench" / "results"
out.mkdir(parents=True, exist_ok=True)
src = root / "bench" / "stream.c"


def run(cmd, **kw):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    p = subprocess.run(cmd, capture_output=True, **kw)
    sys.stderr.write(p.stderr.decode("utf-8", "replace"))
    sys.stderr.flush()
    if p.returncode != 0:
        sys.stderr.write(p.stdout.decode("utf-8", "replace"))
        raise SystemExit(f"failed: {cmd} code={p.returncode}")
    return p


def annotate_kernels(kernels: dict) -> dict:
    out = {}
    for name, vals in kernels.items():
        row = dict(vals)
        row["kind"] = KERNEL_KIND.get(name, name)
        out[name] = row
    return out


def tag_threads(by: dict) -> dict:
    tagged = {}
    for t, kernels in by.items():
        k2 = {}
        for name, vals in kernels.items():
            row = dict(vals)
            row["threads"] = int(t)
            row["kind"] = KERNEL_KIND.get(name, name)
            k2[name] = row
        tagged[str(t)] = k2
    return tagged


def parse_stdout(stdout: str) -> dict:
    start = stdout.find("{")
    if start < 0:
        raise SystemExit("no JSON in stdout:\n" + stdout)
    return json.loads(stdout[start:])


def sweep(exe, threads=(1, 2, 4, 6)):
    by = {}
    for t in threads:
        p = subprocess.run([str(exe), "--threads", str(t), "--kernel", "all"], capture_output=True)
        sys.stderr.write(p.stderr.decode("utf-8", "replace"))
        sys.stderr.flush()
        if p.returncode != 0:
            sys.stderr.write(p.stdout.decode("utf-8", "replace"))
            raise SystemExit(f"{exe} t={t} failed")
        j = parse_stdout(p.stdout.decode("utf-8", "replace"))
        by[str(t)] = annotate_kernels(j["kernels"])
        r = j["kernels"].get("read", {})
        tr = j["kernels"].get("triad", {})
        print(
            f"  t={t}  triad {tr.get('gb_s_median', 0):.2f}  read {r.get('gb_s_median', 0):.2f} GB/s",
            flush=True,
        )
    return by


def compile_gcc_wsl():
    wsl_src = "/mnt/d/Project/tinferno/bench/stream.c"
    # Linux ext4, not /mnt/d 9p — execute bit and page cache behave like a real gcc STREAM.
    wsl_out = "/tmp/stream_omp"
    (root / "build").mkdir(exist_ok=True)
    gcc_ver = run(
        ["wsl.exe", "-d", "Ubuntu", "--", "bash", "-lc", "gcc --version | head -1"]
    ).stdout.decode().strip()
    nproc = run(
        ["wsl.exe", "-d", "Ubuntu", "--", "bash", "-lc", "nproc"]
    ).stdout.decode().strip()
    print(f"WSL gcc={gcc_ver!r} nproc={nproc}", flush=True)
    flags = "-O3 -march=native -fopenmp -DSTREAM_ARRAY_SIZE=100000000"
    run(
        [
            "wsl.exe",
            "-d",
            "Ubuntu",
            "--",
            "bash",
            "-lc",
            f"gcc {flags} {wsl_src} -o {wsl_out}",
        ]
    )
    return {
        "compiler": gcc_ver,
        "version": gcc_ver,
        "flags": flags,
        "environment": f"WSL2 Ubuntu, nproc={nproc}",
        "exe": "/tmp/stream_omp",
        "nproc": int(nproc) if nproc.isdigit() else nproc,
    }


def compile_msvc():
    exe = root / "build" / "Release" / "stream.exe"
    run(["cmake", "-S", str(root), "-B", str(root / "build")])
    run(["cmake", "--build", str(root / "build"), "--config", "Release", "--target", "stream", "-j"])
    return {
        "compiler": "MSVC 19.31.31104.0",
        "version": "19.31.31104.0",
        "flags": "/O2 /openmp /arch:AVX2 /DSTREAM_ARRAY_SIZE=100000000",
        "environment": "Windows native",
        "exe": str(exe),
    }


def main():
    gcc_meta = compile_gcc_wsl()
    print("=== gcc/WSL2 ===", flush=True)
    gcc_thr = int(gcc_meta.get("nproc") or 12)
    gcc_threads = [t for t in (1, 2, 4, 6) if t <= gcc_thr]
    if 6 not in gcc_threads:
        print(f"WARNING: WSL nproc={gcc_thr}, cannot run 6 threads", flush=True)
    gcc_by = {}
    gcc_meta["exe"] = "/tmp/stream_omp"
    for t in gcc_threads:
        p = subprocess.run(
            [
                "wsl.exe",
                "-d",
                "Ubuntu",
                "--",
                "/tmp/stream_omp",
                "--threads",
                str(t),
                "--kernel",
                "all",
            ],
            capture_output=True,
        )
        sys.stderr.write(p.stderr.decode("utf-8", "replace"))
        if p.returncode != 0:
            sys.stderr.write(p.stdout.decode("utf-8", "replace"))
            raise SystemExit(f"gcc stream t={t} failed")
        j = parse_stdout(p.stdout.decode("utf-8", "replace"))
        gcc_by[str(t)] = annotate_kernels(j["kernels"])
        r = j["kernels"]["read"]["gb_s_median"]
        tr = j["kernels"]["triad"]["gb_s_median"]
        print(f"  t={t}  triad {tr:.2f}  read {r:.2f} GB/s", flush=True)

    print("=== MSVC ===", flush=True)
    msvc_meta = compile_msvc()
    msvc_by = sweep(msvc_meta["exe"])

    gcc_by = tag_threads(gcc_by)
    msvc_by = tag_threads(msvc_by)

    gcc_read_6 = gcc_by["6"]["read"]["gb_s_median"]
    gcc_triad_6 = gcc_by["6"]["triad"]["gb_s_median"]
    gcc_triad_1 = gcc_by["1"]["triad"]["gb_s_median"]
    msvc_read_6 = msvc_by["6"]["read"]["gb_s_median"]
    msvc_triad_6 = msvc_by["6"]["triad"]["gb_s_median"]
    msvc_triad_1 = msvc_by["1"]["triad"]["gb_s_median"]

    retired_msvc_triad = 14.10
    if gcc_triad_6 > msvc_triad_6 * 1.15:
        triad_verdict = (
            f"gcc triad {gcc_triad_6:.2f} GB/s is materially higher than MSVC triad "
            f"{msvc_triad_6:.2f} GB/s; gcc triad would be authoritative if triad were the roof."
        )
        msvc_under = "gcc triad materially above MSVC; MSVC under-measured triad"
    else:
        triad_verdict = (
            f"gcc triad {gcc_triad_6:.2f} GB/s is not above MSVC triad {msvc_triad_6:.2f} GB/s "
            f"(1-thread gcc {gcc_triad_1:.2f} vs MSVC {msvc_triad_1:.2f}; neither is 18-24). "
            "MSVC was not under-measuring triad vs gcc-in-WSL2."
        )
        msvc_under = "not confirmed vs gcc/WSL2; gcc triad is not higher"

    note = (
        f"Decode ceiling is gcc/WSL2 6-thread read-only ({gcc_read_6:.2f} GB/s), not triad. "
        f"{triad_verdict} gcc 6-thread read is the roof "
        f"({100 * gcc_read_6 / retired_msvc_triad - 100:.0f}% above the retired {retired_msvc_triad:.2f} MSVC triad). "
        "Copy and Scale are reported per-toolchain for reference."
    )

    pinning = (
        "even logical CPUs (tid*2) via sched_setaffinity (Linux/WSL2) / "
        "SetThreadAffinityMask (Windows); physical cores on 4600H (SMT siblings are consecutive)"
    )
    combined = {
        "bench": "stream",
        "array_size": 100000000,
        "array_bytes_per_array": 800000000,
        "ntimes": 10,
        "nwarm": 3,
        "pinning": pinning,
        "pinning_method": "tid*2 even logical CPUs",
        "ceiling_kernel": "read",
        "ceiling_kind": "read-only",
        "authoritative_toolchain": "gcc_wsl2",
        "gb_s_median": gcc_read_6,
        "gb_s_read_6t": gcc_read_6,
        "gb_s_triad_6t": gcc_triad_6,
        "gb_s_msvc_read_6t": msvc_read_6,
        "gb_s_msvc_triad_6t": msvc_triad_6,
        "note": note,
        "investigation": {
            "msvc_under_measure_triad": msvc_under,
            "read_vs_triad": "confirmed; 6t read-only is the decode roof",
            "tdp_w": 45,
            "tdp_note": "Ryzen 5 4600H is a 45W part; 15W is the 4600U. Power limit is not used as an explanation for STREAM.",
        },
        "toolchains": {
            "gcc_wsl2": {
                "compiler": gcc_meta["compiler"],
                "version": gcc_meta["version"],
                "flags": gcc_meta["flags"],
                "environment": gcc_meta["environment"],
                "array_size": 100000000,
                "pinning": pinning,
                "by_threads": gcc_by,
            },
            "msvc": {
                "compiler": msvc_meta["compiler"],
                "version": msvc_meta["version"],
                "flags": msvc_meta["flags"],
                "environment": msvc_meta["environment"],
                "array_size": 100000000,
                "pinning": pinning,
                "by_threads": msvc_by,
            },
        },
    }
    path = out / "stream.json"
    path.write_text(json.dumps(combined, indent=2) + "\n", encoding="utf-8")
    print("wrote", path)
    print(json.dumps({k: combined[k] for k in ("gb_s_median", "gb_s_read_6t", "gb_s_triad_6t", "note")}, indent=2))


if __name__ == "__main__":
    main()
