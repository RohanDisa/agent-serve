#!/usr/bin/env python3
"""Read bench/results/*.json and write roofline / scaling / context plots."""
import json
import math
import os
import sys

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("pip install matplotlib", file=sys.stderr)
    sys.exit(1)


def load(path):
    raw = open(path, "rb").read()
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        text = raw.decode("utf-16")
    else:
        text = raw.decode("utf-8-sig")
    return json.loads(text)


def find_json(d):
    out = []
    for name in os.listdir(d):
        if name.endswith(".json"):
            try:
                out.append((name, load(os.path.join(d, name))))
            except Exception as e:
                print("skip", name, e, file=sys.stderr)
    return out


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "bench/results"
    files = find_json(d)
    by = {j.get("bench", n): j for n, j in files}
    # also index e2e by filename
    e2e = [(n, j) for n, j in files if j.get("bench") == "e2e"]

    stream = by.get("stream") or by.get("stream_triad")
    gemm = by.get("gemm_f32") or by.get("gemm")
    if stream:
        bw = stream.get("gb_s_read_6t") or stream.get("gb_s_median")
    else:
        bw = None
    peak = gemm["gflop_s_median"] if gemm else None

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    if bw and peak:
        # ridge: y = min(peak, intensity * bw_gbs * 1e9 / 1e9) = min(peak, I * bw)
        xs = [10 ** p for p in [i / 20 for i in range(-20, 41)]]
        ys = [min(peak, x * bw) for x in xs]
        ax.loglog(xs, ys, color="black", lw=1.2, label=f"roofline  STREAM read-only={bw:.1f} GB/s  GEMM={peak:.0f} GFLOP/s")
        ax.axhline(peak, color="0.6", ls="--", lw=0.8)
    for n, j in e2e:
        w = j.get("weight_bytes") or 1
        dec = j.get("decode_tok_s_median") or 0
        pre = j.get("prefill_tok_s_median") or 0
        # decode: 2 FLOP per F32 weight element approximated via bytes; use GB/s point
        # intensity GEMV ~ 2 FLOP / 4 B = 0.5 for F32; for Q8 ~ 2/1 = 2; we plot achieved
        gb = j.get("decode_gb_s") or 0
        # plot decode at I=0.5 (GEMV-ish) and prefill at I=32 (large GEMM-ish) as markers
        if bw and peak and dec > 0:
            # achieved GFLOP/s unknown exactly for quant; plot as % of bandwidth roof
            ax.scatter([0.5], [gb], marker="o", s=60, label=f"decode {j.get('stage')} {dec:.1f} tok/s")
            # prefill: treat as compute-side; put at I=16 with a dummy flop estimate
            ax.scatter([16], [min(peak, pre)], marker="s", s=60, label=f"prefill {j.get('stage')} {pre:.1f} tok/s")
    ax.set_xlabel("arithmetic intensity (FLOP / byte)")
    ax.set_ylabel("GFLOP/s  (decode markers use GB/s on the STREAM slope)")
    ax.set_title("tinferno roofline")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", ls=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(d, "roofline.png"), dpi=140)
    print("wrote", os.path.join(d, "roofline.png"))

    # thread scaling if present
    th = [(n, j) for n, j in e2e if "t" in n]
    if e2e:
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        stages = {}
        for n, j in e2e:
            stages.setdefault(j.get("stage", n), []).append(j)
        for st, js in stages.items():
            js = sorted(js, key=lambda x: x.get("threads", 0))
            ax.plot([x["threads"] for x in js], [x["decode_tok_s_median"] for x in js], "o-", label=f"{st} decode")
            ax.plot([x["threads"] for x in js], [x["prefill_tok_s_median"] for x in js], "s--", label=f"{st} prefill")
        ax.set_xlabel("threads")
        ax.set_ylabel("tok/s")
        ax.set_title("thread scaling")
        ax.legend(fontsize=8)
        ax.grid(True, ls=":", alpha=0.5)
        fig.tight_layout()
        fig.savefig(os.path.join(d, "threads.png"), dpi=140)
        print("wrote", os.path.join(d, "threads.png"))

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ctx = [(n, j) for n, j in files if j.get("bench") == "context"]
    if ctx:
        for n, j in ctx:
            ax.plot(j["ctx"], j["decode_tok_s"], "o-", label=j.get("kv_layout", n))
        ax.set_xlabel("context length")
        ax.set_ylabel("decode tok/s")
        ax.set_title("throughput vs context")
        ax.legend()
        ax.grid(True, ls=":", alpha=0.5)
        fig.tight_layout()
        fig.savefig(os.path.join(d, "context.png"), dpi=140)
        print("wrote", os.path.join(d, "context.png"))
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
