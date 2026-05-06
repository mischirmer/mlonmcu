#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess
import time
from pathlib import Path


def run(cmd: list[str], capture: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, text=True, capture_output=capture)


def compile_and_run(onnx2c: Path, model: Path, workdir: Path, label: str, onnx2c_args: list[str], iterations: int) -> tuple[float, str]:
    cfile = workdir / f"{label}.c"
    wrapper = workdir / "benchmark_wrapper.c"
    exe = workdir / label

    cmd = [str(onnx2c), "-l", "0"] + onnx2c_args + [str(model)]
    gen = run(cmd, capture=True)
    cfile.write_text(gen.stdout)

    wrapper.write_text(
        "#include <stdint.h>\n"
        "#include <stdio.h>\n"
        "#include <stdlib.h>\n"
        "extern void entry(const int8_t* input, int8_t* output);\n"
        "int main(int argc, char** argv){\n"
        "  int iters = (argc>1)?atoi(argv[1]):10;\n"
        "  size_t in_sz = (size_t)1 << 20;\n"
        "  size_t out_sz = (size_t)1 << 20;\n"
        "  int8_t* input = (int8_t*)calloc(in_sz, 1);\n"
        "  int8_t* output = (int8_t*)calloc(out_sz, 1);\n"
        "  if(!input || !output){ fprintf(stderr, \"alloc failed\\n\"); return 2; }\n"
        "  for(int i=0;i<2;i++) entry(input, output);\n"
        "  for(int i=0;i<iters;i++) entry(input, output);\n"
        "  free(input); free(output);\n"
        "  return 0;\n"
        "}\n"
    )

    run(["gcc", "-O3", "-std=c11", "-o", str(exe), str(cfile), str(wrapper), "-lm"])
    t0 = time.perf_counter()
    proc = run([str(exe), str(iterations)], capture=True)
    t1 = time.perf_counter()
    return (t1 - t0), proc.stdout


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=root / "mlonmcu_env" / "models" / "onnx_aww" / "onnx_aww.onnx")
    ap.add_argument("--onnx2c", type=Path, default=root / "mlonmcu_env" / "deps" / "src" / "onnx2c" / "build" / "onnx2c")
    ap.add_argument("--workdir", type=Path, default=root / "mlonmcu_env" / "results" / "abyzft_host_x86")
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--output", type=Path, default=root / "mlonmcu_env" / "results" / "onnx2c_aww_host_x86_abyzft_timing.csv")
    args = ap.parse_args()

    args.workdir.mkdir(parents=True, exist_ok=True)

    # 1) FAIR OVERHEAD: baseline_raw, baseline, and int16_im2col without timing instrumentation
    baseline_raw_opts = []
    baseline_opts = ["-p", "im2col_all"]
    int16_opts = ["-p", "im2col_all,wide_precision"]
    baseline_raw_rt, _ = compile_and_run(args.onnx2c, args.model, args.workdir, "baseline_raw_no_timing", baseline_raw_opts, args.iterations)
    baseline_rt, _ = compile_and_run(args.onnx2c, args.model, args.workdir, "baseline_no_timing", baseline_opts, args.iterations)
    int16_rt, _ = compile_and_run(args.onnx2c, args.model, args.workdir, "int16_im2col_no_timing", int16_opts, args.iterations)
    baseline_overhead_vs_raw_pct = ((baseline_rt / baseline_raw_rt) - 1.0) * 100.0 if baseline_raw_rt > 0 else float("nan")
    int16_overhead_vs_baseline_pct = ((int16_rt / baseline_rt) - 1.0) * 100.0 if baseline_rt > 0 else float("nan")

    abyzft_opts = ["-p", "im2col_all,wide_precision"]
    abyzft_rt, _ = compile_and_run(args.onnx2c, args.model, args.workdir, "abyzft_no_timing", abyzft_opts + ["--abyzft-gemm"], args.iterations)
    abyzft_overhead_vs_baseline_pct = ((abyzft_rt / baseline_rt) - 1.0) * 100.0 if baseline_rt > 0 else float("nan")
    summary = [
        {
            "label": "baseline_raw",
            "runtime_s": baseline_raw_rt,
            "runtime_per_iter_ms": (baseline_raw_rt / float(args.iterations)) * 1e3,
            "overhead_vs_baseline_raw_pct": 0.0,
            "overhead_vs_baseline_pct": ((baseline_raw_rt / baseline_rt) - 1.0) * 100.0 if baseline_rt > 0 else float("nan"),
        },
        {
            "label": "baseline",
            "runtime_s": baseline_rt,
            "runtime_per_iter_ms": (baseline_rt / float(args.iterations)) * 1e3,
            "overhead_vs_baseline_raw_pct": baseline_overhead_vs_raw_pct,
            "overhead_vs_baseline_pct": 0.0,
        },
        {
            "label": "int16_im2col",
            "runtime_s": int16_rt,
            "runtime_per_iter_ms": (int16_rt / float(args.iterations)) * 1e3,
            "overhead_vs_baseline_raw_pct": ((int16_rt / baseline_raw_rt) - 1.0) * 100.0 if baseline_raw_rt > 0 else float("nan"),
            "overhead_vs_baseline_pct": int16_overhead_vs_baseline_pct,
        },
        {
            "label": "abyzft",
            "runtime_s": abyzft_rt,
            "runtime_per_iter_ms": (abyzft_rt / float(args.iterations)) * 1e3,
            "overhead_vs_baseline_raw_pct": ((abyzft_rt / baseline_raw_rt) - 1.0) * 100.0 if baseline_raw_rt > 0 else float("nan"),
            "overhead_vs_baseline_pct": abyzft_overhead_vs_baseline_pct,
        },
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    print(f"Runtime baseline_raw(no timing): {baseline_raw_rt:.6f}s")
    print(f"Runtime baseline(no timing):     {baseline_rt:.6f}s")
    print(f"Runtime int16_im2col(no timing):  {int16_rt:.6f}s")
    print(f"Baseline vs raw overhead:         {baseline_overhead_vs_raw_pct:+.2f}%")
    print(f"Int16 vs baseline overhead:       {int16_overhead_vs_baseline_pct:+.2f}%")
    print(f"Runtime AByzFT(no timing):        {abyzft_rt:.6f}s")
    print(f"AByzFT vs baseline overhead:      {abyzft_overhead_vs_baseline_pct:+.2f}%")
    print(f"Wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
