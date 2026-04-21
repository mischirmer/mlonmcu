#!/usr/bin/env python3
import argparse
import csv
import datetime as dt
import math
import os
import re
import subprocess
import tempfile
from pathlib import Path
try:
    from tqdm import tqdm
except Exception:
    tqdm = None


LAYER_RE = re.compile(r"abft_log_rowcol_delta\s+layer=(\d+)\s+rowMaxDelta=([^\s]+)\s+colMaxDelta=([^\s]+)")


def run_cmd(cmd, cwd):
    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    return proc.returncode, proc.stdout, proc.stderr


def run_cmd_with_retries(cmd, cwd, retries):
    attempts = max(1, retries)
    last = None
    for attempt in range(1, attempts + 1):
        rc, so, se = run_cmd(cmd, cwd)
        last = (rc, so, se, attempt)
        if rc == 0:
            return last
    return last


def read_log(path):
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def parse_layers_and_max_delta(log_text):
    layers = set()
    max_abs = 0.0
    for m in LAYER_RE.finditer(log_text):
        layer = int(m.group(1))
        layers.add(layer)
        for raw in (m.group(2), m.group(3)):
            try:
                val = abs(float(raw))
                if val > max_abs:
                    max_abs = val
            except ValueError:
                pass
    return (max(layers) + 1) if layers else 0, max_abs


def parse_per_layer_deltas(log_text):
    # Last occurrence per layer wins.
    by_layer = {}
    for m in LAYER_RE.finditer(log_text):
        layer = int(m.group(1))
        row_raw = m.group(2)
        col_raw = m.group(3)
        try:
            row_val = float(row_raw)
        except ValueError:
            row_val = math.nan
        try:
            col_val = float(col_raw)
        except ValueError:
            col_val = math.nan
        by_layer[layer] = (row_val, col_val)
    return by_layer


def merge_per_layer_deltas(*texts):
    merged = {}
    for text in texts:
        if text:
            merged.update(parse_per_layer_deltas(text))
    return merged


def summarize_per_layer(per_layer):
    if not per_layer:
        return 0, 0.0, 0.0, 0.0
    row_vals = [abs(v[0]) for v in per_layer.values() if not math.isnan(v[0])]
    col_vals = [abs(v[1]) for v in per_layer.values() if not math.isnan(v[1])]
    row_max = max(row_vals) if row_vals else 0.0
    col_max = max(col_vals) if col_vals else 0.0
    both_max = max([row_max, col_max]) if (row_vals or col_vals) else 0.0
    return len(per_layer), row_max, col_max, both_max


def write_run_output(out_dir, label, stdout_text, stderr_text):
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", label)
    out_path = out_dir / f"{safe}.stdout.log"
    err_path = out_dir / f"{safe}.stderr.log"
    out_path.write_text(stdout_text or "", encoding="utf-8")
    err_path.write_text(stderr_text or "", encoding="utf-8")
    return out_path, err_path


def find_host_log_by_label(mlonmcu_home, run_label):
    sessions_root = Path(mlonmcu_home) / "temp" / "sessions"
    if not sessions_root.exists():
        return None
    # Find the session created for this specific run label.
    label_files = list(sessions_root.glob("*/label.txt"))
    matching_sessions = []
    for lf in label_files:
        try:
            txt = lf.read_text(encoding="utf-8", errors="ignore").strip()
        except Exception:
            continue
        if txt == run_label:
            matching_sessions.append(lf.parent)
    if not matching_sessions:
        return None
    session_dir = max(matching_sessions, key=lambda p: p.stat().st_mtime)
    # Typical path per run invocation.
    candidate = session_dir / "runs" / "0" / "host_x86_out.log"
    if candidate.exists():
        return candidate
    # Fallback: any run log under this session.
    all_candidates = list(session_dir.glob("runs/*/host_x86_out.log"))
    if not all_candidates:
        return None
    return max(all_candidates, key=lambda p: p.stat().st_mtime)


def build_plan(models, modes, faults):
    plan = []
    for dtype, model in models:
        for mode in modes:
            # discovery run
            plan.append(
                {
                    "dtype": dtype,
                    "model": model,
                    "mode": mode,
                    "kind": "discover",
                    "pattern": "none",
                    "delta": 0,
                    "target_layer": -1,
                }
            )
            # FI runs (layer will be expanded at runtime once discovered)
            for pattern, delta in faults:
                plan.append(
                    {
                        "dtype": dtype,
                        "model": model,
                        "mode": mode,
                        "kind": "fi_template",
                        "pattern": pattern,
                        "delta": delta,
                        "target_layer": None,
                    }
                )
    return plan


def preflight_mlonmcu(repo_root):
    rc, so, se = run_cmd(
        ["mlonmcu", "flow", "--home", os.environ["MLONMCU_HOME"], "--list-targets"],
        cwd=repo_root,
    )
    if rc != 0:
        return False, f"mlonmcu command failed in preflight.\nSTDERR:\n{se}\nSTDOUT:\n{so}"
    return True, ""


def parse_int_csv(values_csv, name):
    values = []
    for tok in values_csv.split(","):
        t = tok.strip()
        if not t:
            continue
        try:
            values.append(int(t))
        except ValueError as err:
            raise RuntimeError(f"Invalid integer in --{name}: '{t}'") from err
    if not values:
        raise RuntimeError(f"--{name} must contain at least one integer value")
    return values


def parse_failed_runs_from_csv(csv_path):
    rows = list(csv.DictReader(Path(csv_path).open("r", encoding="utf-8")))
    failed = [r for r in rows if int(r.get("return_code", "0")) != 0]
    parsed = []
    for r in failed:
        parsed.append(
            {
                "dtype": r["dtype"],
                "model": r["model"],
                "mode": r["mode"],
                "pattern": r["pattern"],
                "delta": int(r["delta"]),
                "target_layer": int(r["target_layer"]),
            }
        )
    return parsed


def count_failures_in_csv(csv_path):
    rows = list(csv.DictReader(Path(csv_path).open("r", encoding="utf-8")))
    return sum(1 for r in rows if int(r.get("return_code", "0")) != 0), len(rows)


def merge_summary_csv(base_csv, update_csv, out_csv):
    base_rows = list(csv.DictReader(Path(base_csv).open("r", encoding="utf-8")))
    update_rows = list(csv.DictReader(Path(update_csv).open("r", encoding="utf-8")))
    key_fields = ["dtype", "model", "mode", "pattern", "delta", "target_layer"]

    def key_of(row):
        return tuple(row.get(k, "") for k in key_fields)

    merged = {key_of(r): r for r in base_rows}
    for r in update_rows:
        merged[key_of(r)] = r

    # Preserve stable ordering from base, append new keys at end.
    ordered = []
    seen = set()
    for r in base_rows:
        k = key_of(r)
        if k in merged and k not in seen:
            ordered.append(merged[k])
            seen.add(k)
    for r in update_rows:
        k = key_of(r)
        if k not in seen:
            ordered.append(merged[k])
            seen.add(k)

    fieldnames = list(base_rows[0].keys()) if base_rows else list(update_rows[0].keys())
    with Path(out_csv).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(ordered)


def merge_per_layer_csv(base_csv, update_csv, out_csv):
    base_rows = list(csv.DictReader(Path(base_csv).open("r", encoding="utf-8")))
    update_rows = list(csv.DictReader(Path(update_csv).open("r", encoding="utf-8")))
    key_fields = [
        "dtype",
        "model",
        "mode",
        "run_pattern",
        "run_delta",
        "run_target_layer",
        "logged_layer",
    ]

    def key_of(row):
        return tuple(row.get(k, "") for k in key_fields)

    merged = {key_of(r): r for r in base_rows}
    for r in update_rows:
        merged[key_of(r)] = r

    ordered = []
    seen = set()
    for r in base_rows:
        k = key_of(r)
        if k in merged and k not in seen:
            ordered.append(merged[k])
            seen.add(k)
    for r in update_rows:
        k = key_of(r)
        if k not in seen:
            ordered.append(merged[k])
            seen.add(k)

    fieldnames = list(base_rows[0].keys()) if base_rows else list(update_rows[0].keys())
    with Path(out_csv).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(ordered)


def parse_optional_int_csv(values_csv, name):
    if values_csv is None:
        return None
    values = parse_int_csv(values_csv, name)
    return sorted(set(values))


def generate_heatmaps(run_rows, per_layer_rows, out_csv):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as ex:
        print(f"WARNING: matplotlib unavailable, skipping heatmap generation: {ex}")
        return

    mode_order = {"baseline": 0, "abft": 1, "abyzft": 2}
    pattern_order = {"none": 0, "single_point": 1, "trivial": 2}

    for dtype in sorted(set(r["dtype"] for r in run_rows)):
        deltas_by_run_layer = {}
        logged_layers = sorted(
            {
                int(r["logged_layer"])
                for r in per_layer_rows
                if r.get("dtype") == dtype
            }
        )
        for r in per_layer_rows:
            if r.get("dtype") != dtype:
                continue
            key = (r["label"], int(r["logged_layer"]))
            deltas_by_run_layer[key] = (float(r["row_max_delta"]), float(r["col_max_delta"]))

        if not logged_layers:
            print(f"WARNING: no per-layer deltas found for {dtype}; skipping heatmap generation")
            continue

        dtype_runs = [r for r in run_rows if r["dtype"] == dtype]
        dtype_runs = sorted(
            dtype_runs,
            key=lambda x: (
                mode_order.get(x["mode"], 9),
                pattern_order.get(x["pattern"], 9),
                int(x["delta"]),
                int(x["target_layer"]),
                x["label"],
            ),
        )
        if not dtype_runs:
            continue

        row_mat = np.full((len(dtype_runs), len(logged_layers)), np.nan, dtype=float)
        col_mat = np.full((len(dtype_runs), len(logged_layers)), np.nan, dtype=float)

        ylabels = []
        for i, rr in enumerate(dtype_runs):
            ylabels.append(f'{rr["mode"]}|{rr["pattern"]}|d{rr["delta"]}|t{rr["target_layer"]}')
            for j, ly in enumerate(logged_layers):
                vals = deltas_by_run_layer.get((rr["label"], ly))
                if vals is not None:
                    row_mat[i, j] = vals[0]
                    col_mat[i, j] = vals[1]

        fig, axes = plt.subplots(1, 2, figsize=(16, max(6, len(dtype_runs) * 0.25)), squeeze=False)
        for ax, mat, title in [
            (axes[0, 0], row_mat, "Row Max Delta"),
            (axes[0, 1], col_mat, "Col Max Delta"),
        ]:
            im = ax.imshow(mat, aspect="auto", cmap="viridis")
            ax.set_title(f"{dtype} - {title}")
            ax.set_xlabel("logged_layer")
            ax.set_ylabel("mode|pattern|delta|target_layer")
            ax.set_xticks(range(len(logged_layers)))
            ax.set_xticklabels([str(x) for x in logged_layers])
            ax.set_yticks(range(len(dtype_runs)))
            ax.set_yticklabels(ylabels, fontsize=7)
            # Annotate each cell with its numeric value and choose text color
            # based on background luminance for readability.
            cmap = im.cmap
            norm = im.norm
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    val = mat[i, j]
                    if np.isnan(val):
                        continue
                    rgba = cmap(norm(val))
                    # Perceived luminance in sRGB space.
                    luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                    txt_color = "black" if luminance > 0.55 else "white"
                    ax.text(
                        j,
                        i,
                        f"{val:.2f}",
                        ha="center",
                        va="center",
                        color=txt_color,
                        fontsize=6,
                    )
            cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
            cbar.set_label("delta")
        fig.tight_layout()
        out_png = out_csv.with_name(f"{out_csv.stem}_{dtype}_rowcol_heatmap.png")
        fig.savefig(out_png, dpi=200)
        plt.close(fig)
        print(f"Wrote heatmap: {out_png}")


def mk_base_cmd(args, model, mode):
    cmd = [
        "mlonmcu",
        "flow",
        "run",
        model,
        "--home",
        str(args.mlonmcu_home_resolved),
        "--backend",
        args.backend,
        "--target",
        args.target,
        "--config",
        f"{args.backend}.compiler_mode={mode}",
    ]
    if mode == "abft":
        cmd += ["--config", f"{args.backend}.abft_enable_analysis=true"]
    elif mode == "abyzft":
        cmd += ["--config", f"{args.backend}.abyzft_enable_analysis=true"]
    return cmd


def add_fault_flags(cmd, args, mode, enabled, pattern, delta, layer):
    prefix = "abyzft" if mode == "abyzft" else "abft"
    cmd += ["--config", f"{args.backend}.{prefix}_inject_fault={'true' if enabled else 'false'}"]
    cmd += ["--config", f"{args.backend}.{prefix}_inject_fault_pattern={pattern}"]
    cmd += ["--config", f"{args.backend}.{prefix}_inject_fault_delta={delta}"]
    cmd += ["--config", f"{args.backend}.{prefix}_inject_fault_layer={layer}"]


def add_abyzft_defaults(cmd, args):
    # Keep defaults explicit for reproducibility.
    cmd += ["--config", f"{args.backend}.abyzft_scale_sampling_mode={args.abyzft_scale_sampling_mode}"]
    cmd += ["--config", f"{args.backend}.abyzft_float_disjoint_min_abs={args.abyzft_float_disjoint_min_abs}"]
    cmd += ["--config", f"{args.backend}.abyzft_float_disjoint_max_abs={args.abyzft_float_disjoint_max_abs}"]
    cmd += ["--config", f"{args.backend}.abyzft_float_range_min={args.abyzft_float_range_min}"]
    cmd += ["--config", f"{args.backend}.abyzft_float_range_max={args.abyzft_float_range_max}"]
    cmd += ["--config", f"{args.backend}.abyzft_float_discrete_list={args.abyzft_float_discrete_list}"]
    cmd += ["--config", f"{args.backend}.abyzft_int_discrete_list={args.abyzft_int_discrete_list}"]
    cmd += ["--config", f"{args.backend}.abyzft_int_range_min={args.abyzft_int_range_min}"]
    cmd += ["--config", f"{args.backend}.abyzft_int_range_max={args.abyzft_int_range_max}"]
    cmd += ["--config", f"{args.backend}.abyzft_int_bits_max={args.abyzft_int_bits_max}"]


def main():
    ap = argparse.ArgumentParser(description="Sweep baseline/ABFT/AByzFT with per-layer fault injection for int and fp32 models.")
    ap.add_argument("--repo-root", default=".", help="Path to mlonmcu repository root")
    ap.add_argument("--backend", default="ireellvmc")
    ap.add_argument("--target", default="host_x86")
    ap.add_argument("--int-model", default="aww")
    ap.add_argument("--fp-model", default="sine_model")
    ap.add_argument(
        "--host-log",
        default="mlonmcu_env/temp/sessions/latest/runs/latest/host_x86_out.log",
        help="Path to runtime host log used for layer/delta parsing",
    )
    ap.add_argument("--output-csv", default="ipynb/abft_minimal/out/fi_sweep_results.csv")
    ap.add_argument("--replot-only", action="store_true", help="Only regenerate heatmaps from existing CSV files")
    ap.add_argument("--mlonmcu-home", default=None, help="Optional MLONMCU_HOME override")
    ap.add_argument(
        "--retry-failed-from-csv",
        default=None,
        help="Retry only failed runs from an existing sweep CSV (path to fi_sweep_results.csv)",
    )
    ap.add_argument(
        "--retry-until-clean",
        action="store_true",
        help="Keep retrying failed runs until no failures remain (or max rounds reached)",
    )
    ap.add_argument(
        "--max-retry-rounds",
        type=int,
        default=10,
        help="Maximum rounds for --retry-until-clean (default: 10)",
    )
    ap.add_argument(
        "--max-retries-per-run",
        type=int,
        default=1,
        help="Max attempts per run command (default: 1, set >1 to retry flaky compile failures)",
    )
    ap.add_argument(
        "--fault-values",
        default="0,2,4",
        help="Comma-separated fault delta values used for sweep (default: 0,2,4)",
    )
    ap.add_argument(
        "--int-layers",
        default=None,
        help="Comma-separated target layers for int model injections (default: auto-discover all)",
    )
    ap.add_argument(
        "--fp-layers",
        default=None,
        help="Comma-separated target layers for fp32 model injections (default: auto-discover all)",
    )
    ap.add_argument("--stop-on-error", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--abyzft-scale-sampling-mode", type=int, default=1)
    ap.add_argument("--abyzft-float-disjoint-min-abs", type=float, default=0.5)
    ap.add_argument("--abyzft-float-disjoint-max-abs", type=float, default=2.0)
    ap.add_argument("--abyzft-float-range-min", type=float, default=-2.0)
    ap.add_argument("--abyzft-float-range-max", type=float, default=2.0)
    ap.add_argument("--abyzft-float-discrete-list", default="-8,-4,-2,2,4,8")
    ap.add_argument("--abyzft-int-discrete-list", default="1,2,4")
    ap.add_argument("--abyzft-int-range-min", type=int, default=1)
    ap.add_argument("--abyzft-int-range-max", type=int, default=8)
    ap.add_argument("--abyzft-int-bits-max", type=int, default=2)
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    host_log = (repo_root / args.host_log).resolve()
    out_csv = (repo_root / args.output_csv).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    runlogs_dir = out_csv.parent / f"{out_csv.stem}_runlogs"
    runlogs_dir.mkdir(parents=True, exist_ok=True)

    if args.retry_until_clean and args.replot_only:
        raise RuntimeError("--retry-until-clean cannot be combined with --replot-only")

    if args.retry_until_clean:
        # Iteratively retry failures by recursively invoking this script in
        # retry-only mode and replacing CSVs with each round's outputs.
        if not out_csv.exists():
            raise RuntimeError(
                f"--retry-until-clean needs an existing CSV as seed: {out_csv}"
            )
        per_layer_csv = out_csv.with_name(f"{out_csv.stem}_per_layer.csv")
        if not per_layer_csv.exists():
            raise RuntimeError(
                f"--retry-until-clean needs an existing per-layer CSV: {per_layer_csv}"
            )
        for round_idx in range(1, max(1, args.max_retry_rounds) + 1):
            failed, total = count_failures_in_csv(out_csv)
            print(f"[retry-until-clean] round {round_idx}: failures={failed}/{total}")
            if failed == 0:
                print("[retry-until-clean] completed with zero failures")
                return
            tmp_out = Path(tempfile.gettempdir()) / f"fi_retry_round_{os.getpid()}_{round_idx}.csv"
            cmd = [
                "python3",
                str(Path(__file__).resolve()),
                "--repo-root",
                str(repo_root),
                "--output-csv",
                str(tmp_out),
                "--retry-failed-from-csv",
                str(out_csv),
                "--max-retries-per-run",
                str(args.max_retries_per_run),
                "--fault-values",
                args.fault_values,
            ]
            if args.mlonmcu_home:
                cmd += ["--mlonmcu-home", args.mlonmcu_home]
            if args.int_layers:
                cmd += ["--int-layers", args.int_layers]
            if args.fp_layers:
                cmd += ["--fp-layers", args.fp_layers]
            rc, so, se = run_cmd(cmd, cwd=repo_root)
            if rc != 0:
                raise RuntimeError(
                    "[retry-until-clean] retry round invocation failed.\n"
                    f"CMD: {' '.join(cmd)}\nSTDOUT:\n{so}\nSTDERR:\n{se}"
                )
            tmp_per = tmp_out.with_name(f"{tmp_out.stem}_per_layer.csv")
            if not tmp_out.exists() or not tmp_per.exists():
                raise RuntimeError("[retry-until-clean] retry round did not produce expected CSV outputs")
            # Merge retry results back into full dataset (replace retried configs only).
            merge_summary_csv(out_csv, tmp_out, out_csv)
            merge_per_layer_csv(per_layer_csv, tmp_per, per_layer_csv)
        failed, total = count_failures_in_csv(out_csv)
        raise RuntimeError(
            f"[retry-until-clean] exhausted {args.max_retry_rounds} rounds with failures still present: {failed}/{total}"
        )

    if args.replot_only:
        per_layer_csv = out_csv.with_name(f"{out_csv.stem}_per_layer.csv")
        if not out_csv.exists():
            raise RuntimeError(f"Missing summary CSV for replot: {out_csv}")
        if not per_layer_csv.exists():
            raise RuntimeError(f"Missing per-layer CSV for replot: {per_layer_csv}")
        with out_csv.open("r", newline="", encoding="utf-8") as f:
            run_rows = list(csv.DictReader(f))
        with per_layer_csv.open("r", newline="", encoding="utf-8") as f:
            per_layer_rows = list(csv.DictReader(f))
        print(f"Loaded {len(run_rows)} rows from: {out_csv}")
        print(f"Loaded {len(per_layer_rows)} per-layer rows from: {per_layer_csv}")
        generate_heatmaps(run_rows, per_layer_rows, out_csv)
        return

    if args.mlonmcu_home:
        args.mlonmcu_home_resolved = Path(args.mlonmcu_home).resolve()
    else:
        default_home = repo_root / "mlonmcu_env"
        args.mlonmcu_home_resolved = default_home.resolve() if default_home.exists() else None

    if args.mlonmcu_home_resolved is None:
        raise RuntimeError(
            "Could not resolve MLonMCU home automatically. "
            "Please pass --mlonmcu-home <path containing environment.yml>."
        )

    env_yml = args.mlonmcu_home_resolved / "environment.yml"
    if not env_yml.exists():
        raise RuntimeError(
            f"MLonMCU home does not contain environment.yml: {env_yml}"
        )

    os.environ["MLONMCU_HOME"] = str(args.mlonmcu_home_resolved)

    if not args.dry_run:
        ok, reason = preflight_mlonmcu(repo_root)
        if not ok:
            raise RuntimeError(
                "Preflight failed before sweep. MLonMCU environment is not usable.\n"
                f"{reason}\n"
                "Tip: activate your environment and/or pass --mlonmcu-home <path>."
            )

    models = [("int8", args.int_model), ("fp32", args.fp_model)]
    modes = ["abft", "abyzft"]
    fault_values = parse_int_csv(args.fault_values, "fault-values")
    fault_patterns = ["single_point", "trivial"]
    faults = [(pattern, delta) for pattern in fault_patterns for delta in fault_values]
    int_layers_override = parse_optional_int_csv(args.int_layers, "int-layers")
    fp_layers_override = parse_optional_int_csv(args.fp_layers, "fp-layers")

    rows = []
    per_layer_rows = []
    plan = build_plan(models, modes, faults)
    total_discover = sum(1 for x in plan if x["kind"] == "discover")
    total_templates = sum(1 for x in plan if x["kind"] == "fi_template")
    print(f"Planned runs: {total_discover} discovery + layer-expanded FI from {total_templates} templates")
    completed = 0
    planned_total = total_discover
    pbar = (
        tqdm(total=planned_total, desc="ABFT/AByzFT FI sweep", unit="run", dynamic_ncols=True)
        if tqdm
        else None
    )

    def progress_line(extra=""):
        if pbar is not None:
            pbar.set_postfix_str(extra)
        else:
            width = 30
            ratio = 0 if planned_total == 0 else min(1.0, completed / planned_total)
            filled = int(width * ratio)
            bar = "#" * filled + "-" * (width - filled)
            print(f"[{bar}] {completed}/{planned_total} {extra}")

    retry_only_runs = None
    if args.retry_failed_from_csv:
        retry_csv = Path(args.retry_failed_from_csv).resolve()
        if not retry_csv.exists():
            raise RuntimeError(f"--retry-failed-from-csv file not found: {retry_csv}")
        retry_only_runs = parse_failed_runs_from_csv(retry_csv)
        print(f"Retry-only mode: loaded {len(retry_only_runs)} failed runs from {retry_csv}")

    for dtype, model in models:
        for mode in modes:
            mode_retry_runs = None
            if retry_only_runs is not None:
                mode_retry_runs = [
                    rr for rr in retry_only_runs
                    if rr["dtype"] == dtype and rr["model"] == model and rr["mode"] == mode
                ]
                if not mode_retry_runs:
                    continue
            progress_line(f"discover {dtype}/{mode}")
            discover_cmd = mk_base_cmd(args, model, mode)
            if mode == "abyzft":
                add_abyzft_defaults(discover_cmd, args)
            add_fault_flags(discover_cmd, args, mode, enabled=False, pattern="single_point", delta=0, layer=-1)
            discover_cmd += ["-l", f"discover_layers_{dtype}_{mode}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"]

            discover_label = discover_cmd[-1]
            if args.dry_run:
                print("DRY-RUN:", " ".join(discover_cmd))
                num_layers = 0 if mode == "baseline" else 1
                max_delta = 0.0
                per_layer = {}
                rc = 0
                stdout_log = ""
                stderr_log = ""
            else:
                rc, _so, _se, attempt_used = run_cmd_with_retries(
                    discover_cmd, cwd=repo_root, retries=args.max_retries_per_run
                )
                host_log_for_run = find_host_log_by_label(
                    args.mlonmcu_home_resolved, discover_label
                )
                host_text = read_log(host_log_for_run) if host_log_for_run else ""
                per_layer = merge_per_layer_deltas(_so, _se, host_text)
                num_layers, row_max, col_max, max_delta = summarize_per_layer(per_layer)
                out_path, err_path = write_run_output(
                    runlogs_dir, discover_label, _so, _se
                )
                stdout_log = str(out_path)
                stderr_log = str(err_path)
            rows.append(
                {
                    "dtype": dtype,
                    "model": model,
                    "mode": mode,
                    "pattern": "none",
                    "delta": 0,
                    "target_layer": -1,
                    "return_code": rc,
                    "detected_layers": num_layers,
                    "row_max_logged_delta": row_max if not args.dry_run else 0.0,
                    "col_max_logged_delta": col_max if not args.dry_run else 0.0,
                    "max_abs_logged_delta": max_delta,
                    "label": discover_label,
                    "stdout_log": stdout_log,
                    "stderr_log": stderr_log,
                    "attempts_used": attempt_used if not args.dry_run else 0,
                }
            )
            for logged_layer, (row_delta, col_delta) in sorted(per_layer.items()):
                per_layer_rows.append(
                    {
                        "dtype": dtype,
                        "model": model,
                        "mode": mode,
                        "label": discover_label,
                        "run_pattern": "none",
                        "run_delta": 0,
                        "run_target_layer": -1,
                        "logged_layer": logged_layer,
                        "row_max_delta": row_delta,
                        "col_max_delta": col_delta,
                    }
                )
            if rc != 0 and args.stop_on_error:
                raise RuntimeError(f"Discovery run failed: {' '.join(discover_cmd)}")

            if dtype == "int8" and int_layers_override is not None:
                targets = int_layers_override
            elif dtype == "fp32" and fp_layers_override is not None:
                targets = fp_layers_override
            else:
                if num_layers <= 0:
                    num_layers = 1
                targets = list(range(num_layers))
            if mode_retry_runs is not None:
                targets = sorted(set(rr["target_layer"] for rr in mode_retry_runs))

            # Now that target layers are known for this (dtype, mode), expand plan.
            if mode_retry_runs is not None:
                selected_fi_runs = sum(
                    1
                    for pattern, delta in faults
                    for layer in targets
                    if any(
                        rr["pattern"] == pattern
                        and rr["delta"] == delta
                        and rr["target_layer"] == layer
                        for rr in mode_retry_runs
                    )
                )
                planned_total += selected_fi_runs
            else:
                planned_total += len(faults) * len(targets)
            if pbar is not None:
                pbar.total = planned_total
                pbar.refresh()
            completed += 1
            if pbar is not None:
                pbar.update(1)

            for pattern, delta in faults:
                for layer in targets:
                    if mode_retry_runs is not None:
                        keep = any(
                            rr["pattern"] == pattern and rr["delta"] == delta and rr["target_layer"] == layer
                            for rr in mode_retry_runs
                        )
                        if not keep:
                            continue
                    progress_line(f"run {dtype}/{mode} {pattern} d{delta} layer={layer}")
                    cmd = mk_base_cmd(args, model, mode)
                    if mode == "abyzft":
                        add_abyzft_defaults(cmd, args)
                    add_fault_flags(cmd, args, mode, enabled=True, pattern=pattern, delta=delta, layer=layer)
                    label = f"fi_{dtype}_{mode}_{pattern}_d{delta}_l{layer}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    cmd += ["-l", label]

                    if args.dry_run:
                        print("DRY-RUN:", " ".join(cmd))
                        rc = 0
                        max_delta = 0.0
                        per_layer = {}
                        stdout_log = ""
                        stderr_log = ""
                    else:
                        rc, _so, _se, attempt_used = run_cmd_with_retries(
                            cmd, cwd=repo_root, retries=args.max_retries_per_run
                        )
                        host_log_for_run = find_host_log_by_label(
                            args.mlonmcu_home_resolved, label
                        )
                        host_text = read_log(host_log_for_run) if host_log_for_run else ""
                        per_layer = merge_per_layer_deltas(_so, _se, host_text)
                        _layers_found, row_max, col_max, max_delta = summarize_per_layer(per_layer)
                        out_path, err_path = write_run_output(runlogs_dir, label, _so, _se)
                        stdout_log = str(out_path)
                        stderr_log = str(err_path)
                        if rc != 0 and args.stop_on_error:
                            raise RuntimeError(f"Run failed: {' '.join(cmd)}")

                    rows.append(
                        {
                            "dtype": dtype,
                            "model": model,
                            "mode": mode,
                            "pattern": pattern,
                            "delta": delta,
                            "target_layer": layer,
                            "return_code": rc,
                            "detected_layers": num_layers,
                            "row_max_logged_delta": row_max if not args.dry_run else 0.0,
                            "col_max_logged_delta": col_max if not args.dry_run else 0.0,
                            "max_abs_logged_delta": max_delta,
                            "label": label,
                            "stdout_log": stdout_log,
                            "stderr_log": stderr_log,
                            "attempts_used": attempt_used if not args.dry_run else 0,
                        }
                    )
                    for logged_layer, (row_delta, col_delta) in sorted(per_layer.items()):
                        per_layer_rows.append(
                            {
                                "dtype": dtype,
                                "model": model,
                                "mode": mode,
                                "label": label,
                                "run_pattern": pattern,
                                "run_delta": delta,
                                "run_target_layer": layer,
                                "logged_layer": logged_layer,
                                "row_max_delta": row_delta,
                                "col_max_delta": col_delta,
                            }
                        )
                    completed += 1
                    if pbar is not None:
                        pbar.update(1)

    fieldnames = [
        "dtype",
        "model",
        "mode",
        "pattern",
        "delta",
        "target_layer",
        "return_code",
        "detected_layers",
        "row_max_logged_delta",
        "col_max_logged_delta",
        "max_abs_logged_delta",
        "label",
        "stdout_log",
        "stderr_log",
        "attempts_used",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    per_layer_csv = out_csv.with_name(f"{out_csv.stem}_per_layer.csv")
    per_layer_fieldnames = [
        "dtype",
        "model",
        "mode",
        "label",
        "run_pattern",
        "run_delta",
        "run_target_layer",
        "logged_layer",
        "row_max_delta",
        "col_max_delta",
    ]
    with per_layer_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=per_layer_fieldnames)
        writer.writeheader()
        writer.writerows(per_layer_rows)

    print(f"Wrote {len(rows)} rows to: {out_csv}")
    print(f"Wrote {len(per_layer_rows)} per-layer rows to: {per_layer_csv}")
    failed = sum(1 for r in rows if int(r["return_code"]) != 0)
    print(f"Run failures: {failed}/{len(rows)}")
    if failed:
        print(f"Per-run stdout/stderr logs: {runlogs_dir}")
    generate_heatmaps(rows, per_layer_rows, out_csv)
    if pbar is not None:
        pbar.close()


if __name__ == "__main__":
    main()
