#!/usr/bin/env python3
"""Benchmark onnx2c AWW variants on ETISS and report protection overheads.

The script runs two reference configurations:
* baseline_raw: onnx2c with no extra optimization pass set
* baseline: onnx2c with im2col_all enabled

It then benchmarks the protection variants on top of the optimized baseline and
reports overheads relative to both references.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from mlonmcu.context import MlonMcuContext
from mlonmcu.models.model import Model, ModelFormats
from mlonmcu.session.run import RunStage


FRONTEND_NAME = "onnx"
BACKEND_NAME = "onnx2c"
PLATFORM_NAME = "mlif"

DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "mlonmcu_env"
    / "deps"
    / "src"
    / "onnx2c"
    / "examples"
    / "aww"
    / "mlperf_tiny_micro_kws_fp32.onnx"
)
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parents[1] / "mlonmcu_env" / "results" / "onnx2c_aww_etiss_overheads.csv"

BASE_CONFIG_ETISS = {
    "etiss.rom_size": "0x2000000",
    "etiss.ram_start": "0x3000000",
    "onnx2c.func_name": "entry",
    "onnx2c.log_level": 0,
    "onnx2c.protection": "baseline",
}

BASE_CONFIG_HOST_X86 = {
    "onnx2c.func_name": "entry",
    "onnx2c.log_level": 0,
    "onnx2c.protection": "baseline",
    "benchmark.num_runs": 1000000,
    "benchmark.num_repeat": 1000000,
}

DEFAULT_ONNX2C_EXE = (
    Path(__file__).resolve().parents[1]
    / "mlonmcu_env"
    / "deps"
    / "src"
    / "onnx2c"
    / "build"
    / "onnx2c"
)

RUNS = [
    ("baseline_raw", None, None, None, None),
    ("baseline", "im2col_all", None, None, None),
    # Keep a visible "int16 only" variant for diagnostics.
    ("int16_im2col", "im2col_all,wide_precision", None, None, None),
    # Protection variants should run with im2col baseline settings.
    ("abft", "im2col_all", "abft", None, None),
    # AByzFT uses wide_precision in the current codegen flow.
    ("abyzft", "im2col_all,wide_precision", "abyzft", None, None),
    ("freivalds_1x", "im2col_all", "freivalds", 1, None),
    ("freivalds_2x", "im2col_all", "freivalds", 2, None),
    ("freivalds_3x", "im2col_all", "freivalds", 3, None),
    ("freivalds_4x", "im2col_all", "freivalds", 4, None),
    ("gvfa_1x", "im2col_all", "gvfa", None, 1),
    ("gvfa_2x", "im2col_all", "gvfa", None, 2),
]


def build_model(model_name: str, model_path: Path) -> Model:
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return Model(model_name, model_path, formats=ModelFormats.ONNX)


def build_run_config(
    target: str,
    optimizations: str | None,
    protection: str | None,
    freivalds_checks: int,
    gvfa_checks: int,
    onnx2c_exe: Path | None = None,
) -> dict:
    base_config = BASE_CONFIG_ETISS if target == "etiss" else BASE_CONFIG_HOST_X86
    config = dict(base_config)
    if optimizations is not None:
        config["onnx2c.optimizations"] = optimizations
    if protection is not None:
        config["onnx2c.protection"] = protection
    if protection == "freivalds":
        config["onnx2c.freivalds_checks"] = freivalds_checks
    if protection == "gvfa":
        config["onnx2c.gvfa_checks"] = gvfa_checks
    if onnx2c_exe is not None:
        config["onnx2c.exe"] = str(onnx2c_exe)
    return config


def run_variant(
    context: MlonMcuContext,
    model: Model,
    label: str,
    target: str,
    optimizations: str | None,
    protection: str | None,
    freivalds_checks: int,
    gvfa_checks: int,
    onnx2c_exe: Path | None = None,
) -> dict:
    run_config = build_run_config(target, optimizations, protection, freivalds_checks, gvfa_checks, onnx2c_exe)
    session = context.create_session(label=f"onnx2c_{model.name}_{target}_{label}")
    run = session.create_run(config=run_config)

    features = ["benchmark"] if target == "host_x86" else []

    with session:
        run.add_frontend_by_name(FRONTEND_NAME, context=context)
        run.add_model(model)
        run.add_backend_by_name(BACKEND_NAME, context=context)
        run.add_platform_by_name(PLATFORM_NAME, context=context)
        run.add_target_by_name(target, context=context)
        if features:
            run.add_features_by_name(features, context=context)
        if not session.process_runs(until=RunStage.RUN, context=context):
            report = session.get_reports(results=session.results)
            reason = "unknown"
            if len(report.df) >= 1:
                row0 = report.df.iloc[0].to_dict()
                reason = str(row0.get("Reason", row0.get("Fail Reason", row0.get("Comment", "unknown"))))
            raise RuntimeError(
                f"Run failed for variant '{label}' (session={session.idx}, reason={reason}). "
                f"Inspect: mlonmcu_env/temp/sessions/{session.idx}/report.csv"
            )

    report = session.get_reports(results=session.results)
    df = report.df
    if len(df) != 1:
        raise RuntimeError(f"Expected exactly one report row for '{label}', got {len(df)}")

    row = df.iloc[0].to_dict()
    row["label"] = label
    row["protection"] = protection or "baseline"
    row["optimizations"] = optimizations or "<none>"
    return row


def add_overhead_columns(df: pd.DataFrame, reference_labels: list[str]) -> pd.DataFrame:
    result = df.copy()
    # Detect which metrics are available in the dataframe
    available_metrics = [col for col in ["Total Cycles", "Total Instructions", "Total ROM", "Total RAM", "Average End-to-End Runtime [s]"] if col in result.columns]
    
    for reference_label in reference_labels:
        reference = result.loc[result["label"] == reference_label]
        if reference.empty:
            continue  # Skip if reference not found
        for metric in available_metrics:
            baseline_value = float(reference.iloc[0][metric])
            result[f"{metric} vs {reference_label} (%)"] = (result[metric].astype(float) / baseline_value - 1.0) * 100.0
    return result


def plot_summary(summary: pd.DataFrame, metric: str | None = None, output_path: Path | None = None) -> None:
    import matplotlib.pyplot as plt

    # Auto-select metric if not specified or doesn't exist
    if metric is None or metric not in summary.columns:
        # Try to find a suitable metric column
        candidates = [col for col in summary.columns if col not in ["label", "protection", "optimizations", "Validation"] and not col.endswith("(%)")]
        if not candidates:
            raise ValueError("No suitable metric column found for plotting")
        metric = candidates[0]
        print(f"Using metric: {metric}")

    if metric not in summary.columns:
        raise KeyError(f"Missing metric column '{metric}' in summary table")

    plot_df = summary.copy()
    plot_df[metric] = plot_df[metric].astype(float)

    order = [
        "baseline_raw",
        "baseline",
        "int16_im2col",
        "abft",
        "abyzft",
        "freivalds_1x",
        "freivalds_2x",
        "freivalds_3x",
        "freivalds_4x",
        "gvfa_1x",
        "gvfa_2x",
    ]
    plot_df["_rank"] = plot_df["label"].map(lambda value: order.index(value) if value in order else len(order))
    plot_df = plot_df.sort_values(["_rank", "label"]).drop(columns=["_rank"]).reset_index(drop=True)

    baseline_series = plot_df.loc[plot_df["label"] == "baseline", metric]
    if baseline_series.empty:
        raise ValueError("No baseline row found for plotting")
    baseline_value = float(baseline_series.iloc[0])

    plt.figure(figsize=(11, 4))
    bars = plt.bar(plot_df["label"], plot_df[metric])
    plt.ylabel(metric)
    plt.title(f"ONNX2C AWW Comparison ({metric})")
    plt.grid(axis="y", alpha=0.3)

    for bar, (_, row) in zip(bars, plot_df.iterrows()):
        if row["label"] == "baseline":
            label = "baseline"
        else:
            overhead = (float(row[metric]) / baseline_value - 1.0) * 100.0
            label = f"{overhead:+.1f}%"
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            label,
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"Wrote plot to: {output_path}")
    else:
        plt.show()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run onnx2c AWW benchmarks with protection variants.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="Path to the AWW ONNX model")
    parser.add_argument("--model-name", type=str, default="onnx_aww_fp", help="Model name to use in MLonMCU")
    parser.add_argument("--target", type=str, default="etiss", choices=["etiss", "host_x86"], help="Target platform (etiss for simulation, host_x86 for native benchmark)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="CSV file to write the summary to")
    parser.add_argument("--plot", action="store_true", help="Generate a bar plot for the selected metric")
    parser.add_argument("--plot-output", type=Path, default=None, help="Path to save the plot image")
    parser.add_argument("--plot-metric", type=str, default=None, help="Metric column to plot (auto-detected if not specified)")
    parser.add_argument("--skip-failed", action="store_true", help="Skip failed variants instead of aborting the full benchmark run")
    parser.add_argument(
        "--onnx2c-exe",
        type=Path,
        default=DEFAULT_ONNX2C_EXE,
        help="Path to onnx2c executable (defaults to local build binary)",
    )
    args = parser.parse_args()

    model = build_model(args.model_name, args.model)
    rows = []

    onnx2c_exe = args.onnx2c_exe
    if onnx2c_exe is not None and not onnx2c_exe.is_file():
        raise FileNotFoundError(f"onnx2c executable not found: {onnx2c_exe}")

    with MlonMcuContext() as context:
        for label, optimizations, protection, freivalds_checks, gvfa_checks in RUNS:
            try:
                rows.append(
                    run_variant(
                        context,
                        model,
                        label,
                        args.target,
                        optimizations,
                        protection,
                        freivalds_checks or 1,
                        gvfa_checks or 1,
                        onnx2c_exe=onnx2c_exe,
                    )
                )
            except Exception as exc:
                if not args.skip_failed:
                    raise
                print(f"[WARN] Skipping variant '{label}': {exc}")

    df = pd.DataFrame(rows)
    df = add_overhead_columns(df, ["baseline_raw", "baseline"])

    # Build summary columns dynamically based on what's available
    base_columns = ["label", "protection", "optimizations"]
    if "Total Cycles" in df.columns:
        metric_columns = [
            "Total Cycles",
            "Total Cycles vs baseline_raw (%)",
            "Total Cycles vs baseline (%)",
            "Total Instructions",
            "Total Instructions vs baseline_raw (%)",
            "Total Instructions vs baseline (%)",
            "Total ROM",
            "Total ROM vs baseline_raw (%)",
            "Total ROM vs baseline (%)",
            "Total RAM",
            "Total RAM vs baseline_raw (%)",
            "Total RAM vs baseline (%)",
        ]
    else:
        # For host_x86 benchmark
        metric_columns = [
            "Average End-to-End Runtime [s]",
            "Average End-to-End Runtime [s] vs baseline_raw (%)",
            "Average End-to-End Runtime [s] vs baseline (%)",
        ]
    
    summary_columns = base_columns + metric_columns + ["Validation"]
    existing_columns = [column for column in summary_columns if column in df.columns]
    summary = df[existing_columns].copy()

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 0)
    print(summary.to_string(index=False))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    print(f"\nWrote summary to: {args.output}")

    if args.plot:
        plot_output = args.plot_output
        if plot_output is None:
            plot_output = args.output.with_suffix(".png")
        plot_summary(summary, args.plot_metric, plot_output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
