"""Export benchmark data as a consolidated JSON for mlx-stack consumption.

Aggregates all LLM benchmark measurements and quality benchmark results
into a single JSON file keyed by mlx-community HuggingFace repo ID.

Usage:
    python scripts/export_for_mlx_stack.py
    python scripts/export_for_mlx_stack.py -o benchmark_data.json
    python scripts/export_for_mlx_stack.py --copy-to ../mlx-stack/src/mlx_stack/data/

The output JSON is designed to be committed into the mlx-stack project
as static benchmark data that drives the model recommendation engine
and onboarding experience.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from mtb.llm_benchmarks import MODEL_SPECS
from mtb.llm_benchmarks.models.base import ModelSpec

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

MEASUREMENTS_ROOT = Path(__file__).parent.parent / "measurements"
LLM_BENCHMARKS_DIR = MEASUREMENTS_ROOT / "llm_benchmarks"
QUALITY_BENCHMARKS_DIR = MEASUREMENTS_ROOT / "quality_benchmarks"

DEFAULT_OUTPUT = Path(__file__).parent.parent / "benchmark_data.json"

# Map hardware_string from settings.json to a short profile ID
# that mlx-stack uses for hardware-aware recommendations.
HARDWARE_PROFILE_MAP = {
    "Apple_M4_Pro_10P+4E+20GPU_24GB": "m4-pro-24",
    "Apple_M4_Pro_10P+4E+20GPU_48GB": "m4-pro-48",
    "Apple_M4_Pro_10P+4E+20GPU_64GB": "m4-pro-64",
    "Apple_M4_Max_10P+4E+40GPU_128GB": "m4-max-128",
    "Apple_M5_Max_XP+XE+40GPU_128GB": "m5-max-128",
}


# --------------------------------------------------------------------------- #
# Model name → HF repo mapping
# --------------------------------------------------------------------------- #


def build_model_registry() -> dict[str, ModelSpec]:
    """Build a lookup from benchmark model name to ModelSpec."""
    return {spec.name: spec for spec in MODEL_SPECS}


def get_hf_repo(spec: ModelSpec, dtype: str) -> str | None:
    """Get the mlx-community HuggingFace repo for a model + dtype.

    Returns None if no MLX model_id is defined for that dtype.
    """
    mlx_ids = spec.model_ids.get("mlx", {})
    return mlx_ids.get(dtype)


# --------------------------------------------------------------------------- #
# LLM benchmark aggregation
# --------------------------------------------------------------------------- #


def load_llm_benchmarks() -> pd.DataFrame:
    """Load all LLM benchmark CSVs across all hardware profiles."""
    if not LLM_BENCHMARKS_DIR.exists():
        return pd.DataFrame()

    frames = []
    for csv_path in LLM_BENCHMARKS_DIR.glob("*/*/benchmark_results.csv"):
        settings_path = csv_path.parent / "settings.json"
        if not settings_path.exists():
            continue

        df = pd.read_csv(csv_path)
        with settings_path.open() as f:
            settings = json.load(f)

        hw_string = settings.get("hardware_info", {}).get("hardware_string", "unknown")
        df["hardware_string"] = hw_string
        df["run_datetime"] = settings.get("datetime", "")
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def aggregate_llm_data(
    df: pd.DataFrame,
    registry: dict[str, ModelSpec],
) -> dict:
    """Aggregate LLM benchmarks into per-model, per-hardware, per-dtype dicts.

    For each combination, takes the most recent run and averages across
    prompt lengths to produce a single representative set of metrics.
    """
    if df.empty:
        return {}

    # Filter to MLX framework only (that's what mlx-stack serves)
    df = df[df["framework"] == "mlx"].copy()

    results: dict = defaultdict(lambda: defaultdict(dict))

    # Group by model + hardware + dtype, take most recent run
    for (name, hw_string, dtype), group in df.groupby(
        ["name", "hardware_string", "dtype"]
    ):
        if name not in registry:
            continue

        # Use most recent run
        latest = group.sort_values("run_datetime", ascending=False)
        latest_run = latest["run_datetime"].iloc[0]
        run_data = latest[latest["run_datetime"] == latest_run]

        # Average across prompt lengths for representative metrics
        metrics = {
            "generation_tps": round(float(run_data["generation_tps"].mean()), 1),
            "prompt_tps": round(float(run_data["prompt_tps"].mean()), 1),
        }

        # Peak memory (take max across prompt lengths)
        if "peak_memory_gib" in run_data.columns:
            mem = run_data["peak_memory_gib"].max()
            if pd.notna(mem):
                metrics["peak_memory_gib"] = round(float(mem), 2)

        hw_profile = HARDWARE_PROFILE_MAP.get(hw_string, hw_string)
        results[name][hw_profile][dtype] = metrics

    return dict(results)


# --------------------------------------------------------------------------- #
# Quality benchmark aggregation
# --------------------------------------------------------------------------- #


def load_quality_benchmarks() -> pd.DataFrame:
    """Load all quality benchmark CSVs across all hardware profiles."""
    if not QUALITY_BENCHMARKS_DIR.exists():
        return pd.DataFrame()

    frames = []
    for csv_path in QUALITY_BENCHMARKS_DIR.glob("*/*/quality_results.csv"):
        settings_path = csv_path.parent / "settings.json"
        if not settings_path.exists():
            continue

        df = pd.read_csv(csv_path)
        with settings_path.open() as f:
            settings = json.load(f)

        hw_string = settings.get("hardware_info", {}).get("hardware_string", "unknown")
        df["hardware_string"] = hw_string
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def aggregate_quality_data(df: pd.DataFrame) -> dict:
    """Aggregate quality benchmarks into per-model pass rates by category.

    Returns dict mapping model_name -> {category -> pass_rate, "overall" -> pass_rate}
    """
    if df.empty:
        return {}

    # Filter to MLX only
    df = df[df["framework"] == "mlx"].copy()

    results: dict = {}

    for name, group in df.groupby("model"):
        total_problems = len(group)
        total_passed = int(group["passed"].sum())

        quality: dict = {
            "overall_pass_rate": round(total_passed / total_problems, 2)
            if total_problems > 0
            else 0.0,
        }

        # Per-category pass rates
        by_category: dict = {}
        for category, cat_group in group.groupby("category"):
            cat_total = len(cat_group)
            cat_passed = int(cat_group["passed"].sum())
            by_category[str(category)] = round(cat_passed / cat_total, 2) if cat_total > 0 else 0.0

        quality["by_category"] = by_category
        results[str(name)] = quality

    return results


# --------------------------------------------------------------------------- #
# Export builder
# --------------------------------------------------------------------------- #


def build_export(
    llm_data: dict,
    quality_data: dict,
    registry: dict[str, ModelSpec],
) -> dict:
    """Build the final export JSON structure.

    Output format:
    {
        "generated_at": "...",
        "source": "mlx-coding-bench@<commit>",
        "models": {
            "mlx-community/Qwen3.5-9B-4bit": {
                "benchmark_name": "qwen-3.5-9b",
                "params_b": 9.0,
                "thinking": true,
                "benchmarks": {
                    "m4-pro-64": {
                        "int4": {
                            "generation_tps": 62.3,
                            "prompt_tps": 358.7,
                            "peak_memory_gib": 5.2
                        }
                    }
                },
                "quality": {
                    "overall_pass_rate": 0.82,
                    "by_category": {"coding": 0.87, ...}
                }
            }
        }
    }
    """
    models: dict = {}

    for spec in MODEL_SPECS:
        mlx_ids = spec.model_ids.get("mlx", {})
        if not mlx_ids:
            continue

        # Create an entry for each quantization variant
        for dtype, hf_repo in mlx_ids.items():
            entry: dict = {
                "benchmark_name": spec.name,
                "params_b": round(spec.num_params / 1e9, 1),
                "thinking": spec.thinking,
            }

            # Add performance benchmarks
            if spec.name in llm_data:
                benchmarks: dict = {}
                for hw_profile, dtype_data in llm_data[spec.name].items():
                    if dtype in dtype_data:
                        benchmarks[hw_profile] = dtype_data[dtype]
                if benchmarks:
                    entry["benchmarks"] = benchmarks

            # Add quality data (quality is model-level, not dtype-specific)
            if spec.name in quality_data:
                entry["quality"] = quality_data[spec.name]

                # Derive tool_calling capability from benchmark results
                tc_rate = quality_data[spec.name].get("by_category", {}).get("tool_calling")
                if tc_rate is not None:
                    entry["tool_calling"] = tc_rate >= 0.6  # passes if 60%+ of tool-calling tests pass

            models[hf_repo] = entry

    # Get git commit
    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=Path(__file__).parent.parent,
            )
            .decode()
            .strip()
        )
    except Exception:
        commit = "unknown"

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": f"mlx-coding-bench@{commit}",
        "models": models,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export benchmark data as JSON for mlx-stack",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output JSON file (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--copy-to",
        type=Path,
        default=None,
        help="Copy output to this directory (e.g., ../mlx-stack/src/mlx_stack/data/)",
    )
    args = parser.parse_args()

    registry = build_model_registry()
    print(f"Loaded {len(registry)} model specs")

    # Load and aggregate LLM benchmarks
    llm_df = load_llm_benchmarks()
    print(f"Loaded {len(llm_df)} LLM benchmark rows")
    llm_data = aggregate_llm_data(llm_df, registry)
    print(f"Aggregated LLM data for {len(llm_data)} models")

    # Load and aggregate quality benchmarks
    quality_df = load_quality_benchmarks()
    print(f"Loaded {len(quality_df)} quality benchmark rows")
    quality_data = aggregate_quality_data(quality_df)
    print(f"Aggregated quality data for {len(quality_data)} models")

    # Build export
    export = build_export(llm_data, quality_data, registry)
    print(f"Export contains {len(export['models'])} model entries")

    # Write output
    args.output.write_text(
        json.dumps(export, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Written to {args.output}")

    # Copy if requested
    if args.copy_to:
        dest = args.copy_to / args.output.name
        args.copy_to.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.output, dest)
        print(f"Copied to {dest}")

    # Summary
    print("\n--- Summary ---")
    for hf_repo, entry in export["models"].items():
        hw_count = len(entry.get("benchmarks", {}))
        has_quality = "quality" in entry
        print(
            f"  {hf_repo}: {entry['params_b']}B, "
            f"{hw_count} hardware profile(s), "
            f"quality={'yes' if has_quality else 'no'}"
        )


if __name__ == "__main__":
    main()
