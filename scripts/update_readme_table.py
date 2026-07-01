"""Regenerate the agentic coding benchmark tables in README.md.

Reads all speed and quality benchmark CSVs, computes the latest results
for each model at int4 and int8 per hardware profile, and rewrites the
benchmark section of the README between the markers.

Usage:
    uv run python scripts/update_readme_table.py
    uv run python scripts/update_readme_table.py --models '["gemma-4-e2b-it","lfm2-24b-a2b"]'
    uv run python scripts/update_readme_table.py --dry-run
"""

import glob
import json
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import fire
import pandas as pd

import mtb

REPO_ROOT = mtb.REPO_ROOT
README_PATH = REPO_ROOT / "README.md"
SPEED_ROOT = REPO_ROOT / "measurements" / "llm_benchmarks"
QUALITY_ROOT = REPO_ROOT / "measurements" / "quality_benchmarks"

BEGIN_MARKER = "<!-- BEGIN BENCHMARK TABLE -->"
END_MARKER = "<!-- END BENCHMARK TABLE -->"

# Hardware display names
HARDWARE_DISPLAY = {
    "Apple_M4_Pro_10P+4E+20GPU_64GB": "M4 Pro 64GB",
    "Apple_M4_Pro_10P+4E+20GPU_24GB": "M4 Pro 24GB",
    "Apple_M5_Max_XP+XE+40GPU_128GB": "M5 Max 128GB",
}


def _short_model_id(model_id: str) -> str:
    """'mlx-community/Qwen3-0.6B-4bit' -> 'Qwen3-0.6B' for compact display."""
    if not model_id:
        return "--"
    name = model_id.split("/")[-1]
    for suffix in ("-4bit", "-8bit", "-bf16", "-MLX", "-mlx", "-Instruct"):
        name = name.replace(suffix, "")
    return name


# Registry of optimizations layered on top of the standard config. Each entry
# controls how that optimization renders in the Optimizations section. Add a new
# key here (and set benchmark.optimization to match) to introduce a new method —
# the section, speedup math, and table are all generic.
OPTIMIZATION_DISPLAY = {
    "speculative_decoding": {
        "title": "Speculative decoding",
        "blurb": (
            "A small **drafter** model proposes tokens that the base model "
            "verifies in parallel. Output is identical to the standard config — "
            "only throughput changes. Requires a drafter sharing the base "
            "model's tokenizer, and a base architecture with a trimmable KV "
            "cache (Qwen/Llama/Gemma work; some newer architectures don't yet)."
        ),
        "config_header": "Drafter",
        "config_fn": _short_model_id,
        # spec-dec preserves output, so quantify only speed (no memory/quality).
    },
}

# Model display names and metadata
MODEL_META = {
    "gemma-4-e2b-it": {"display": "Gemma 4 E2B-it", "arch": "2.3B dense"},
    "gemma-4-e4b-it": {"display": "Gemma 4 E4B-it", "arch": "4.5B dense"},
    "gemma-4-12b-it": {"display": "Gemma 4 12B-it", "arch": "12B dense"},
    "gemma-4-26b-a4b-it": {"display": "Gemma 4 26B-A4B-it", "arch": "3.8B MoE"},
    "gemma-4-31b-it": {"display": "Gemma 4 31B-it", "arch": "31B dense"},
    "qwen3-coder-30b-a3b": {"display": "Qwen3-Coder-30B-A3B", "arch": "3B MoE"},
    "glm-4.7-flash": {"display": "GLM-4.7-Flash", "arch": "3B MoE"},
    "lfm2-24b-a2b": {"display": "LFM2-24B-A2B", "arch": "2B MoE"},
    "lfm2.5-8b-a1b": {"display": "LFM2.5-8B-A1B", "arch": "1.5B MoE"},
    "qwen-3.5-35b-a3b": {"display": "Qwen 3.5-35B-A3B", "arch": "3B MoE"},
    "qwen-3.6-35b-a3b": {"display": "Qwen 3.6-35B-A3B", "arch": "3B MoE"},
    "qwen-3.6-27b": {"display": "Qwen 3.6-27B", "arch": "27B dense"},
    "qwen-3.5-27b": {"display": "Qwen 3.5-27B", "arch": "27B dense"},
    "qwen-3.5-27b-claude-opus-distilled": {
        "display": "Qwen 3.5-27B Opus Distilled",
        "arch": "27B dense",
    },
    "qwen-3.5-9b": {"display": "Qwen 3.5-9B", "arch": "9B dense"},
    "qwen-3.5-4b": {"display": "Qwen 3.5-4B", "arch": "4B dense"},
    "qwen-3.5-2b": {"display": "Qwen 3.5-2B", "arch": "2B dense"},
    "qwen-3.5-0.8b": {"display": "Qwen 3.5-0.8B", "arch": "0.8B dense"},
    "nemotron-nano-9b-v2": {"display": "Nemotron-Nano-9B-v2", "arch": "9B dense"},
    "nemotron-3-nano-4b": {"display": "Nemotron-3-Nano-4B", "arch": "4B dense"},
    "nemotron-cascade-2-30b-a3b": {
        "display": "Nemotron-Cascade-2-30B-A3B",
        "arch": "3B MoE",
    },
    "Deepseek-R1-0528_Qwen3-8B": {
        "display": "DeepSeek-R1-0528-Qwen3-8B",
        "arch": "8B dense",
    },
    "Deepseek-R1-Distill-7B": {"display": "DeepSeek-R1-Distill-7B", "arch": "7B dense"},
    "gemma-3-4b-it": {"display": "Gemma 3-4B-it", "arch": "4B dense"},
    "gemma-3-4b-it-qat": {"display": "Gemma 3-4B-it QAT", "arch": "4B dense"},
    "gemma-3-1b-it": {"display": "Gemma 3-1B-it", "arch": "1B dense"},
    "gemma-3-1b-it-qat": {"display": "Gemma 3-1B-it QAT", "arch": "1B dense"},
    "gemma-3-12b-it-qat": {"display": "Gemma 3-12B-it QAT", "arch": "12B dense"},
    "gemma-3-27b-it": {"display": "Gemma 3-27B-it", "arch": "27B dense"},
    "qwen-2.5-0.5b-it": {"display": "Qwen 2.5-0.5B-it", "arch": "0.5B dense"},
    "qwen-2.5-3b-it": {"display": "Qwen 2.5-3B-it", "arch": "3B dense"},
    "qwen-2.5-coder-0.5b-it": {"display": "Qwen 2.5-Coder-0.5B", "arch": "0.5B dense"},
    "qwen-2.5-coder-3b-it": {"display": "Qwen 2.5-Coder-3B", "arch": "3B dense"},
    "qwen-3-0.6b-it": {"display": "Qwen 3-0.6B-it", "arch": "0.6B dense"},
    "qwen-3-8B-it": {"display": "Qwen 3-8B-it", "arch": "8B dense"},
    "qwen-3-14B-it": {"display": "Qwen 3-14B-it", "arch": "14B dense"},
    "qwen-3-32B-it": {"display": "Qwen 3-32B-it", "arch": "32B dense"},
    "llama-3.3-70b-it": {"display": "Llama 3.3-70B-it", "arch": "70B dense"},
}


def _min_hw_from_memory(mem_gib: float) -> str:
    if mem_gib <= 8:
        return "Any Mac"
    elif mem_gib <= 14:
        return "16GB+"
    elif mem_gib <= 22:
        return "24GB+"
    elif mem_gib <= 30:
        return "36GB+"
    elif mem_gib <= 44:
        return "48GB+"
    elif mem_gib <= 58:
        return "64GB+"
    else:
        return "128GB+"


def format_model_name(name: str, include_arch: bool = False) -> str:
    meta = MODEL_META.get(name)
    if meta:
        if include_arch:
            return f"{meta['display']} ({meta['arch']})"
        return meta["display"]
    return name


def get_arch(name: str) -> str:
    meta = MODEL_META.get(name)
    if meta:
        return meta["arch"]
    return "unknown"


def load_speed_data(
    models: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Load speed data with hardware profile info."""
    files = glob.glob(str(SPEED_ROOT / "**" / "benchmark_results.csv"), recursive=True)
    if not files:
        return pd.DataFrame()

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        p = Path(f)
        df["_source_dir"] = p.parent.name
        df["hardware"] = p.parts[-3]  # hardware directory name
        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)

    # Optimization + prompt-profile dimensions (backward-compat: older CSVs
    # predate them, so their rows are the standard config on generic prompts).
    for col in ("optimization", "optimization_detail"):
        if col not in all_df.columns:
            all_df[col] = ""
        all_df[col] = all_df[col].fillna("")
    if "prompt_profile" not in all_df.columns:
        all_df["prompt_profile"] = "generic"
    all_df["prompt_profile"] = all_df["prompt_profile"].fillna("generic")

    # Filter to 1024 prompt tokens, MLX metal
    mask = (all_df["num_prompt_tokens"] == 1024) & (all_df["framework"] == "mlx")
    all_df = all_df[mask].copy()

    if models:
        all_df = all_df[all_df["name"].isin(models)]

    # Keep latest per model/dtype/hardware/optimization/prompt_profile
    # (standard and each optimization×profile are distinct rows for a model).
    all_df = (
        all_df.sort_values("_source_dir")
        .groupby(["hardware", "name", "dtype", "optimization", "prompt_profile"])
        .last()
        .reset_index()
    )

    return all_df[
        [
            "hardware",
            "name",
            "dtype",
            "optimization",
            "optimization_detail",
            "prompt_profile",
            "generation_tps",
            "prompt_tps",
            "peak_memory_gib",
        ]
    ]


def load_quality_data(
    models: Optional[List[str]] = None,
) -> pd.DataFrame:
    files = glob.glob(str(QUALITY_ROOT / "**" / "quality_results.csv"), recursive=True)
    if not files:
        return pd.DataFrame()

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        p = Path(f)
        df["_source_dir"] = p.parent.name
        df["hardware"] = p.parts[-3]
        # Read the per-run token-budget multiplier (>1 for verbose thinking
        # models run with raised caps) so the table can flag those rows.
        mult = 1.0
        settings_path = p.parent / "settings.json"
        if settings_path.exists():
            try:
                settings = json.loads(settings_path.read_text())
                mult = settings.get("benchmark_settings", {}).get(
                    "max_tokens_multiplier", 1.0
                )
            except (ValueError, OSError):
                mult = 1.0
        df["_max_tokens_multiplier"] = mult
        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)

    if models:
        all_df = all_df[all_df["model"].isin(models)]

    # Keep only data from the latest run per model/dtype/hardware.
    # This avoids merging stale problem names from older runs that used
    # a different problem set (e.g. 46-problem suite vs 81-problem suite).
    latest_source = (
        all_df.sort_values("_source_dir")
        .groupby(["hardware", "model", "dtype"])["_source_dir"]
        .last()
        .reset_index()
        .rename(columns={"_source_dir": "_latest_dir"})
    )
    all_df = all_df.merge(latest_source, on=["hardware", "model", "dtype"])
    all_df = all_df[all_df["_source_dir"] == all_df["_latest_dir"]].drop(
        columns=["_latest_dir"]
    )

    return all_df


def compute_quality_summary(
    quality_df: pd.DataFrame, hardware: str, dtype: str = "int4"
) -> pd.DataFrame:
    if quality_df.empty or "dtype" not in quality_df.columns:
        return pd.DataFrame()

    df = quality_df[
        (quality_df["dtype"] == dtype) & (quality_df["hardware"] == hardware)
    ].copy()
    if df.empty:
        return pd.DataFrame()

    # Raw (flat) pass rate
    overall = (
        df.groupby("model")
        .agg(passed=("passed", "sum"), total=("passed", "count"))
        .reset_index()
    )
    overall["quality_pct"] = (overall["passed"] / overall["total"] * 100).round(1)

    # Weighted score via scoring module
    from mtb.quality_benchmarks.scoring import (
        compute_weighted_score,
        _build_problem_tier_map,
        _resolve_variant_name,
    )

    tier_map = _build_problem_tier_map()

    weighted_scores = []
    for model_name in overall["model"]:
        model_df = df[df["model"] == model_name]
        results = {
            row["problem"]: bool(row["passed"]) for _, row in model_df.iterrows()
        }

        # Count how many CSV problems are recognized by the tier mapping
        # (resolving variant names like fizzbuzz_variant_1 -> fizzbuzz)
        recognized = sum(
            1 for name in results if _resolve_variant_name(name) in tier_map
        )
        total_in_csv = len(results)

        # If fewer than half the problems are recognized, the weighted score
        # is unreliable (legacy/unknown problem IDs).  Fall back to flat rate.
        if total_in_csv > 0 and recognized / total_in_csv < 0.5:
            weighted_scores.append(float("nan"))
        else:
            score = compute_weighted_score(results)
            weighted_scores.append(round(score["weighted_score"] * 100, 1))
    overall["weighted_pct"] = weighted_scores

    # Per-category breakdowns
    cats = {}
    for cat in ["coding", "tool_calling", "reasoning"]:
        cat_df = df[df["category"] == cat]
        if cat_df.empty:
            continue
        cat_sum = (
            cat_df.groupby("model")
            .agg(p=("passed", "sum"), t=("passed", "count"))
            .reset_index()
        )
        cat_sum[cat] = cat_sum.apply(lambda r: f"{int(r['p'])}/{int(r['t'])}", axis=1)
        cats[cat] = cat_sum[["model", cat]]

    result = overall[["model", "weighted_pct", "quality_pct"]].copy()
    for cat, cat_df in cats.items():
        result = result.merge(cat_df, on="model", how="left")

    # Per-model token-budget multiplier (max across the model's rows).
    if "_max_tokens_multiplier" in df.columns:
        mult = (
            df.groupby("model")["_max_tokens_multiplier"].max().reset_index()
        )
        result = result.merge(mult, on="model", how="left")

    return result


def pick_quick_picks(combined: pd.DataFrame) -> List[dict]:
    picks = []
    if combined.empty:
        return picks

    # Use weighted_pct as primary quality metric, fall back to quality_pct
    qual_col = (
        "weighted_pct"
        if "weighted_pct" in combined.columns and combined["weighted_pct"].notna().any()
        else "quality_pct"
    )
    has_quality = qual_col in combined.columns and combined[qual_col].notna().any()

    if has_quality:
        # Best overall: highest quality, then fastest
        with_quality = combined[combined[qual_col].notna()]
        if not with_quality.empty:
            best = with_quality.sort_values(
                [qual_col, "generation_tps"], ascending=[False, False]
            ).iloc[0]
            picks.append({"use_case": "Best overall", "row": best})

            # Best MoE
            moe = with_quality[with_quality["arch"].str.contains("MoE")]
            if not moe.empty:
                best_moe = moe.sort_values(
                    [qual_col, "generation_tps"], ascending=[False, False]
                ).iloc[0]
                if best_moe["name"] != best["name"]:
                    picks.append({"use_case": "Best MoE", "row": best_moe})

            # Best coder: not yet picked, highest quality then fastest
            for _, row in with_quality.sort_values(
                [qual_col, "generation_tps"], ascending=[False, False]
            ).iterrows():
                if row["name"] not in [p["row"]["name"] for p in picks]:
                    picks.append({"use_case": "Best coder", "row": row})
                    break

            # Best reasoning: not yet picked
            for _, row in with_quality.sort_values(
                [qual_col, "generation_tps"], ascending=[False, False]
            ).iterrows():
                if row["name"] not in [p["row"]["name"] for p in picks]:
                    picks.append({"use_case": "Best reasoning", "row": row})
                    break
    else:
        # No quality data — just pick fastest
        best = combined.sort_values("generation_tps", ascending=False).iloc[0]
        picks.append({"use_case": "Fastest", "row": best})

    return picks


def _format_quality_cell(row, top_quality: float, has_weighted: bool) -> str:
    """Format the Quality column cell."""
    weighted = row.get("weighted_pct") if has_weighted else None
    raw = row.get("quality_pct")

    if pd.notna(weighted):
        primary = weighted
    elif pd.notna(raw):
        primary = raw
    else:
        return "--"

    is_top = top_quality is not None and primary >= top_quality - 0.1
    if is_top:
        return f"**{primary}%**"
    return f"{primary}%"


def _format_speed_cell(tps: float) -> str:
    """Format speed with a usability indicator.

    Community consensus thresholds:
    - 100+ tok/s: fast (autocomplete)
    - 50+ tok/s: good (agentic coding)
    - 30+ tok/s: ok (interactive chat)
    - <30 tok/s: slow
    """
    tps_int = int(round(tps))
    return str(tps_int)


def generate_hardware_table(
    speed_df: pd.DataFrame,
    quality_df: pd.DataFrame,
    hardware: str,
) -> str:
    """Generate tables for a single hardware profile.

    Layout: quality-scored models first (sorted by quality desc),
    then speed-only models in a collapsible section (sorted by speed desc).
    Columns: Model (with arch), RAM, Quality, Gen tok/s.
    """
    hw_speed = speed_df[speed_df["hardware"] == hardware].copy()
    if hw_speed.empty:
        return ""

    int4_speed = hw_speed[hw_speed["dtype"] == "int4"].copy()
    int8_speed = hw_speed[hw_speed["dtype"] == "int8"].copy()

    if int4_speed.empty:
        return ""

    # Quality
    quality_summary = compute_quality_summary(quality_df, hardware, "int4")

    # Combine
    combined = int4_speed.rename(columns={"name": "model"}).copy()
    if not quality_summary.empty:
        combined = combined.merge(quality_summary, on="model", how="left")
    else:
        combined["weighted_pct"] = None
        combined["quality_pct"] = None

    combined["min_hw"] = combined.apply(
        lambda r: _min_hw_from_memory(r["peak_memory_gib"]), axis=1
    )
    combined = combined.rename(columns={"model": "name"})

    has_weighted = (
        "weighted_pct" in combined.columns and combined["weighted_pct"].notna().any()
    )
    has_quality = (
        "quality_pct" in combined.columns and combined["quality_pct"].notna().any()
    )

    # Determine quality column for sorting
    qual_col = "weighted_pct" if has_weighted else "quality_pct"

    # Split into models with and without quality scores
    if has_weighted or has_quality:
        with_quality = combined[combined[qual_col].notna()].copy()
        without_quality = combined[~combined[qual_col].notna()].copy()
    else:
        with_quality = pd.DataFrame()
        without_quality = combined.copy()

    # Top quality for bolding
    top_quality = with_quality[qual_col].max() if not with_quality.empty else None

    lines = []

    # --- Main table: models with quality, sorted by quality desc ---
    if not with_quality.empty:
        with_quality = with_quality.sort_values(
            [qual_col, "generation_tps"], ascending=[False, False]
        )

        lines.append("| Model | RAM | Quality | Gen tok/s |")
        lines.append("|---|---:|---:|---:|")

        for _, r in with_quality.iterrows():
            name = format_model_name(r["name"], include_arch=True)
            qual_str = _format_quality_cell(r, top_quality, has_weighted)
            mult = r.get("_max_tokens_multiplier", 1.0)
            if pd.notna(mult) and mult and mult > 1:
                qual_str = f"{qual_str} †"
            tps_str = _format_speed_cell(r["generation_tps"])
            mem = f"{r['peak_memory_gib']:.1f} GiB"
            lines.append(f"| {name} | {mem} | {qual_str} | {tps_str} |")

    # --- Speed-only models in collapsible section ---
    if not without_quality.empty:
        without_quality = without_quality.sort_values(
            "generation_tps", ascending=False
        )

        if not with_quality.empty:
            lines.append("")
            lines.append("<details>")
            lines.append("<summary>Speed-only models (no quality scores yet)</summary>")
            lines.append("")

        lines.append("| Model | RAM | Gen tok/s | Prefill tok/s |")
        lines.append("|---|---:|---:|---:|")

        for _, r in without_quality.iterrows():
            name = format_model_name(r["name"], include_arch=True)
            tps_str = _format_speed_cell(r["generation_tps"])
            mem = f"{r['peak_memory_gib']:.1f} GiB"
            lines.append(
                f"| {name} | {mem} | {tps_str} | {r['prompt_tps']:.0f} |"
            )

        if not with_quality.empty:
            lines.append("")
            lines.append("</details>")

    # --- int8 collapsible ---
    if not int8_speed.empty:
        int8_speed = int8_speed.sort_values("generation_tps", ascending=False)
        lines.append("")
        lines.append("<details>")
        lines.append("<summary>int8 speed results</summary>")
        lines.append("")
        lines.append("| Model | RAM | Gen tok/s | Prefill tok/s |")
        lines.append("|---|---:|---:|---:|")
        for _, r in int8_speed.iterrows():
            name = format_model_name(r["name"], include_arch=True)
            mem = f"{r['peak_memory_gib']:.1f} GiB"
            tps_str = _format_speed_cell(r["generation_tps"])
            lines.append(
                f"| {name} | {mem} | {tps_str} | {r['prompt_tps']:.0f} |"
            )
        lines.append("")
        lines.append("</details>")

    return "\n".join(lines)


def _get_combined_for_hardware(
    speed_df: pd.DataFrame,
    quality_df: pd.DataFrame,
    hardware: str,
) -> pd.DataFrame:
    """Build combined speed+quality dataframe for a hardware profile."""
    hw_speed = speed_df[speed_df["hardware"] == hardware].copy()
    int4_speed = hw_speed[hw_speed["dtype"] == "int4"].copy()
    if int4_speed.empty:
        return pd.DataFrame()

    quality_summary = compute_quality_summary(quality_df, hardware, "int4")
    combined = int4_speed.rename(columns={"name": "model"}).copy()
    if not quality_summary.empty:
        combined = combined.merge(quality_summary, on="model", how="left")
    else:
        combined["weighted_pct"] = None
        combined["quality_pct"] = None

    combined["arch"] = combined["model"].map(get_arch)
    combined["min_hw"] = combined.apply(
        lambda r: _min_hw_from_memory(r["peak_memory_gib"]), axis=1
    )
    combined = combined.rename(columns={"model": "name"})
    combined = combined.sort_values("generation_tps", ascending=False)
    return combined


def _format_pick(name: str, tps: float, qual_pct=None) -> str:
    """Format a model pick for the summary table."""
    display = format_model_name(name)
    if qual_pct is not None and pd.notna(qual_pct):
        return f"{display} ({tps:.0f} tok/s, {qual_pct}%)"
    return f"{display} ({tps:.0f} tok/s)"


def generate_cross_hardware_summary(
    speed_df: pd.DataFrame,
    quality_df: pd.DataFrame,
    hardware_profiles: List[str],
) -> str:
    """Generate a cross-hardware Best Models summary table.

    Categories:
    - Best Quality: highest weighted score
    - Best Balance: highest quality with >= 50 tok/s (agentic-usable)
    - Best Speed: fastest with quality >= 60%
    """
    lines = []
    lines.append("### Best Models by Hardware")
    lines.append("")
    lines.append("| Hardware | Best Quality | Best Balance | Best Speed |")
    lines.append("|---|---|---|---|")

    for hw in hardware_profiles:
        hw_display = HARDWARE_DISPLAY.get(hw, hw)
        combined = _get_combined_for_hardware(speed_df, quality_df, hw)
        if combined.empty:
            continue

        qual_col = (
            "weighted_pct"
            if "weighted_pct" in combined.columns
            and combined["weighted_pct"].notna().any()
            else "quality_pct"
        )
        has_quality = qual_col in combined.columns and combined[qual_col].notna().any()

        if not has_quality:
            fastest = combined.sort_values("generation_tps", ascending=False).iloc[0]
            f = _format_pick(fastest["name"], fastest["generation_tps"])
            lines.append(f"| **{hw_display}** | -- | -- | {f} |")
            continue

        with_quality = combined[combined[qual_col].notna()]

        # Best Quality: highest score
        best_q = with_quality.sort_values(
            [qual_col, "generation_tps"], ascending=[False, False]
        ).iloc[0]
        quality_str = _format_pick(
            best_q["name"], best_q["generation_tps"], best_q[qual_col]
        )

        # Best Balance: highest quality with >= 50 tok/s (agentic-usable)
        balance_candidates = with_quality[with_quality["generation_tps"] >= 50.0]
        if balance_candidates.empty:
            balance_candidates = with_quality[with_quality["generation_tps"] >= 30.0]
        if not balance_candidates.empty:
            best_b = balance_candidates.sort_values(
                [qual_col, "generation_tps"], ascending=[False, False]
            ).iloc[0]
            balance_str = _format_pick(
                best_b["name"], best_b["generation_tps"], best_b[qual_col]
            )
        else:
            balance_str = "--"

        # Best Speed: fastest with quality >= 60%
        speed_candidates = with_quality[with_quality[qual_col] >= 60.0]
        if speed_candidates.empty:
            speed_candidates = with_quality
        best_s = speed_candidates.sort_values(
            "generation_tps", ascending=False
        ).iloc[0]
        speed_str = _format_pick(
            best_s["name"], best_s["generation_tps"], best_s[qual_col]
        )

        lines.append(
            f"| **{hw_display}** | {quality_str} | {balance_str} | {speed_str} |"
        )

    return "\n".join(lines)


def generate_optimizations_section(speed_df: pd.DataFrame) -> str:
    """Render the generic 'Optimizations' section.

    Any speed row with a non-empty `optimization` is compared against the
    matching standard row (same hardware/name/dtype, optimization == "") to
    report a speedup. One sub-table per optimization method, driven by
    OPTIMIZATION_DISPLAY, so new methods need no bespoke rendering code.
    """
    if speed_df.empty or "optimization" not in speed_df.columns:
        return ""

    optimized = speed_df[speed_df["optimization"].astype(str) != ""].copy()
    if optimized.empty:
        return ""

    # Each optimized row is compared against the standard row measured on the
    # SAME prompt profile (spec-dec speedup depends heavily on the workload).
    standard = (
        speed_df[speed_df["optimization"].astype(str) == ""][
            ["hardware", "name", "dtype", "prompt_profile", "generation_tps"]
        ].rename(columns={"generation_tps": "standard_tps"})
    )

    lines = ["## Optimizations", ""]
    lines.append(
        "Techniques layered on top of the standard config above. They trade "
        "extra setup for speed/memory gains and are opt-in — not every model or "
        "architecture supports each one — so they're reported separately rather "
        "than mixed into the main ranking. Speedup is vs. that model's standard "
        "row on the same hardware **and prompt profile**: `generic` uses the "
        "standard benchmark prompt — general text, near worst-case acceptance "
        "for speculative decoding — while `code` uses a code-continuation "
        "prompt, closer to agentic-coding use."
    )
    lines.append("")

    for opt_name, opt_df in optimized.groupby("optimization"):
        meta = OPTIMIZATION_DISPLAY.get(
            opt_name,
            {
                "title": opt_name.replace("_", " ").title(),
                "blurb": "",
                "config_header": "Config",
                "config_fn": lambda d: d or "--",
            },
        )
        merged = opt_df.merge(
            standard, on=["hardware", "name", "dtype", "prompt_profile"], how="left"
        )
        merged["speedup"] = merged["generation_tps"] / merged["standard_tps"]
        # Group each model's profiles together, best speedup first.
        merged = merged.sort_values(
            ["name", "prompt_profile"], ascending=[True, True]
        )

        lines.append(f"### ⚡ {meta['title']}")
        lines.append("")
        if meta.get("blurb"):
            lines.append(meta["blurb"])
            lines.append("")
        lines.append(
            f"| Model | {meta['config_header']} | Prompt | Hardware | Standard "
            "| + Optimized | Speedup |"
        )
        lines.append("|---|---|---|---|---:|---:|---:|")

        config_fn = meta.get("config_fn", lambda d: d or "--")
        for _, r in merged.iterrows():
            name = format_model_name(r["name"], include_arch=True)
            cfg = config_fn(r.get("optimization_detail", ""))
            profile = "code" if r.get("prompt_profile") == "code" else "generic"
            hw = HARDWARE_DISPLAY.get(r["hardware"], r["hardware"])
            std = (
                f"{r['standard_tps']:.0f} tok/s"
                if pd.notna(r["standard_tps"])
                else "--"
            )
            opt = f"{r['generation_tps']:.0f} tok/s"
            spd = f"{r['speedup']:.2f}×" if pd.notna(r["speedup"]) else "--"
            lines.append(
                f"| {name} | {cfg} | {profile} | {hw} | {std} | {opt} | {spd} |"
            )
        lines.append("")

    return "\n".join(lines)


def generate_tables(
    models: Optional[List[str]] = None,
) -> str:
    """Generate benchmark tables for all hardware profiles."""
    speed_df = load_speed_data(models)
    quality_df = load_quality_data(models)

    if speed_df.empty:
        return "No benchmark data found.\n"

    # The main tables use only the standard (unoptimized) config on the generic
    # prompt, so every model is compared on the same footing. Optimizations and
    # alternate prompt profiles get their own section.
    standard_speed = speed_df[
        (speed_df["optimization"].astype(str) == "")
        & (speed_df["prompt_profile"].astype(str) == "generic")
    ].copy()

    # Determine hardware profiles with data, sorted by preference
    hardware_order = [
        "Apple_M4_Pro_10P+4E+20GPU_64GB",
        "Apple_M5_Max_XP+XE+40GPU_128GB",
        "Apple_M4_Pro_10P+4E+20GPU_24GB",
    ]
    available_hw = standard_speed["hardware"].unique()
    hardware_profiles = [h for h in hardware_order if h in available_hw]
    for h in sorted(available_hw):
        if h not in hardware_profiles:
            hardware_profiles.append(h)

    today = datetime.now().strftime("%B %Y")

    # Problem count from the canonical suite (variants replace base problems,
    # so the suite size is always the base count).
    from mtb.quality_benchmarks import (
        EVAL_PROBLEMS,
        EXPERT_EVAL_PROBLEMS,
        HARD_EVAL_PROBLEMS,
        TOOL_CALLING_PROBLEMS,
    )

    num_problems = (
        len(EVAL_PROBLEMS)
        + len(HARD_EVAL_PROBLEMS)
        + len(EXPERT_EVAL_PROBLEMS)
        + len(TOOL_CALLING_PROBLEMS)
    )

    lines = []
    lines.append(f"> MLX Metal | int4 quantization | {today}")
    lines.append("> Speed: 1024 prompt tokens, 100 generated tokens")
    lines.append(
        f"> Quality: {num_problems} problems across coding, reasoning, tool calling, math, writing (3 runs each, majority vote)"
    )
    lines.append(
        "> **API baseline:** Claude Opus 4.6 scores 86.7% on the same quality benchmark (via Anthropic API, not local)"
    )
    # Footnote for rows run with a raised token-budget multiplier (only shown
    # when some model used one). The cap is a ceiling, not forced length.
    if (
        not quality_df.empty
        and "_max_tokens_multiplier" in quality_df.columns
        and (quality_df["_max_tokens_multiplier"] > 1).any()
    ):
        lines.append(
            "> † Run with a raised per-problem token-budget multiplier "
            "(`--max_tokens_multiplier`) so its verbose thinking can finish "
            "within budget; the cap is a ceiling, not forced length. Other "
            "rows use the default 1×, so this score is not strictly "
            "budget-comparable. Each run's multiplier is recorded in its "
            "`settings.json`."
        )
    lines.append("")

    # Cross-hardware summary at top
    # Only include primary hardware profiles (skip older/legacy)
    primary_hw = [h for h in hardware_profiles if "24GB" not in h]
    if len(primary_hw) >= 1:
        lines.append(
            generate_cross_hardware_summary(standard_speed, quality_df, primary_hw)
        )
        lines.append("")

    # Hardware profiles that still use the older (smaller) quality suite
    OLD_QUALITY_SUITE_HW = set()

    # Per-hardware detailed tables
    for hw in hardware_profiles:
        hw_display = HARDWARE_DISPLAY.get(hw, hw)
        is_legacy = "24GB" in hw

        if is_legacy:
            lines.append(f"<details>")
            lines.append(f"<summary><h3>{hw_display} (legacy)</h3></summary>")
            lines.append("")
        else:
            lines.append(f"### {hw_display}")
            lines.append("")
            if hw in OLD_QUALITY_SUITE_HW:
                lines.append(
                    "> **Note:** Quality scores on this hardware use the older "
                    "46-problem suite and are not directly comparable to M4 Pro "
                    "scores. A re-run with the full 81-problem suite is planned."
                )
                lines.append("")

        table = generate_hardware_table(standard_speed, quality_df, hw)
        if table:
            lines.append(table)
        else:
            lines.append("No data available.")

        if is_legacy:
            lines.append("")
            lines.append("</details>")

        lines.append("")

    # Optimizations section (only rendered if any optimized rows exist).
    optimizations = generate_optimizations_section(speed_df)
    if optimizations:
        lines.append(optimizations)
        lines.append("")

    return "\n".join(lines)


def update_readme(
    models: Optional[List[str]] = None,
    dry_run: bool = False,
):
    """Update the README.md benchmark section between markers."""
    readme = README_PATH.read_text()

    table_content = generate_tables(models)

    if BEGIN_MARKER in readme and END_MARKER in readme:
        pattern = re.compile(
            re.escape(BEGIN_MARKER) + r".*?" + re.escape(END_MARKER),
            re.DOTALL,
        )
        new_readme = pattern.sub(
            f"{BEGIN_MARKER}\n\n{table_content}\n\n{END_MARKER}",
            readme,
        )
    else:
        old_section_start = "## Agentic Coding Model Benchmarks"
        if old_section_start in readme:
            start_idx = readme.index(old_section_start)
            heading_after = readme.find("\n## ", start_idx + len(old_section_start))
            if heading_after == -1:
                heading_after = len(readme)
            new_readme = (
                readme[:start_idx]
                + f"## Agentic Coding Model Benchmarks (MLX on Apple Silicon)\n\n"
                + f"{BEGIN_MARKER}\n\n{table_content}\n\n{END_MARKER}\n\n"
                + readme[heading_after + 1 :]
            )
        else:
            new_readme = (
                readme
                + f"\n\n## Agentic Coding Model Benchmarks (MLX on Apple Silicon)\n\n{BEGIN_MARKER}\n\n{table_content}\n\n{END_MARKER}\n"
            )

    if dry_run:
        print("=== DRY RUN — would write: ===")
        print(table_content)
    else:
        README_PATH.write_text(new_readme)
        print(f"Updated {README_PATH}")


if __name__ == "__main__":
    fire.Fire(update_readme)
