"""Generate the GitHub social preview image (1280x640).

Renders the speed-vs-quality scatter with the repo title overlay.
Upload the output to repo Settings > Social preview.

Usage:
    uv run python scripts/generate_social_preview.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.update_readme_table import (
    HARDWARE_DISPLAY,
    format_model_name,
    get_arch,
    load_quality_data,
    load_speed_data,
)
from mtb.quality_benchmarks.scoring import (
    compute_weighted_score,
    _build_problem_tier_map,
    _resolve_variant_name,
)

OUT_PATH = Path(__file__).parent.parent / "social-preview.png"

BG = "#0d1117"
FG = "#e6edf3"
GRID = "#21262d"
MUTED = "#8b949e"
MOE_COLOR = "#58a6ff"
DENSE_COLOR = "#f78166"
GREEN = "#3fb950"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "axes.edgecolor": GRID,
    "axes.labelcolor": FG,
    "text.color": FG,
    "xtick.color": FG,
    "ytick.color": FG,
    "grid.color": GRID,
    "grid.alpha": 0.2,
    "font.family": "sans-serif",
    "font.size": 14,
})


def main():
    speed_df = load_speed_data()
    quality_df = load_quality_data()

    hw = "Apple_M4_Pro_10P+4E+20GPU_64GB"
    dtype = "int4"

    standard_speed = speed_df[
        (speed_df["optimization"].astype(str) == "")
        & (speed_df["prompt_profile"].astype(str) == "generic")
        & (speed_df["hardware"] == hw)
        & (speed_df["dtype"] == dtype)
    ]

    tier_map = _build_problem_tier_map()
    models = []

    for _, row in standard_speed.iterrows():
        name = row["name"]
        q_mask = (
            (quality_df["model"] == name)
            & (quality_df["hardware"] == hw)
            & (quality_df["dtype"] == dtype)
        )
        q_rows = quality_df[q_mask]
        if q_rows.empty:
            continue

        results = {r["problem"]: bool(r["passed"]) for _, r in q_rows.iterrows()}
        recognized = sum(1 for n in results if _resolve_variant_name(n) in tier_map)
        if recognized / max(len(results), 1) < 0.5:
            continue

        score = compute_weighted_score(results)
        quality_pct = round(score["weighted_score"] * 100, 1)
        arch = get_arch(name)
        display = format_model_name(name, include_arch=False)

        models.append({
            "name": name,
            "display": display,
            "arch": arch,
            "tps": row["generation_tps"],
            "quality": quality_pct,
            "memory": row["peak_memory_gib"],
        })

    fig, ax = plt.subplots(figsize=(12.8, 6.4), dpi=100)

    moe = [m for m in models if "MoE" in m["arch"]]
    dense = [m for m in models if "MoE" not in m["arch"]]

    # Plot dense
    if dense:
        ax.scatter(
            [m["tps"] for m in dense],
            [m["quality"] for m in dense],
            c=DENSE_COLOR, s=90, alpha=0.9, zorder=3, label="Dense",
            edgecolors="none",
        )

    # Plot MoE
    if moe:
        ax.scatter(
            [m["tps"] for m in moe],
            [m["quality"] for m in moe],
            c=MOE_COLOR, s=110, alpha=0.9, zorder=3, label="MoE",
            marker="D", edgecolors="none",
        )

    # Label top models
    top = sorted(models, key=lambda m: m["quality"], reverse=True)[:5]
    fastest_qualified = sorted(
        [m for m in models if m["quality"] >= 60],
        key=lambda m: m["tps"], reverse=True
    )[:2]
    to_label = {m["name"] for m in top + fastest_qualified}

    for m in models:
        if m["name"] in to_label:
            color = MOE_COLOR if "MoE" in m["arch"] else DENSE_COLOR
            ax.annotate(
                m["display"],
                (m["tps"], m["quality"]),
                xytext=(6, 6), textcoords="offset points",
                fontsize=10, color=color, alpha=0.9,
                fontweight="bold",
            )

    # Threshold lines
    ax.axvline(50, color=MUTED, linestyle=":", linewidth=0.8, alpha=0.5)
    ax.axhline(60, color=MUTED, linestyle=":", linewidth=0.8, alpha=0.5)

    # Sweet spot shading
    xlim = ax.get_xlim()
    ax.fill_between(
        [50, xlim[1] if xlim[1] > 50 else 300],
        60, 100,
        alpha=0.03, color=GREEN, zorder=0,
    )

    ax.set_xlabel("Generation Speed (tok/s)", fontsize=13)
    ax.set_ylabel("Quality (%)", fontsize=13)
    ax.set_ylim(10, 100)
    ax.legend(loc="lower right", fontsize=11, framealpha=0.3)
    ax.grid(True, alpha=0.15)

    # Title overlay
    fig.text(
        0.03, 0.95, "mlx-coding-bench",
        fontsize=26, fontweight="bold", color=FG,
        va="top", ha="left",
    )
    fig.text(
        0.03, 0.88, "Speed vs Quality — Local LLMs for Coding on Apple Silicon",
        fontsize=13, color=MUTED,
        va="top", ha="left",
    )
    fig.text(
        0.97, 0.04, "weklund.github.io/mlx-coding-bench",
        fontsize=10, color=MUTED, alpha=0.7,
        va="bottom", ha="right",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.85])
    plt.savefig(OUT_PATH, dpi=100, bbox_inches="tight", pad_inches=0.2)
    print(f"Saved: {OUT_PATH} ({OUT_PATH.stat().st_size // 1024} KB)")
    print(f"Upload to: https://github.com/weklund/mlx-coding-bench/settings")


if __name__ == "__main__":
    main()
