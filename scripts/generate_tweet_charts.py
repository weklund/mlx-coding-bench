"""Generate billboard-style dark-mode charts for X/Twitter thread.

Each chart conveys ONE idea, readable at thumbnail size.
Fewer data points, much bigger text, more whitespace.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from pathlib import Path

# Output directory
OUT_DIR = Path(__file__).parent.parent / "charts"
OUT_DIR.mkdir(exist_ok=True)

# ── Global style ──────────────────────────────────────────────────────────────
BG = "#0d1117"
FG = "#e6edf3"
GRID = "#21262d"
ACCENT = "#58a6ff"
ACCENT2 = "#f78166"
ACCENT3 = "#3fb950"
ACCENT4 = "#d2a8ff"
MOE_COLOR = "#58a6ff"
DENSE_COLOR = "#f78166"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "axes.edgecolor": GRID,
    "axes.labelcolor": FG,
    "text.color": FG,
    "xtick.color": FG,
    "ytick.color": FG,
    "grid.color": GRID,
    "grid.alpha": 0.3,
    "font.family": "sans-serif",
    "font.size": 18,
})


def chart1a_two_bar():
    """Tweet 1 option A: Simple two-bar face-off with 7.6x badge."""
    fig, ax = plt.subplots(figsize=(19.2, 10.8))

    # The two models
    names = ["Qwen3-Coder 30B-A3B\n(3B active MoE)", "Gemma 4 31B-it\n(31B dense)"]
    speeds = [129, 17]
    quals = [75.5, 72.7]
    colors = [MOE_COLOR, DENSE_COLOR]

    bars = ax.barh([1, 0], speeds, height=0.55, color=colors, alpha=0.9,
                   edgecolor="white", linewidth=1)

    ax.set_yticks([1, 0])
    ax.set_yticklabels(names, fontsize=22, fontweight="bold")

    # Speed labels on bars
    for i, (s, q) in enumerate(zip(speeds, quals)):
        y = 1 - i
        ax.text(s - 3, y, f"{s} tok/s", ha="right", va="center",
                fontsize=26, fontweight="bold", color="white")
        ax.text(s + 3, y, f"quality: {q}%", ha="left", va="center",
                fontsize=18, color=FG, alpha=0.7)

    # Giant 7.6x badge in the center
    ax.text(73, 0.5, "7.6x", fontsize=72, fontweight="bold", color=ACCENT3,
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=BG,
                      edgecolor=ACCENT3, linewidth=4, alpha=0.95))
    ax.text(73, 0.12, "faster", fontsize=24, fontweight="bold", color=ACCENT3,
            ha="center", va="center", alpha=0.8)

    ax.set_xlabel("Generation Speed (tok/s)", fontsize=22, fontweight="bold")
    ax.set_title("A 3B Model Beats a 31B Model",
                 fontsize=32, fontweight="bold", pad=24)
    ax.tick_params(labelsize=16)
    ax.set_xlim(0, 155)
    ax.set_ylim(-0.5, 1.8)
    ax.grid(True, axis="x", alpha=0.3)

    # Subtitle
    ax.text(77, 1.55, "Same quality. A fraction of the compute.",
            fontsize=20, ha="center", color=FG, alpha=0.5, fontstyle="italic")

    plt.tight_layout()
    fig.savefig(OUT_DIR / "1a_two_bar.png", dpi=100, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUT_DIR / '1a_two_bar.png'}")


def chart1b_scoreboard():
    """Tweet 1 option B: Sports-style matchup scoreboard, no axes."""
    fig = plt.figure(figsize=(19.2, 10.8))
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Subtle card background
    from matplotlib.patches import FancyBboxPatch
    card = FancyBboxPatch((1, 1), 98, 98, boxstyle="round,pad=1.5",
                          facecolor="#161b22", edgecolor="#30363d",
                          linewidth=2, alpha=0.9)
    ax.add_patch(card)

    # Title -- bigger, pushed down from top edge
    ax.text(50, 91, "Google's New Gemma 4 31B Lost to a 3B Model", fontsize=36,
            fontweight="bold", ha="center", va="center", color=FG)
    ax.text(50, 84.5, "Agentic coding benchmarks  ·  Apple M5 Max  ·  int4",
            fontsize=17, ha="center", va="center", color=FG, alpha=0.45)

    # VS badge in center -- larger
    ax.text(50, 55, "VS", fontsize=42, fontweight="bold", ha="center",
            va="center", color=FG, alpha=0.25)

    # Left model (MoE - winner)
    ax.text(25, 75, "Qwen3-Coder", fontsize=32, fontweight="bold",
            ha="center", va="center", color=MOE_COLOR)
    ax.text(25, 68, "30B-A3B", fontsize=24, ha="center", va="center",
            color=MOE_COLOR, alpha=0.8)
    ax.text(25, 62, "3B active  ·  MoE", fontsize=17, ha="center",
            va="center", color=FG, alpha=0.5)

    # Left stats -- bigger numbers, tighter spacing
    ax.text(25, 48, "129", fontsize=68, fontweight="bold", ha="center",
            va="center", color=MOE_COLOR)
    ax.text(25, 39, "tok/s", fontsize=22, ha="center", va="center",
            color=FG, alpha=0.6)
    ax.text(25, 29, "75.5%", fontsize=42, fontweight="bold", ha="center",
            va="center", color=MOE_COLOR)
    ax.text(25, 22, "quality", fontsize=20, ha="center", va="center",
            color=FG, alpha=0.6)
    ax.text(25, 15, "17.8 GB RAM", fontsize=17, ha="center", va="center",
            color=FG, alpha=0.4)

    # Right model (Dense)
    ax.text(75, 75, "Gemma 4", fontsize=32, fontweight="bold",
            ha="center", va="center", color=DENSE_COLOR)
    ax.text(75, 68, "31B-it", fontsize=24, ha="center", va="center",
            color=DENSE_COLOR, alpha=0.8)
    ax.text(75, 62, "31B  ·  Dense", fontsize=17, ha="center",
            va="center", color=FG, alpha=0.5)

    # Right stats
    ax.text(75, 48, "17", fontsize=68, fontweight="bold", ha="center",
            va="center", color=DENSE_COLOR)
    ax.text(75, 39, "tok/s", fontsize=22, ha="center", va="center",
            color=FG, alpha=0.6)
    ax.text(75, 29, "72.7%", fontsize=42, fontweight="bold", ha="center",
            va="center", color=DENSE_COLOR)
    ax.text(75, 22, "quality", fontsize=20, ha="center", va="center",
            color=FG, alpha=0.6)
    ax.text(75, 15, "18.9 GB RAM", fontsize=17, ha="center", va="center",
            color=FG, alpha=0.4)

    # 7.6x callout at bottom -- bigger, bolder
    ax.text(50, 7, "7.6x faster  ·  same quality  ·  same RAM",
            fontsize=24, fontweight="bold", ha="center", va="center",
            color=ACCENT3)

    # Divider line
    ax.plot([50, 50], [12, 80], color=FG, alpha=0.08, lw=2.5)

    fig.savefig(OUT_DIR / "1b_scoreboard.png", dpi=100, bbox_inches="tight",
                facecolor=BG)
    plt.close()
    print(f"Saved: {OUT_DIR / '1b_scoreboard.png'}")


def chart1_scatter():
    """Tweet 1: Quality vs Speed -- only the 6 most interesting models."""
    fig, ax = plt.subplots(figsize=(19.2, 10.8))

    # Only show key models that tell the story
    spotlight = [
        ("Qwen3-Coder\n30B-A3B", 75.5, 129, 17.8, "MoE", "3B active"),
        ("Gemma 4\n31B-it",       72.7, 17,  18.9, "Dense", "31B"),
        ("LFM2-24B\nA2B",        70.6, 180, 14.2, "MoE", "2B active"),
        ("Gemma 4\nE2B-it",      67.8, 205, 3.5,  "Dense", "2.3B"),
        ("Qwen 3.5\n9B",         64.5, 79,  7.3,  "Dense", "9B"),
        ("Qwen 3.5\n0.8B",       24.9, 409, 2.5,  "Dense", "0.8B"),
    ]

    for name, qual, speed, ram, mtype, active in spotlight:
        color = MOE_COLOR if mtype == "MoE" else DENSE_COLOR
        size = max(ram * 18, 150)
        ax.scatter(qual, speed, s=size, c=color, alpha=0.9,
                   edgecolors="white", linewidths=1, zorder=3)

    # Labels with manual positioning for zero overlap
    labels = {
        "Qwen3-Coder\n30B-A3B": (10, 20, "left"),
        "Gemma 4\n31B-it":      (10, -22, "left"),
        "LFM2-24B\nA2B":        (-10, 18, "right"),
        "Gemma 4\nE2B-it":      (-10, 18, "right"),
        "Qwen 3.5\n9B":         (10, -20, "left"),
        "Qwen 3.5\n0.8B":       (-10, 16, "right"),
    }

    for name, qual, speed, ram, mtype, active in spotlight:
        ox, oy, ha = labels[name]
        display = name.replace("\n", " ")
        label_text = f"{display}\n({active})"
        ax.annotate(label_text, (qual, speed),
                    xytext=(ox, oy), textcoords="offset points",
                    fontsize=16, ha=ha, va="center", color=FG,
                    fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=FG, alpha=0.4, lw=1))

    # Legend
    ax.scatter([], [], c=MOE_COLOR, s=200, label="MoE", edgecolors="white", linewidths=1)
    ax.scatter([], [], c=DENSE_COLOR, s=200, label="Dense", edgecolors="white", linewidths=1)
    ax.legend(loc="upper right", fontsize=20, framealpha=0.3, edgecolor=GRID)

    ax.set_xlabel("Quality Score (%)", fontsize=22, fontweight="bold")
    ax.set_ylabel("Speed (tok/s)", fontsize=22, fontweight="bold")
    ax.set_title("Local LLMs on M5 Max — Quality vs Speed",
                 fontsize=28, fontweight="bold", pad=24)
    ax.tick_params(labelsize=16)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(18, 82)
    ax.set_ylim(-20, 450)

    # Sweet spot annotation
    from matplotlib.patches import FancyBboxPatch
    rect = FancyBboxPatch((62, 65), 18, 200, boxstyle="round,pad=3",
                          facecolor=ACCENT3, alpha=0.06, edgecolor=ACCENT3,
                          linewidth=2, linestyle="--")
    ax.add_patch(rect)
    ax.text(71, 270, "Sweet\nSpot", fontsize=18, ha="center", va="center",
            color=ACCENT3, alpha=0.5, fontweight="bold")

    plt.tight_layout()
    fig.savefig(OUT_DIR / "1_quality_vs_speed.png", dpi=100, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUT_DIR / '1_quality_vs_speed.png'}")


def chart2_top_quality():
    """Tweet 2: Top 5 models by quality -- big bars, big text."""
    fig, ax = plt.subplots(figsize=(19.2, 10.8))

    data = [
        ("Qwen3-Coder 30B-A3B", 75.5, 129, "MoE", "3B active"),
        ("Gemma 4 31B-it",       72.7, 17,  "Dense", "31B"),
        ("Qwen 3.5-27B",         71.0, 25,  "Dense", "27B"),
        ("LFM2-24B-A2B",         70.6, 180, "MoE", "2B active"),
        ("Gemma 4 E2B-it",       67.8, 205, "Dense", "2.3B"),
    ]
    data.reverse()

    names = [d[0] for d in data]
    quals = [d[1] for d in data]
    speeds = [d[2] for d in data]
    colors = [MOE_COLOR if d[3] == "MoE" else DENSE_COLOR for d in data]
    archs = [d[4] for d in data]

    bars = ax.barh(range(len(names)), quals, color=colors, alpha=0.9,
                   edgecolor="white", linewidth=0.8, height=0.6)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=20, fontweight="bold")
    ax.set_xlabel("Quality Score (%)", fontsize=22, fontweight="bold")
    ax.set_title("Top 5 Local LLMs for Agentic Coding",
                 fontsize=28, fontweight="bold", pad=24)
    ax.tick_params(labelsize=16)

    for i, (q, s, arch) in enumerate(zip(quals, speeds, archs)):
        ax.text(q - 1.5, i, f"{q}%", ha="right", va="center",
                fontsize=20, fontweight="bold", color="white")
        ax.text(q + 1.5, i, f"{s} tok/s  ·  {arch}", ha="left", va="center",
                fontsize=16, color=FG, alpha=0.8)

    ax.set_xlim(0, 95)
    ax.grid(True, axis="x", alpha=0.3)

    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=MOE_COLOR, label="MoE"),
                       Patch(facecolor=DENSE_COLOR, label="Dense")],
              loc="lower right", fontsize=20, framealpha=0.3, edgecolor=GRID)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "2_top_quality.png", dpi=100, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUT_DIR / '2_top_quality.png'}")


def chart3_e2b_comparison():
    """Tweet 3: Just two models side by side -- E2B ties Qwen3-Coder on coding."""
    fig, ax = plt.subplots(figsize=(19.2, 10.8))

    # Just the two models
    models = [
        ("Gemma 4 E2B-it", 85.3, "2.3B dense", "205 tok/s", "3.5 GB", ACCENT3),
        ("Qwen3-Coder 30B-A3B", 85.3, "3B MoE (30B total)", "129 tok/s", "17.8 GB", MOE_COLOR),
    ]

    y_pos = [1, 0]

    for i, (name, score, arch, speed, ram, color) in enumerate(models):
        ax.barh(y_pos[i], score, height=0.5, color=color, alpha=0.9,
                edgecolor="white", linewidth=1)
        ax.text(score - 2, y_pos[i], f"{score}%", ha="right", va="center",
                fontsize=28, fontweight="bold", color="white")

    # Model labels on y-axis
    ax.set_yticks(y_pos)
    ax.set_yticklabels(["", ""])  # Clear default

    # Custom labels with details
    ax.text(-2, 1, "Gemma 4 E2B-it\n2.3B dense  ·  205 tok/s  ·  3.5 GB",
            ha="right", va="center", fontsize=18, color=ACCENT3, fontweight="bold")
    ax.text(-2, 0, "Qwen3-Coder 30B-A3B\n3B MoE  ·  129 tok/s  ·  17.8 GB",
            ha="right", va="center", fontsize=18, color=MOE_COLOR, fontweight="bold")

    # TIE badge
    ax.text(87, 0.5, "TIE", fontsize=36, fontweight="bold", color=ACCENT3,
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=BG, edgecolor=ACCENT3,
                      linewidth=3, alpha=0.9))

    # Bracket
    ax.plot([86, 86], [0.15, 0.85], color=ACCENT3, lw=3)
    ax.plot([85.5, 86], [0.15, 0.15], color=ACCENT3, lw=3)
    ax.plot([85.5, 86], [0.85, 0.85], color=ACCENT3, lw=3)

    ax.set_xlabel("Coding Score (%)", fontsize=22, fontweight="bold")
    ax.set_title("A 2.3B Model Ties a 30B Model on Coding",
                 fontsize=28, fontweight="bold", pad=24)
    ax.tick_params(labelsize=16)
    ax.set_xlim(-35, 100)
    ax.set_ylim(-0.5, 1.7)
    ax.grid(True, axis="x", alpha=0.3)

    # Subtitle
    ax.text(50, 1.45, "13x smaller  ·  1.6x faster  ·  5x less RAM",
            ha="center", va="center", fontsize=20, color=FG, alpha=0.6,
            fontweight="bold")

    plt.tight_layout()
    fig.savefig(OUT_DIR / "3_e2b_coding.png", dpi=100, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUT_DIR / '3_e2b_coding.png'}")


def chart4_moe_vs_dense():
    """Tweet 4: MoE vs Dense -- one pair, huge visual."""
    fig, ax = plt.subplots(figsize=(19.2, 10.8))

    # Three pairs but bigger spacing
    pairs = [
        ("Qwen3-Coder 30B-A3B\n3B active", 75.5, 129,
         "Gemma 4 31B-it\n31B dense", 72.7, 17),
        ("LFM2-24B-A2B\n2B active", 70.6, 180,
         "Qwen 3.5-27B\n27B dense", 71.0, 25),
        ("Gemma 4 26B-A4B\n4B active", 65.3, 110,
         "Qwen 3.5-9B\n9B dense", 64.5, 79),
    ]

    y_positions = np.arange(len(pairs)) * 4
    bar_height = 1.0

    for i, (moe_name, moe_q, moe_s, dense_name, dense_q, dense_s) in enumerate(pairs):
        y = y_positions[i]

        # MoE bar
        ax.barh(y + 0.7, moe_s, height=bar_height, color=MOE_COLOR, alpha=0.9,
                edgecolor="white", linewidth=0.8)
        ax.text(moe_s + 5, y + 0.7, f"{moe_s} tok/s  ({moe_q}%)",
                va="center", fontsize=18, color=MOE_COLOR, fontweight="bold")
        ax.text(-5, y + 0.7, moe_name, va="center", ha="right",
                fontsize=16, color=FG, fontweight="bold")

        # Dense bar
        ax.barh(y - 0.7, dense_s, height=bar_height, color=DENSE_COLOR, alpha=0.9,
                edgecolor="white", linewidth=0.8)
        ax.text(dense_s + 5, y - 0.7, f"{dense_s} tok/s  ({dense_q}%)",
                va="center", fontsize=18, color=DENSE_COLOR, fontweight="bold")
        ax.text(-5, y - 0.7, dense_name, va="center", ha="right",
                fontsize=16, color=FG, fontweight="bold")

        # Speed multiplier badge -- THE focal point
        mult = round(moe_s / dense_s, 1)
        ax.text(max(moe_s, dense_s) * 0.45, y, f"{mult}x",
                fontsize=32, fontweight="bold", color=ACCENT3,
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.5", facecolor=BG,
                          edgecolor=ACCENT3, linewidth=3, alpha=0.95))

    ax.set_xlabel("Generation Speed (tok/s)", fontsize=22, fontweight="bold")
    ax.set_title("MoE vs Dense: Same Quality, Wildly Different Speed",
                 fontsize=28, fontweight="bold", pad=24)
    ax.tick_params(labelsize=16)
    ax.set_yticks([])
    ax.set_xlim(-110, 310)
    ax.grid(True, axis="x", alpha=0.3)

    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=MOE_COLOR, label="MoE (active params)"),
                       Patch(facecolor=DENSE_COLOR, label="Dense")],
              loc="lower right", fontsize=20, framealpha=0.3, edgecolor=GRID)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "4_moe_vs_dense.png", dpi=100, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUT_DIR / '4_moe_vs_dense.png'}")


def chart5_tool_calling():
    """Tweet 5: Tool calling gap -- just 4 models that tell the story."""
    fig, ax = plt.subplots(figsize=(19.2, 10.8))

    # Only models that illustrate the gap dramatically
    data = [
        ("Gemma 4 31B-it",       80.0, 52.9),   # high TC, lower coding
        ("Qwen3-Coder 30B-A3B",  75.0, 85.3),   # high both
        ("Nemotron-Nano-9B-v2",  22.5, 47.1),   # math beast, TC disaster
        ("Gemma 3-4B-it",        20.0, 82.4),   # coding great, TC disaster
    ]

    names = [d[0] for d in data]
    tc = [d[1] for d in data]
    coding = [d[2] for d in data]

    y = np.arange(len(names))
    bar_height = 0.35

    bars_tc = ax.barh(y + bar_height/2, tc, bar_height, label="Tool Calling",
                      color=ACCENT, alpha=0.9, edgecolor="white", linewidth=0.8)
    bars_cod = ax.barh(y - bar_height/2, coding, bar_height, label="Coding",
                       color=ACCENT4, alpha=0.7, edgecolor="white", linewidth=0.8)

    # Score labels on bars
    for i in range(len(names)):
        # Tool calling score
        ax.text(tc[i] - 2, y[i] + bar_height/2, f"{tc[i]}%",
                ha="right", va="center", fontsize=18, fontweight="bold", color="white")
        # Coding score
        ax.text(coding[i] - 2, y[i] - bar_height/2, f"{coding[i]}%",
                ha="right", va="center", fontsize=18, fontweight="bold", color="white")

    # Gap annotations for dramatic cases
    for i, (name, t, c) in enumerate(data):
        gap = c - t
        if gap > 40:
            ax.annotate(f"{gap:.0f}pp gap",
                        xy=(max(t, c) + 3, y[i]),
                        fontsize=18, fontweight="bold", color=ACCENT2,
                        va="center")

    # 60% threshold
    ax.axvline(x=60, color=ACCENT3, linestyle="--", linewidth=2.5, alpha=0.7)
    ax.text(57, len(names) - 0.5, "60%", ha="right",
            fontsize=18, color=ACCENT3, fontweight="bold", va="center", alpha=0.8)

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=20, fontweight="bold")
    ax.set_xlabel("Weighted Score (%)", fontsize=22, fontweight="bold")
    ax.set_title("Tool Calling: Where Models Fall Apart",
                 fontsize=28, fontweight="bold", pad=24)
    ax.tick_params(labelsize=16)
    ax.legend(fontsize=20, framealpha=0.3, edgecolor=GRID, loc="lower right")
    ax.grid(True, axis="x", alpha=0.3)
    ax.set_xlim(0, 105)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "5_tool_calling.png", dpi=100, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUT_DIR / '5_tool_calling.png'}")


if __name__ == "__main__":
    chart1a_two_bar()
    chart1b_scoreboard()
    chart1_scatter()
    chart2_top_quality()
    chart3_e2b_comparison()
    chart4_moe_vs_dense()
    chart5_tool_calling()
    print(f"\nAll charts saved to: {OUT_DIR}")
