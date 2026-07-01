"""Generate the mlx-coding-bench GitHub Pages site.

Produces a single-page interactive dashboard with:
1. Speed vs Quality scatter (the hero chart nobody else has)
2. Model rankings table with filtering
3. Hardware comparison
4. Per-category quality breakdowns

Usage:
    uv run python scripts/generate_site.py
    uv run python scripts/generate_site.py --output-dir site
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import fire
import pandas as pd

import mtb
from scripts.update_readme_table import (
    HARDWARE_DISPLAY,
    MODEL_META,
    format_model_name,
    get_arch,
    load_quality_data,
    load_speed_data,
)
from mtb.quality_benchmarks.scoring import compute_weighted_score, _build_problem_tier_map, _resolve_variant_name

REPO_ROOT = mtb.REPO_ROOT
SITE_DIR = REPO_ROOT / "site"


def _build_site_data() -> dict:
    """Load and merge speed + quality data into a JSON-serializable structure."""
    speed_df = load_speed_data()
    quality_df = load_quality_data()

    if speed_df.empty:
        return {"models": [], "hardware": [], "generated": ""}

    # Standard config only (no speculative decoding), generic prompt
    standard_speed = speed_df[
        (speed_df["optimization"].astype(str) == "")
        & (speed_df["prompt_profile"].astype(str) == "generic")
    ].copy()

    hardware_profiles = sorted(standard_speed["hardware"].unique())

    models = []
    seen = set()

    for hw in hardware_profiles:
        hw_speed = standard_speed[standard_speed["hardware"] == hw]
        hw_display = HARDWARE_DISPLAY.get(hw, hw)

        for dtype in ["int4", "int8"]:
            dtype_speed = hw_speed[hw_speed["dtype"] == dtype]

            for _, row in dtype_speed.iterrows():
                name = row["name"]
                key = (hw, name, dtype)
                if key in seen:
                    continue
                seen.add(key)

                display_name = format_model_name(name, include_arch=False)
                arch = get_arch(name)

                entry = {
                    "name": name,
                    "display_name": display_name,
                    "arch": arch,
                    "hardware": hw_display,
                    "hardware_key": hw,
                    "dtype": dtype,
                    "gen_tps": round(row["generation_tps"], 1),
                    "prefill_tps": round(row["prompt_tps"], 0),
                    "memory_gib": round(row["peak_memory_gib"], 1),
                    "quality_pct": None,
                    "category_scores": {},
                }

                # Merge quality data
                if not quality_df.empty:
                    q_mask = (
                        (quality_df["model"] == name)
                        & (quality_df["hardware"] == hw)
                        & (quality_df["dtype"] == dtype)
                    )
                    q_rows = quality_df[q_mask]
                    if not q_rows.empty:
                        results = {
                            r["problem"]: bool(r["passed"])
                            for _, r in q_rows.iterrows()
                        }
                        tier_map = _build_problem_tier_map()
                        recognized = sum(
                            1 for n in results if _resolve_variant_name(n) in tier_map
                        )
                        if recognized / max(len(results), 1) >= 0.5:
                            score = compute_weighted_score(results)
                            entry["quality_pct"] = round(
                                score["weighted_score"] * 100, 1
                            )
                            entry["category_scores"] = {
                                k: round(v * 100, 1)
                                for k, v in score.get("category_scores", {}).items()
                            }

                models.append(entry)

    return {
        "models": models,
        "hardware": [
            {"key": hw, "display": HARDWARE_DISPLAY.get(hw, hw)}
            for hw in hardware_profiles
        ],
        "generated": datetime.now().strftime("%Y-%m-%d"),
    }


def _render_html(data: dict) -> str:
    """Render the single-page site HTML using Pico CSS."""
    data_json = json.dumps(data, indent=None)
    generated = data["generated"]

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>mlx-coding-bench — Best Local LLM for Coding on Mac</title>
    <meta name="description" content="Speed and quality benchmarks for local LLMs on Apple Silicon. Find the best model for agentic coding on your Mac.">
    <meta property="og:title" content="mlx-coding-bench — Best Local LLM for Coding on Mac">
    <meta property="og:description" content="Which local LLM runs best for coding on your Mac? Interactive speed vs quality benchmarks for MLX models on Apple Silicon.">
    <meta property="og:type" content="website">
    <meta name="twitter:card" content="summary_large_image">
    <meta name="twitter:title" content="mlx-coding-bench — Best Local LLM for Coding on Mac">
    <meta name="twitter:description" content="Speed + quality benchmarks for 35+ local LLMs on Apple Silicon. The only benchmark that measures both.">

    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@picocss/pico@2/css/pico.min.css">
    <script>
        // Apply theme before render to avoid flash
        (function() {{
            const saved = localStorage.getItem('theme');
            const system = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
            document.documentElement.setAttribute('data-theme', saved || system);
        }})();
    </script>
    <script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>

    <style>
        :root {{
            --pico-font-size: 16px;
        }}
        body {{ padding: 0; }}
        header {{ text-align: center; padding: 2rem 0 1rem; }}
        header h1 {{ margin-bottom: 0.25rem; }}
        header p {{ opacity: 0.7; max-width: 38rem; margin: 0 auto; }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 1rem;
            margin: 2rem 0;
        }}
        .stat-card {{
            text-align: center;
            padding: 1.25rem 1rem;
            border-radius: var(--pico-border-radius);
            background: var(--pico-card-background-color);
            border: 1px solid var(--pico-muted-border-color);
        }}
        .stat-card .value {{
            font-size: 1.75rem;
            font-weight: 700;
            color: var(--pico-primary);
            line-height: 1.2;
        }}
        .stat-card .label {{
            font-size: 0.8rem;
            opacity: 0.65;
            margin-top: 0.25rem;
        }}

        .filters {{
            display: flex;
            gap: 1rem;
            flex-wrap: wrap;
            align-items: end;
            padding: 0.75rem 1rem;
            margin: 0 -1rem 1.5rem;
            position: sticky;
            top: 0;
            z-index: 100;
            background: var(--pico-background-color);
            border-bottom: 1px solid var(--pico-muted-border-color);
            border-radius: var(--pico-border-radius);
            transition: box-shadow 0.2s;
        }}
        .filters.stuck {{
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
            border-radius: 0;
            margin: 0 -1.5rem 1.5rem;
            padding: 0.75rem 1.5rem;
        }}
        .filters label {{
            margin-bottom: 0;
            font-size: 0.85rem;
        }}
        .filters select {{
            margin-bottom: 0;
            padding: 0.4rem 0.75rem;
            width: auto;
            min-width: 120px;
        }}

        .chart-wrap {{
            border-radius: var(--pico-border-radius);
            overflow: hidden;
            margin-bottom: 2rem;
        }}

        .table-wrap {{
            overflow-x: auto;
            margin-bottom: 2rem;
        }}
        .table-wrap table {{
            font-size: 0.875rem;
            white-space: nowrap;
        }}
        .table-wrap th {{
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.03em;
            opacity: 0.7;
        }}
        td.r {{ text-align: right; font-variant-numeric: tabular-nums; }}

        .quality-bar {{
            display: inline-block;
            height: 5px;
            border-radius: 3px;
            background: var(--pico-primary);
            vertical-align: middle;
            margin-right: 0.4rem;
        }}
        .badge {{
            display: inline-block;
            padding: 0.1rem 0.45rem;
            border-radius: 4px;
            font-size: 0.7rem;
            font-weight: 600;
        }}
        .badge-moe {{ background: color-mix(in srgb, var(--pico-primary) 20%, transparent); color: var(--pico-primary); }}
        .badge-dense {{ background: color-mix(in srgb, #f78166 20%, transparent); color: #f78166; }}

        .theme-toggle {{
            position: absolute;
            top: 1.5rem;
            right: 1.5rem;
            background: none;
            border: 1px solid var(--pico-muted-border-color);
            border-radius: 50%;
            width: 2.2rem;
            height: 2.2rem;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.1rem;
            transition: border-color 0.2s;
            padding: 0;
            margin: 0;
        }}
        .theme-toggle:hover {{
            border-color: var(--pico-primary);
        }}

        section {{ margin-bottom: 3rem; }}
        section > p:first-of-type {{ opacity: 0.65; margin-bottom: 1.5rem; }}

        footer {{
            text-align: center;
            opacity: 0.55;
            font-size: 0.8rem;
            padding: 2rem 0;
        }}
    </style>
</head>
<body>
<main class="container" style="position:relative;">
    <button class="theme-toggle" onclick="toggleTheme()" aria-label="Toggle theme" id="theme-btn"></button>
    <header>
        <h1>mlx-coding-bench</h1>
        <p>Which local LLM runs best for coding on your Mac? The only benchmark measuring both <strong>speed</strong> and <strong>quality</strong> for agentic coding on Apple Silicon.</p>
    </header>

    <div class="stats-grid" id="hero-stats"></div>

    <div class="filters">
        <label>
            Hardware
            <select id="hw-filter" onchange="updateCharts()"></select>
        </label>
        <label>
            Quantization
            <select id="dtype-filter" onchange="updateCharts()">
                <option value="int4" selected>int4</option>
                <option value="int8">int8</option>
            </select>
        </label>
        <label>
            RAM limit
            <select id="ram-filter" onchange="updateCharts()">
                <option value="999">All</option>
                <option value="8">&le; 8 GB</option>
                <option value="16">&le; 16 GB</option>
                <option value="24">&le; 24 GB</option>
                <option value="32">&le; 32 GB</option>
                <option value="48">&le; 48 GB</option>
            </select>
        </label>
    </div>

    <section>
        <h2>Speed vs Quality</h2>
        <p>The tradeoff that matters. Top-right is the sweet spot — fast enough for interactive use AND smart enough for real coding tasks.</p>
        <div class="chart-wrap" id="scatter-chart" style="height:520px;"></div>
    </section>

    <section>
        <h2>Rankings</h2>
        <p>Sorted by weighted quality score (coding, reasoning, tool calling, math — harder problems count more).</p>
        <div class="table-wrap">
            <table id="rankings-table" role="grid">
                <thead>
                    <tr>
                        <th scope="col">Model</th>
                        <th scope="col">Type</th>
                        <th scope="col" style="text-align:right">RAM</th>
                        <th scope="col" style="text-align:right">Quality</th>
                        <th scope="col" style="text-align:right">tok/s</th>
                        <th scope="col" style="text-align:right">Coding</th>
                        <th scope="col" style="text-align:right">Tools</th>
                        <th scope="col" style="text-align:right">Reasoning</th>
                    </tr>
                </thead>
                <tbody id="rankings-body"></tbody>
            </table>
        </div>
    </section>

    <section>
        <h2>Speed Tiers</h2>
        <p>How fast is fast enough? Color-coded by real-world usability: <span style="color:#3fb950">green</span> = autocomplete-fast, <span style="color:#58a6ff">blue</span> = agentic coding, <span style="color:#d2a8ff">purple</span> = interactive chat, <span style="color:#f78166">orange</span> = slow.</p>
        <div class="chart-wrap" id="speed-tiers-chart" style="height:400px;"></div>
    </section>

    <footer>
        <p>Generated {generated} &middot; <a href="https://github.com/weklund/mlx-coding-bench">GitHub</a> &middot; MLX Metal &middot; int4 &middot; 81 problems &middot; 3 runs &middot; majority vote</p>
        <p><a href="https://github.com/weklund/mlx-coding-bench/blob/main/QUALITY_METHODOLOGY.md">Methodology</a></p>
    </footer>
</main>

<script>
const DATA = {data_json};

function isDark() {{
    return document.documentElement.getAttribute('data-theme') === 'dark';
}}

function themeColors() {{
    const dark = isDark();
    return {{
        text: dark ? '#c9d1d9' : '#24292f',
        grid: dark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.08)',
        line: dark ? 'rgba(255,255,255,0.12)' : 'rgba(0,0,0,0.12)',
        muted: dark ? 'rgba(255,255,255,0.35)' : 'rgba(0,0,0,0.4)',
        zone: dark ? 'rgba(63,185,80,0.03)' : 'rgba(63,185,80,0.06)',
    }};
}}

function toggleTheme() {{
    const current = document.documentElement.getAttribute('data-theme');
    const next = current === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('theme', next);
    updateThemeIcon();
    updateCharts();
}}

function updateThemeIcon() {{
    const btn = document.getElementById('theme-btn');
    btn.textContent = isDark() ? '☀️' : '🌙';
}}

function getFilters() {{
    const hw = document.getElementById('hw-filter').value;
    const dtype = document.getElementById('dtype-filter').value;
    const ramLimit = parseFloat(document.getElementById('ram-filter').value);
    return {{ hw, dtype, ramLimit }};
}}

function filteredModels() {{
    const {{ hw, dtype, ramLimit }} = getFilters();
    return DATA.models.filter(m =>
        m.hardware_key === hw && m.dtype === dtype && m.memory_gib <= ramLimit
    );
}}

function initFilters() {{
    const sel = document.getElementById('hw-filter');
    const preferred = ['Apple_M4_Pro_10P+4E+20GPU_64GB', 'Apple_M5_Max_XP+XE+40GPU_128GB'];
    DATA.hardware.forEach(h => {{
        const opt = document.createElement('option');
        opt.value = h.key;
        opt.textContent = h.display;
        sel.appendChild(opt);
    }});
    for (const p of preferred) {{
        if (DATA.hardware.some(h => h.key === p)) {{
            sel.value = p;
            break;
        }}
    }}
}}

function updateHeroStats() {{
    const models = filteredModels().filter(m => m.quality_pct !== null);
    const container = document.getElementById('hero-stats');
    if (models.length === 0) {{
        container.innerHTML = '<div class="stat-card"><div class="value">—</div><div class="label">No quality data</div></div>';
        return;
    }}
    const best = models.reduce((a, b) => a.quality_pct > b.quality_pct ? a : b);
    const fastest = models.reduce((a, b) => a.gen_tps > b.gen_tps ? a : b);
    const balanced = models.filter(m => m.gen_tps >= 50);
    const bestBal = balanced.length > 0
        ? balanced.reduce((a, b) => a.quality_pct > b.quality_pct ? a : b)
        : null;

    container.innerHTML = `
        <div class="stat-card">
            <div class="value">${{best.quality_pct}}%</div>
            <div class="label">Best Quality<br>${{best.display_name}}</div>
        </div>
        ${{bestBal ? `<div class="stat-card">
            <div class="value">${{bestBal.gen_tps}} <small style="font-size:0.6em">tok/s</small></div>
            <div class="label">Best Balance<br>${{bestBal.display_name}} (${{bestBal.quality_pct}}%)</div>
        </div>` : ''}}
        <div class="stat-card">
            <div class="value">${{fastest.gen_tps}} <small style="font-size:0.6em">tok/s</small></div>
            <div class="label">Fastest<br>${{fastest.display_name}} (${{fastest.quality_pct}}%)</div>
        </div>
        <div class="stat-card">
            <div class="value">${{models.length}}</div>
            <div class="label">Models Tested</div>
        </div>
    `;
}}

function updateScatter() {{
    const models = filteredModels();
    const withQuality = models.filter(m => m.quality_pct !== null);

    const isMoE = m => m.arch.includes('MoE');
    const traces = [];
    const tc = themeColors();

    const moQ = withQuality.filter(isMoE);
    if (moQ.length) {{
        traces.push({{
            x: moQ.map(m => m.gen_tps),
            y: moQ.map(m => m.quality_pct),
            text: moQ.map(m => m.display_name),
            customdata: moQ.map(m => [m.arch, m.memory_gib]),
            mode: 'markers+text',
            textposition: 'top center',
            textfont: {{ size: 10, color: '#58a6ff' }},
            marker: {{ size: 11, color: '#58a6ff', symbol: 'diamond', line: {{ width: 0 }} }},
            name: 'MoE',
            hovertemplate: '<b>%{{text}}</b><br>%{{customdata[0]}}<br>%{{x:.0f}} tok/s · %{{y:.1f}}%<br>%{{customdata[1]}} GiB<extra></extra>',
        }});
    }}

    const deQ = withQuality.filter(m => !isMoE(m));
    if (deQ.length) {{
        traces.push({{
            x: deQ.map(m => m.gen_tps),
            y: deQ.map(m => m.quality_pct),
            text: deQ.map(m => m.display_name),
            customdata: deQ.map(m => [m.arch, m.memory_gib]),
            mode: 'markers+text',
            textposition: 'top center',
            textfont: {{ size: 10, color: '#f78166' }},
            marker: {{ size: 11, color: '#f78166', symbol: 'circle', line: {{ width: 0 }} }},
            name: 'Dense',
            hovertemplate: '<b>%{{text}}</b><br>%{{customdata[0]}}<br>%{{x:.0f}} tok/s · %{{y:.1f}}%<br>%{{customdata[1]}} GiB<extra></extra>',
        }});
    }}

    const layout = {{
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: {{ color: tc.text, size: 12 }},
        xaxis: {{
            title: {{ text: 'Generation Speed (tok/s)', standoff: 10 }},
            gridcolor: tc.grid,
            zeroline: false,
            linecolor: tc.line,
        }},
        yaxis: {{
            title: {{ text: 'Quality (%)', standoff: 10 }},
            gridcolor: tc.grid,
            zeroline: false,
            range: [10, 100],
            linecolor: tc.line,
        }},
        legend: {{ x: 0.01, y: 0.99, bgcolor: 'rgba(0,0,0,0)', font: {{ size: 11 }} }},
        margin: {{ t: 20, r: 20, b: 50, l: 55 }},
        shapes: [
            {{ type: 'line', x0: 50, x1: 50, y0: 10, y1: 100, line: {{ color: tc.line, dash: 'dot', width: 1 }} }},
            {{ type: 'line', x0: 0, x1: 500, y0: 60, y1: 60, line: {{ color: tc.line, dash: 'dot', width: 1 }} }},
            {{ type: 'rect', x0: 50, x1: 500, y0: 60, y1: 100, fillcolor: tc.zone, line: {{ width: 0 }} }},
        ],
        annotations: [
            {{ x: 52, y: 13, text: '50 tok/s', showarrow: false, font: {{ size: 9, color: tc.muted }}, xanchor: 'left' }},
            {{ x: 3, y: 62, text: '60% quality', showarrow: false, font: {{ size: 9, color: tc.muted }}, xanchor: 'left' }},
        ],
    }};

    Plotly.newPlot('scatter-chart', traces, layout, {{ responsive: true, displayModeBar: false }});
}}

function updateTable() {{
    const models = filteredModels().filter(m => m.quality_pct !== null);
    models.sort((a, b) => b.quality_pct - a.quality_pct);

    const tbody = document.getElementById('rankings-body');
    const topQuality = models.length > 0 ? models[0].quality_pct : 0;

    tbody.innerHTML = models.map((m, i) => {{
        const barWidth = Math.round((m.quality_pct / topQuality) * 60);
        const badge = m.arch.includes('MoE')
            ? '<span class="badge badge-moe">MoE</span>'
            : '<span class="badge badge-dense">Dense</span>';
        const coding = m.category_scores.coding != null ? m.category_scores.coding + '%' : '—';
        const toolCalling = m.category_scores.tool_calling != null ? m.category_scores.tool_calling + '%' : '—';
        const reasoning = m.category_scores.reasoning != null ? m.category_scores.reasoning + '%' : '—';
        return `<tr>
            <td><strong>${{m.display_name}}</strong></td>
            <td>${{badge}}</td>
            <td class="r">${{m.memory_gib}} GiB</td>
            <td class="r"><span class="quality-bar" style="width:${{barWidth}}px"></span>${{m.quality_pct}}%</td>
            <td class="r">${{m.gen_tps}}</td>
            <td class="r">${{coding}}</td>
            <td class="r">${{toolCalling}}</td>
            <td class="r">${{reasoning}}</td>
        </tr>`;
    }}).join('');
}}

function updateSpeedTiers() {{
    const models = filteredModels().filter(m => m.quality_pct !== null);
    models.sort((a, b) => b.gen_tps - a.gen_tps);

    const colors = models.map(m => {{
        if (m.gen_tps >= 100) return '#3fb950';
        if (m.gen_tps >= 50) return '#58a6ff';
        if (m.gen_tps >= 30) return '#d2a8ff';
        return '#f78166';
    }});

    const tc = themeColors();
    const trace = {{
        type: 'bar',
        orientation: 'h',
        y: models.map(m => m.display_name),
        x: models.map(m => m.gen_tps),
        marker: {{ color: colors }},
        text: models.map(m => `${{m.gen_tps}} tok/s`),
        textposition: 'outside',
        textfont: {{ size: 11, color: tc.text }},
        hovertemplate: models.map(m =>
            `<b>${{m.display_name}}</b><br>${{m.gen_tps}} tok/s<br>${{m.quality_pct}}% quality<br>${{m.memory_gib}} GiB<extra></extra>`
        ),
        showlegend: false,
    }};

    const layout = {{
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: {{ color: tc.text, size: 11 }},
        xaxis: {{
            title: 'Generation Speed (tok/s)',
            gridcolor: tc.grid,
            zeroline: false,
        }},
        yaxis: {{ autorange: 'reversed' }},
        margin: {{ t: 8, r: 70, b: 40, l: 160 }},
        height: Math.max(350, models.length * 26 + 60),
        bargap: 0.25,
    }};

    Plotly.newPlot('speed-tiers-chart', [trace], layout, {{ responsive: true, displayModeBar: false }});
}}

function updateCharts() {{
    updateHeroStats();
    updateScatter();
    updateTable();
    updateSpeedTiers();
}}

initFilters();
updateThemeIcon();
updateCharts();

// Sticky filter shadow
const filters = document.querySelector('.filters');
const observer = new IntersectionObserver(
    ([e]) => filters.classList.toggle('stuck', e.intersectionRatio < 1),
    {{ threshold: [1], rootMargin: '-1px 0px 0px 0px' }}
);
observer.observe(filters);

// Listen for system theme changes
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', e => {{
    if (!localStorage.getItem('theme')) {{
        document.documentElement.setAttribute('data-theme', e.matches ? 'dark' : 'light');
        updateThemeIcon();
        updateCharts();
    }}
}});
</script>
</body>
</html>"""


def main(output_dir: str = "site"):
    """Generate the GitHub Pages site."""
    output_path = Path(output_dir)
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    output_path.mkdir(parents=True, exist_ok=True)

    print("Loading benchmark data...")
    data = _build_site_data()
    print(f"  {len(data['models'])} model entries across {len(data['hardware'])} hardware profiles")

    models_with_quality = [m for m in data["models"] if m["quality_pct"] is not None]
    print(f"  {len(models_with_quality)} with quality scores")

    print("Rendering site...")
    html = _render_html(data)

    index_path = output_path / "index.html"
    index_path.write_text(html)

    (output_path / ".nojekyll").touch()

    print(f"  Written to {index_path}")
    print(f"  Open: file://{index_path}")


if __name__ == "__main__":
    fire.Fire(main)
