# mlx-coding-bench

[![tests-Mac](https://github.com/weklund/mlx-coding-bench/actions/workflows/tests-mac.yaml/badge.svg)](https://github.com/weklund/mlx-coding-bench/actions/workflows/tests-mac.yaml)

Which local LLM runs best for coding on your Mac? Speed **and** quality benchmarks for MLX models on Apple Silicon — the only benchmark that measures both.

**[Live Dashboard](https://weklund.github.io/mlx-coding-bench)** · 35+ models · 81 problems · Real hardware

<!-- BEGIN BENCHMARK TABLE -->

> MLX Metal | int4 quantization | July 2026
> Speed: 1024 prompt tokens, 100 generated tokens
> Quality: 81 problems across coding, reasoning, tool calling, math, writing (3 runs each, majority vote)
> **API baseline:** Claude Opus 4.6 scores 86.7% on the same quality benchmark (via Anthropic API, not local)
> † Run with a raised per-problem token-budget multiplier (`--max_tokens_multiplier`) so its verbose thinking can finish within budget; the cap is a ceiling, not forced length. Other rows use the default 1×, so this score is not strictly budget-comparable. Each run's multiplier is recorded in its `settings.json`.

### Best Models by Hardware

| Hardware | Best Quality | Best Balance | Best Speed |
|---|---|---|---|
| **M4 Pro 64GB** | Qwen 3.6-27B (16 tok/s, 90.1%) | Qwen 3.6-35B-A3B (86 tok/s, 86.2%) | LFM2.5-8B-A1B (165 tok/s, 60.1%) |
| **M5 Max 128GB** | Qwen 3.6-27B (32 tok/s, 86.2%) | Qwen 3.6-35B-A3B (133 tok/s, 83.3%) | Gemma 4 E2B-it (205 tok/s, 67.8%) |

### M4 Pro 64GB

| Model | RAM | Quality | Gen tok/s |
|---|---:|---:|---:|
| Qwen 3.6-27B (27B dense) | 17.0 GiB | **90.1%** † | 16 |
| Qwen 3.6-35B-A3B (3B MoE) | 20.7 GiB | 86.2% † | 86 |
| ornith-1.0-9b | 6.3 GiB | 77.3% † | 50 |
| Gemma 4 31B-it (31B dense) | 18.9 GiB | 72.2% | 13 |
| Qwen 3.5-27B (27B dense) | 18.8 GiB | 68.6% | 12 |
| LFM2-24B-A2B (2B MoE) | 14.2 GiB | 67.3% | 117 |
| gpt-oss-20b | 11.8 GiB | 67.0% † | 88 |
| Qwen 3.5-35B-A3B (3B MoE) | 21.9 GiB | 66.9% | 26 |
| phi-4-mini | 3.2 GiB | 66.5% | 94 |
| phi-4 | 9.2 GiB | 66.0% | 28 |
| Qwen3-Coder-30B-A3B (3B MoE) | 17.8 GiB | 65.7% | 80 |
| Gemma 4 E2B-it (2.3B dense) | 3.5 GiB | 65.3% | 121 |
| Gemma 4 26B-A4B-it (3.8B MoE) | 15.3 GiB | 64.1% | 65 |
| Qwen 3.5-9B (9B dense) | 7.3 GiB | 61.6% | 39 |
| LFM2.5-8B-A1B (1.5B MoE) | 5.7 GiB | 60.1% † | 165 |
| Gemma 4 12B-it (12B dense) | 7.8 GiB | 59.1% † | 31 |
| Gemma 4 E4B-it (4.5B dense) | 5.0 GiB | 50.6% | 69 |
| Qwen 3.5-4B (4B dense) | 4.9 GiB | 50.6% | 67 |
| Qwen 2.5-Coder-3B (3B dense) | 2.6 GiB | 46.9% | 117 |
| Gemma 3-4B-it QAT (4B dense) | 3.5 GiB | 45.7% | 88 |
| GLM-4.7-Flash (3B MoE) | 17.6 GiB | 45.3% | 62 |
| Gemma 3-4B-it (4B dense) | 3.5 GiB | 44.5% | 88 |
| Nemotron-Nano-9B-v2 (9B dense) | 8.1 GiB | 42.9% | 47 |
| Qwen 3.5-2B (2B dense) | 3.3 GiB | 41.2% | 142 |
| Qwen 2.5-3B-it (3B dense) | 2.6 GiB | 37.6% | 111 |
| Qwen 3.5-0.8B (0.8B dense) | 2.5 GiB | 28.6% | 275 |
| DeepSeek-R1-0528-Qwen3-8B (8B dense) | 5.4 GiB | 17.1% | 51 |

<details>
<summary>Speed-only models (no quality scores yet)</summary>

| Model | RAM | Gen tok/s | Prefill tok/s |
|---|---:|---:|---:|
| Qwen 2.5-Coder-0.5B (0.5B dense) | 1.1 GiB | 420 | 7054 |
| Qwen 2.5-0.5B-it (0.5B dense) | 1.1 GiB | 380 | 7182 |
| Qwen 3-0.6B-it (0.6B dense) | 1.3 GiB | 334 | 5329 |
| Gemma 3-1B-it (1B dense) | 1.6 GiB | 252 | 3679 |
| Gemma 3-1B-it QAT (1B dense) | 1.6 GiB | 250 | 3774 |
| Nemotron-3-Nano-4B (4B dense) | 4.5 GiB | 102 | 833 |
| DeepSeek-R1-Distill-7B (7B dense) | 5.1 GiB | 56 | 531 |
| Qwen 3-8B-it (8B dense) | 5.4 GiB | 52 | 457 |
| Gemma 3-12B-it QAT (12B dense) | 8.3 GiB | 32 | 271 |
| Qwen 3-14B-it (14B dense) | 9.1 GiB | 29 | 239 |
| phi-4-reasoning-plus | 10.1 GiB | 26 | 239 |
| Qwen 3.5-27B Opus Distilled (27B dense) | 16.9 GiB | 16 | 128 |

</details>

<details>
<summary>int8 speed results</summary>

| Model | RAM | Gen tok/s | Prefill tok/s |
|---|---:|---:|---:|
| Qwen 2.5-Coder-0.5B (0.5B dense) | 1.3 GiB | 272 | 7032 |
| Qwen 2.5-0.5B-it (0.5B dense) | 1.3 GiB | 272 | 6790 |
| Qwen 3-0.6B-it (0.6B dense) | 1.6 GiB | 238 | 5183 |
| Qwen 3.5-0.8B (0.8B dense) | 2.9 GiB | 195 | 3494 |
| Gemma 3-1B-it QAT (1B dense) | 2.2 GiB | 171 | 3722 |
| Gemma 3-1B-it (1B dense) | 2.2 GiB | 169 | 3560 |
| Qwen 3.5-2B (2B dense) | 4.0 GiB | 93 | 1668 |
| Gemma 4 E2B-it (2.3B dense) | 5.8 GiB | 78 | 4265 |
| LFM2-24B-A2B (2B MoE) | 25.9 GiB | 75 | 1186 |
| gpt-oss-20b | 12.7 GiB | 72 | 837 |
| Qwen 2.5-Coder-3B (3B dense) | 4.0 GiB | 72 | 1093 |
| Qwen 2.5-3B-it (3B dense) | 4.0 GiB | 68 | 1135 |
| Nemotron-3-Nano-4B (4B dense) | 6.5 GiB | 58 | 824 |
| Qwen3-Coder-30B-A3B (3B MoE) | 33.1 GiB | 54 | 865 |
| Gemma 3-4B-it QAT (4B dense) | 5.6 GiB | 52 | 911 |
| Gemma 3-4B-it (4B dense) | 5.6 GiB | 52 | 887 |
| Gemma 4 26B-A4B-it (3.8B MoE) | 27.7 GiB | 45 | 785 |
| Qwen 3.5-4B (4B dense) | 6.8 GiB | 43 | 684 |
| GLM-4.7-Flash (3B MoE) | 32.5 GiB | 43 | 674 |
| Gemma 4 E4B-it (4.5B dense) | 8.7 GiB | 42 | 1229 |
| Qwen 3-8B-it (8B dense) | 9.5 GiB | 30 | 454 |
| DeepSeek-R1-0528-Qwen3-8B (8B dense) | 9.5 GiB | 29 | 450 |
| ornith-1.0-9b | 10.6 GiB | 28 | 431 |
| Qwen 3.5-9B (9B dense) | 11.7 GiB | 24 | 383 |
| Qwen 3.5-35B-A3B (3B MoE) | 39.2 GiB | 22 | 716 |
| Gemma 3-12B-it QAT (12B dense) | 14.6 GiB | 18 | 280 |
| Qwen 3-14B-it (14B dense) | 16.4 GiB | 16 | 237 |
| phi-4-reasoning-plus | 16.5 GiB | 16 | 243 |
| phi-4 | 16.5 GiB | 16 | 228 |
| Qwen 3.5-27B (27B dense) | 32.0 GiB | 7 | 108 |
| Gemma 4 31B-it (31B dense) | 34.1 GiB | 7 | 99 |

</details>

### M5 Max 128GB

| Model | RAM | Quality | Gen tok/s |
|---|---:|---:|---:|
| Qwen 3.6-27B (27B dense) | 17.0 GiB | **86.2%** † | 32 |
| Qwen 3.6-35B-A3B (3B MoE) | 20.7 GiB | 83.3% † | 133 |
| Qwen3-Coder-30B-A3B (3B MoE) | 17.8 GiB | 75.5% | 129 |
| Gemma 4 31B-it (31B dense) | 18.9 GiB | 72.7% | 17 |
| Qwen 3.5-27B (27B dense) | 18.8 GiB | 71.0% | 25 |
| LFM2-24B-A2B (2B MoE) | 14.2 GiB | 70.6% | 180 |
| Gemma 4 E2B-it (2.3B dense) | 3.5 GiB | 67.8% | 205 |
| Qwen 3.5-35B-A3B (3B MoE) | 21.9 GiB | 67.8% | 44 |
| Gemma 4 26B-A4B-it (3.8B MoE) | 15.3 GiB | 65.3% | 110 |
| Qwen 3.5-9B (9B dense) | 7.3 GiB | 64.5% | 79 |
| Gemma 4 12B-it (12B dense) | 7.8 GiB | 56.2% † | 19 |
| Qwen 3.5-4B (4B dense) | 4.9 GiB | 50.6% | 131 |
| Gemma 4 E4B-it (4.5B dense) | 5.0 GiB | 50.6% | 130 |
| Qwen 2.5-Coder-3B (3B dense) | 2.6 GiB | 48.2% | 226 |
| GLM-4.7-Flash (3B MoE) | 17.6 GiB | 48.2% | 96 |
| Gemma 3-4B-it (4B dense) | 3.5 GiB | 46.9% | 171 |
| Nemotron-Nano-9B-v2 (9B dense) | 8.1 GiB | 46.5% | 67 |
| Gemma 3-4B-it QAT (4B dense) | 3.5 GiB | 45.3% | 172 |
| Qwen 2.5-3B-it (3B dense) | 2.6 GiB | 38.4% | 209 |
| Qwen 3.5-2B (2B dense) | 3.3 GiB | 37.6% | 235 |
| Qwen 3.5-0.8B (0.8B dense) | 2.5 GiB | 24.9% | 409 |
| DeepSeek-R1-0528-Qwen3-8B (8B dense) | 5.4 GiB | 16.3% | 104 |

<details>
<summary>Speed-only models (no quality scores yet)</summary>

| Model | RAM | Gen tok/s | Prefill tok/s |
|---|---:|---:|---:|
| Qwen 2.5-Coder-0.5B (0.5B dense) | 1.1 GiB | 611 | 20459 |
| Qwen 2.5-0.5B-it (0.5B dense) | 1.1 GiB | 542 | 19425 |
| Qwen 3-0.6B-it (0.6B dense) | 1.3 GiB | 527 | 17023 |
| Gemma 3-1B-it QAT (1B dense) | 1.6 GiB | 373 | 14613 |
| Gemma 3-1B-it (1B dense) | 1.6 GiB | 371 | 15039 |
| DeepSeek-R1-Distill-7B (7B dense) | 5.1 GiB | 114 | 3116 |
| Qwen 3-8B-it (8B dense) | 5.4 GiB | 103 | 2890 |
| Qwen 3-14B-it (14B dense) | 9.1 GiB | 60 | 1398 |
| Gemma 3-12B-it QAT (12B dense) | 8.2 GiB | 48 | 1275 |
| Qwen 3.5-27B Opus Distilled (27B dense) | 16.9 GiB | 28 | 436 |

</details>

<details>
<summary>int8 speed results</summary>

| Model | RAM | Gen tok/s | Prefill tok/s |
|---|---:|---:|---:|
| Qwen 3-0.6B-it (0.6B dense) | 1.6 GiB | 422 | 16697 |
| Qwen 2.5-Coder-0.5B (0.5B dense) | 1.3 GiB | 416 | 20203 |
| Qwen 2.5-0.5B-it (0.5B dense) | 1.3 GiB | 411 | 18370 |
| Qwen 3.5-0.8B (0.8B dense) | 2.9 GiB | 291 | 9616 |
| Gemma 3-1B-it (1B dense) | 2.2 GiB | 277 | 14773 |
| Gemma 3-1B-it QAT (1B dense) | 2.2 GiB | 276 | 14428 |
| Qwen 3.5-2B (2B dense) | 4.0 GiB | 151 | 5177 |
| Gemma 4 E2B-it (2.3B dense) | 5.8 GiB | 147 | 14893 |
| Qwen 2.5-Coder-3B (3B dense) | 4.0 GiB | 144 | 6504 |
| Qwen 2.5-3B-it (3B dense) | 4.0 GiB | 134 | 6342 |
| LFM2-24B-A2B (2B MoE) | 25.9 GiB | 132 | 5297 |
| Gemma 3-4B-it QAT (4B dense) | 5.6 GiB | 104 | 5157 |
| Gemma 3-4B-it (4B dense) | 5.6 GiB | 104 | 5087 |
| Qwen 3.6-35B-A3B (3B MoE) | 38.0 GiB | 102 | 3488 |
| Qwen3-Coder-30B-A3B (3B MoE) | 33.1 GiB | 96 | 3353 |
| Gemma 4 26B-A4B-it (3.8B MoE) | 27.7 GiB | 85 | 3208 |
| Qwen 3.5-4B (4B dense) | 6.8 GiB | 85 | 2584 |
| Gemma 4 E4B-it (4.5B dense) | 8.7 GiB | 85 | 6374 |
| GLM-4.7-Flash (3B MoE) | 32.5 GiB | 72 | 2565 |
| Qwen 3-8B-it (8B dense) | 9.5 GiB | 63 | 2818 |
| DeepSeek-R1-0528-Qwen3-8B (8B dense) | 9.5 GiB | 62 | 2719 |
| Qwen 3.5-9B (9B dense) | 11.7 GiB | 49 | 1511 |
| Qwen 3.5-35B-A3B (3B MoE) | 39.2 GiB | 45 | 2151 |
| Qwen 3-14B-it (14B dense) | 16.4 GiB | 34 | 1426 |
| Gemma 3-12B-it QAT (12B dense) | 14.6 GiB | 32 | 1060 |
| Qwen 3.6-27B (27B dense) | 30.4 GiB | 18 | 669 |
| Qwen 3.5-27B Opus Distilled (27B dense) | 30.3 GiB | 17 | 434 |
| Qwen 3.5-27B (27B dense) | 32.0 GiB | 15 | 480 |
| Gemma 4 31B-it (31B dense) | 34.1 GiB | 9 | 544 |

</details>

<details>
<summary><h3>M4 Pro 24GB (legacy)</h3></summary>

| Model | RAM | Gen tok/s | Prefill tok/s |
|---|---:|---:|---:|
| Qwen 2.5-0.5B-it (0.5B dense) | 1.4 GiB | 376 | 5530 |
| Qwen 2.5-Coder-0.5B (0.5B dense) | 1.4 GiB | 371 | 5464 |
| Qwen 3-0.6B-it (0.6B dense) | 1.6 GiB | 328 | 4277 |
| Gemma 3-1B-it QAT (1B dense) | 2.0 GiB | 222 | 2858 |
| Gemma 3-1B-it (1B dense) | 2.0 GiB | 222 | 2866 |
| Qwen 2.5-Coder-3B (3B dense) | 2.9 GiB | 118 | 1050 |
| Qwen 2.5-3B-it (3B dense) | 2.9 GiB | 112 | 1051 |
| Gemma 3-4B-it QAT (4B dense) | 3.9 GiB | 84 | 756 |
| Gemma 3-4B-it (4B dense) | 3.9 GiB | 81 | 739 |
| DeepSeek-R1-Distill-7B (7B dense) | 5.3 GiB | 57 | 476 |
| Qwen 3-8B-it (8B dense) | 5.6 GiB | 52 | 414 |
| DeepSeek-R1-0528-Qwen3-8B (8B dense) | 5.6 GiB | 52 | 416 |
| Gemma 3-12B-it QAT (12B dense) | 8.6 GiB | 31 | 256 |
| Qwen 3-14B-it (14B dense) | 9.3 GiB | 29 | 224 |

<details>
<summary>int8 speed results</summary>

| Model | RAM | Gen tok/s | Prefill tok/s |
|---|---:|---:|---:|
| Qwen 2.5-0.5B-it (0.5B dense) | 1.6 GiB | 273 | 5745 |
| Qwen 2.5-Coder-0.5B (0.5B dense) | 1.6 GiB | 271 | 5717 |
| Qwen 3-0.6B-it (0.6B dense) | 1.8 GiB | 239 | 4323 |
| Gemma 3-1B-it QAT (1B dense) | 2.7 GiB | 156 | 2861 |
| Gemma 3-1B-it (1B dense) | 2.7 GiB | 155 | 2850 |
| Qwen 2.5-Coder-3B (3B dense) | 4.2 GiB | 72 | 1052 |
| Qwen 2.5-3B-it (3B dense) | 4.2 GiB | 67 | 1046 |
| Gemma 3-4B-it QAT (4B dense) | 6.1 GiB | 51 | 808 |
| Gemma 3-4B-it (4B dense) | 6.1 GiB | 50 | 744 |
| Qwen 3-8B-it (8B dense) | 9.7 GiB | 30 | 413 |
| DeepSeek-R1-0528-Qwen3-8B (8B dense) | 9.7 GiB | 30 | 413 |
| Gemma 3-12B-it QAT (12B dense) | 15.0 GiB | 18 | 254 |

</details>

</details>

## Optimizations

Techniques layered on top of the standard config above. They trade extra setup for speed/memory gains and are opt-in — not every model or architecture supports each one — so they're reported separately rather than mixed into the main ranking. Speedup is vs. that model's standard row on the same hardware **and prompt profile**: `generic` uses the standard benchmark prompt — general text, near worst-case acceptance for speculative decoding — while `code` uses a code-continuation prompt, closer to agentic-coding use.

### ⚡ Speculative decoding

A small **drafter** model proposes tokens that the base model verifies in parallel. Output is identical to the standard config — only throughput changes. Requires a drafter sharing the base model's tokenizer, and a base architecture with a trimmable KV cache (Qwen/Llama/Gemma work; some newer architectures don't yet).

| Model | Drafter | Prompt | Hardware | Standard | + Optimized | Speedup |
|---|---|---|---|---:|---:|---:|
| Gemma 3-12B-it QAT (12B dense) | gemma-3-1b-it-qat | code | M4 Pro 64GB | 32 tok/s | 45 tok/s | 1.43× |
| Gemma 3-12B-it QAT (12B dense) | gemma-3-1b-it-qat | generic | M4 Pro 64GB | 32 tok/s | 28 tok/s | 0.87× |
| Qwen 3-14B-it (14B dense) | Qwen3-0.6B | code | M4 Pro 64GB | 29 tok/s | 39 tok/s | 1.36× |
| Qwen 3-14B-it (14B dense) | Qwen3-0.6B | generic | M4 Pro 64GB | 29 tok/s | 46 tok/s | 1.60× |
| Qwen 3-8B-it (8B dense) | Qwen3-0.6B | code | M4 Pro 64GB | 51 tok/s | 55 tok/s | 1.08× |
| Qwen 3-8B-it (8B dense) | Qwen3-0.6B | generic | M4 Pro 64GB | 52 tok/s | 52 tok/s | 1.00× |
| Qwen3-Coder-30B-A3B (3B MoE) | Qwen3-0.6B | code | M4 Pro 64GB | 80 tok/s | 80 tok/s | 1.00× |
| Qwen3-Coder-30B-A3B (3B MoE) | Qwen3-0.6B | generic | M4 Pro 64GB | 80 tok/s | 53 tok/s | 0.66× |



<!-- END BENCHMARK TABLE -->

## Reading the Table

Models are sorted by **Quality** (best first) — a weighted score across 81 agentic coding problems. Harder problems count more: Easy (1x), Hard (2x), Expert (3x), Tool Calling (3x). Each problem runs 3 times with majority vote. Models without quality scores are in a collapsible "Speed-only" section below.

**RAM** is the int4 memory footprint — check this first to see what fits on your hardware.

**Gen tok/s** is generation speed at int4 with a 1024-token prompt. As a rule of thumb: 100+ is great for autocomplete, 50+ is comfortable for agentic coding, 30+ is usable for interactive chat, and below 15 feels slow.

For full scoring methodology, see [QUALITY_METHODOLOGY.md](QUALITY_METHODOLOGY.md).


## Quick Start

```bash
git clone git@github.com:weklund/mlx-coding-bench.git
cd mlx-coding-bench
make setup

# Run speed benchmarks
uv run python scripts/run_llm_benchmarks.py \
    --run_only_benchmarks '["gemma-4-e2b-it"]' \
    --dtypes '["int4"]' --num_iterations 3

# Run quality benchmarks
uv run python scripts/run_quality_benchmarks.py \
    --difficulty all \
    --run_only_benchmarks '["gemma-4-e2b-it"]' \
    --dtypes '["int4"]' --num_runs 3

# Update the README table
uv run python scripts/update_readme_table.py

# Generate the interactive site
uv run python scripts/generate_site.py
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for full setup instructions and how to submit results.


## Related

- **[Live Dashboard](https://weklund.github.io/mlx-coding-bench)** -- interactive speed vs quality charts, filterable by hardware and RAM.
- **[mlx-stack](https://github.com/weklund/mlx-stack)** -- runs multiple MLX models behind a single OpenAI-compatible endpoint. This benchmark suite feeds its model catalog.
- **[QUALITY_METHODOLOGY.md](QUALITY_METHODOLOGY.md)** -- detailed breakdown of the quality benchmark suite, scoring formula, and problem categories.
- **[CONTRIBUTING.md](CONTRIBUTING.md)** -- how to add models, run benchmarks, and submit measurements.
