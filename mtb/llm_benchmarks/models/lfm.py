from typing import Any

from mtb.llm_benchmarks.models.base import ModelSpec

__all__ = [
    "LFM2_24B_A2B",
    "LFM2p5_8B_A1B",
]


def format_lfm_prompt(prompt: str) -> Any:
    """LFM2 models use standard system/user prompt format."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    return messages


LFM2_24B_A2B = ModelSpec(
    name="lfm2-24b-a2b",
    num_params=2e9,
    prompt_formatter=format_lfm_prompt,
    model_ids={
        "mlx": {
            "int4": "LiquidAI/LFM2-24B-A2B-MLX-4bit",
            "int8": "LiquidAI/LFM2-24B-A2B-MLX-8bit",
        },
    },
)


# 8.3B total / 1.5B active MoE (lfm2_moe). Emits <think> reasoning by default
# (enable_thinking=False does NOT suppress it), so run as a thinking model and
# benchmark quality with a raised --max_tokens_multiplier (like Qwen 3.6).
LFM2p5_8B_A1B = ModelSpec(
    name="lfm2.5-8b-a1b",
    num_params=1.5e9,
    prompt_formatter=format_lfm_prompt,
    thinking=True,
    model_ids={
        "mlx": {
            "int4": "LiquidAI/LFM2.5-8B-A1B-MLX-4bit",
            "int8": "LiquidAI/LFM2.5-8B-A1B-MLX-8bit",
        },
    },
)
