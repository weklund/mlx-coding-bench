from typing import Any

from mtb.llm_benchmarks.models.base import ModelSpec

__all__ = [
    "Qwen3p6_35B_A3B",
]


def format_qwen36_prompt(prompt: str) -> Any:
    """Qwen3.6 models use the same prompt format as Qwen3/Qwen3.5."""
    messages = [
        {
            "role": "system",
            "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]
    return messages


# NOTE: mlx-community/Qwen3.6-35B-A3B-* are VLM-wrapper checkpoints; mlx-lm loads
# them text-only via the qwen3_5_moe sanitize() (vision weights are stripped).
Qwen3p6_35B_A3B = ModelSpec(
    name="qwen-3.6-35b-a3b",
    num_params=int(35e9),
    prompt_formatter=format_qwen36_prompt,
    thinking=True,
    model_ids={
        "mlx": {
            "int4": "mlx-community/Qwen3.6-35B-A3B-4bit",
            "int8": "mlx-community/Qwen3.6-35B-A3B-8bit",
        },
    },
)
