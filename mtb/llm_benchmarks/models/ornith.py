from typing import Any

from mtb.llm_benchmarks.models.base import ModelSpec

__all__ = [
    "Ornith_1p0_9B",
]


def format_ornith_prompt(prompt: str) -> Any:
    """Ornith-1.0 is a Qwen3.5-based agentic coding model (qwen3_5 arch).

    It uses the standard system/user message format and reasons in
    `<think>...</think>` blocks by default.
    """
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]
    return messages


# Dense 9B (qwen3_5 architecture), reasoning-first agentic coding model.
Ornith_1p0_9B = ModelSpec(
    name="ornith-1.0-9b",
    num_params=int(9e9),
    prompt_formatter=format_ornith_prompt,
    thinking=True,
    model_ids={
        "mlx": {
            "int4": "mlx-community/Ornith-1.0-9B-4bit",
            "int8": "mlx-community/Ornith-1.0-9B-8bit",
            "bfloat16": "mlx-community/Ornith-1.0-9B-bf16",
        },
    },
)
