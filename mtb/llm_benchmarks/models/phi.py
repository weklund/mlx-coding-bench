from typing import Any

from mtb.llm_benchmarks.models.base import ModelSpec

__all__ = [
    "Phi4",
    "Phi4_Reasoning_Plus",
    "Phi4_Mini",
]


def format_phi_prompt(prompt: str) -> Any:
    """Phi-4 models use im_start/im_sep/im_end chat format."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    return messages


Phi4 = ModelSpec(
    name="phi-4",
    num_params=int(14e9),
    prompt_formatter=format_phi_prompt,
    model_ids={
        "mlx": {
            "int4": "mlx-community/phi-4-4bit",
            "int8": "mlx-community/phi-4-8bit",
            "bfloat16": "mlx-community/phi-4-bf16",
        },
    },
)

Phi4_Reasoning_Plus = ModelSpec(
    name="phi-4-reasoning-plus",
    num_params=int(14e9),
    prompt_formatter=format_phi_prompt,
    thinking=True,
    model_ids={
        "mlx": {
            "int4": "mlx-community/Phi-4-reasoning-plus-4bit",
            "int8": "mlx-community/Phi-4-reasoning-plus-8bit",
        },
    },
)

Phi4_Mini = ModelSpec(
    name="phi-4-mini",
    num_params=int(3_800_000_000),
    prompt_formatter=format_phi_prompt,
    model_ids={
        "mlx": {
            "int4": "mlx-community/Phi-4-mini-instruct-4bit",
        },
    },
)
