from typing import Any

from mtb.llm_benchmarks.models.base import ModelSpec

__all__ = [
    # Fast / on-device tier
    "Gemma4_E2B_it",
    "Gemma4_E4B_it",
    # Reasoning / agentic tier
    "Gemma4_12B_it",
    "Gemma4_26B_A4B_it",
    "Gemma4_31B_it",
]


def format_gemma4_prompt(prompt: str) -> Any:
    """Gemma 4 models expect a system prompt with data types."""
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        },
    ]
    return messages


# --- Fast / on-device tier ---

Gemma4_E2B_it = ModelSpec(
    name="gemma-4-e2b-it",
    num_params=2.3e9,
    prompt_formatter=format_gemma4_prompt,
    model_ids={
        "mlx": {
            "int4": "mlx-community/gemma-4-e2b-it-4bit",
            "int8": "mlx-community/gemma-4-e2b-it-8bit",
        },
    },
)

Gemma4_E4B_it = ModelSpec(
    name="gemma-4-e4b-it",
    num_params=4.5e9,
    prompt_formatter=format_gemma4_prompt,
    model_ids={
        "mlx": {
            "int4": "mlx-community/gemma-4-e4b-it-4bit",
            "int8": "mlx-community/gemma-4-e4b-it-8bit",
        },
    },
)


# --- Reasoning / agentic tier ---

# Dense ~12B. model_type=gemma4_unified (encoder-free multimodal); mlx-lm loads
# it text-only via the gemma4 remap (PR #1349) — requires mlx-lm >= the commit
# with that fix (this repo pins ml-explore/mlx-lm main).
Gemma4_12B_it = ModelSpec(
    name="gemma-4-12b-it",
    num_params=12e9,
    prompt_formatter=format_gemma4_prompt,
    thinking=True,
    model_ids={
        "mlx": {
            "int4": "mlx-community/gemma-4-12B-it-4bit",
        },
    },
)

Gemma4_26B_A4B_it = ModelSpec(
    name="gemma-4-26b-a4b-it",
    num_params=3.8e9,
    prompt_formatter=format_gemma4_prompt,
    thinking=True,
    model_ids={
        "mlx": {
            "int4": "mlx-community/gemma-4-26b-a4b-it-4bit",
            "int8": "mlx-community/gemma-4-26b-a4b-it-8bit",
            "bfloat16": "mlx-community/gemma-4-26b-a4b-it-bf16",
        },
    },
)

Gemma4_31B_it = ModelSpec(
    name="gemma-4-31b-it",
    num_params=30.7e9,
    prompt_formatter=format_gemma4_prompt,
    thinking=True,
    model_ids={
        "mlx": {
            "int4": "mlx-community/gemma-4-31b-it-4bit",
            "int8": "mlx-community/gemma-4-31b-it-8bit",
        },
    },
)
