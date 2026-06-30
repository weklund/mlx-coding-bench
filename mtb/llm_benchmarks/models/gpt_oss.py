from typing import Any

from mtb.llm_benchmarks.models.base import ModelSpec

__all__ = [
    "GptOss_20B",
]


def format_gpt_oss_prompt(prompt: str) -> Any:
    """OpenAI gpt-oss uses the harmony response format.

    The harmony channels (analysis/final) are applied by the tokenizer's chat
    template, so the formatter just returns standard system/user messages.
    gpt-oss is a reasoning model (adjustable reasoning_effort, default medium).
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


# MoE: 21B total / 3.6B active. MLX builds are native MXFP4 for the experts;
# the Q4/Q8 suffix is the precision of the remaining (non-expert) weights.
GptOss_20B = ModelSpec(
    name="gpt-oss-20b",
    num_params=int(3.6e9),
    prompt_formatter=format_gpt_oss_prompt,
    thinking=True,
    model_ids={
        "mlx": {
            "int4": "mlx-community/gpt-oss-20b-MXFP4-Q4",
            "int8": "mlx-community/gpt-oss-20b-MXFP4-Q8",
            "bfloat16": "mlx-community/gpt-oss-20b-mxfp4-bf16",
        },
    },
)
