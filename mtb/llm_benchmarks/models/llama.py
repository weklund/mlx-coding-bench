from typing import Any

from mtb.llm_benchmarks.models.base import ModelSpec

__all__ = [
    "Llama4_Scout_17B_16E_it",
    "Llama3p3_70B_it",
]


def format_llama_prompt(prompt: str) -> Any:
    """Llama instruct models expect a system/user prompt format."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    return messages


# --- 128GB+ models (MoE: 17B active / 109B total — int4 ~54GB weights, no headroom on 64GB) ---

Llama4_Scout_17B_16E_it = ModelSpec(
    name="llama-4-scout-17b-16e-it",
    num_params=int(17e9),
    prompt_formatter=format_llama_prompt,
    model_ids={
        "mlx": {
            "int4": "mlx-community/Llama-4-Scout-17B-16E-Instruct-4bit",
            "int8": "mlx-community/Llama-4-Scout-17B-16E-Instruct-8bit",
        },
    },
)


# --- 128GB+ models (require >=128GB unified memory) ---

Llama3p3_70B_it = ModelSpec(
    name="llama-3.3-70b-it",
    num_params=int(70e9),
    prompt_formatter=format_llama_prompt,
    model_ids={
        "mlx": {
            "int4": "mlx-community/Llama-3.3-70B-Instruct-4bit",
            "int8": "mlx-community/Llama-3.3-70B-Instruct-8bit",
        },
    },
)
