from typing import Any

from mtb.llm_benchmarks.models.base import ModelSpec

__all__ = [
    "Qwen2p5_0p5B_it",
    "Qwen2p5_Coder_0p5B_it",
    "Qwen2p5_3B_it",
    "Qwen2p5_Coder_3B_it",
    "Qwen3_0p6B_it",
    "Qwen3_8B_it",
    "Qwen3_14B_it",
    "Qwen3_32B_it",
]


def format_qwen_prompt(prompt: str) -> Any:
    """Input-finetuned models ('-pi' suffix) expect a specific format."""
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


Qwen2p5_0p5B_it = ModelSpec(
    name="qwen-2.5-0.5b-it",
    num_params=5e8,
    prompt_formatter=format_qwen_prompt,
    model_ids={
        "torch": {
            "bfloat16": "Qwen/Qwen2.5-0.5B-Instruct",
        },
        "mlx": {
            "bfloat16": "mlx-community/Qwen2.5-0.5B-Instruct-bf16",
            "int8": "mlx-community/Qwen2.5-0.5B-Instruct-8bit",
            "int4": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        },
        "lmstudio": {
            "int4": "lmstudio-community/Qwen2.5-0.5B-Instruct-GGUF",
        },
        "ollama": {
            "int4": "qwen2.5:0.5b",
        },
    },
)


Qwen2p5_3B_it = ModelSpec(
    name="qwen-2.5-3b-it",
    num_params=3e9,
    prompt_formatter=format_qwen_prompt,
    model_ids={
        "torch": {
            "bfloat16": "Qwen/Qwen2.5-3B-Instruct",
        },
        "mlx": {
            "bfloat16": "mlx-community/Qwen2.5-3B-Instruct-bf16",
            "int8": "mlx-community/Qwen2.5-3B-Instruct-8bit",
            "int4": "mlx-community/Qwen2.5-3B-Instruct-4bit",
        },
        "lmstudio": {
            "int4": "lmstudio-community/Qwen2.5-3B-Instruct-GGUF",
        },
        "ollama": {
            "int4": "qwen2.5:3b",
        },
    },
)


Qwen2p5_Coder_0p5B_it = ModelSpec(
    name="qwen-2.5-coder-0.5b-it",
    num_params=5e8,
    prompt_formatter=format_qwen_prompt,
    model_ids={
        "torch": {
            "bfloat16": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        },
        "mlx": {
            "bfloat16": "mlx-community/Qwen2.5-Coder-0.5B-Instruct-bf16",
            "int8": "mlx-community/Qwen2.5-Coder-0.5B-Instruct-8bit",
            "int4": "mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit",
        },
        "lmstudio": {
            "int4": "lmstudio-community/Qwen2.5-Coder-0.5B-Instruct-GGUF",
        },
        "ollama": {
            "int4": "qwen2.5-coder:0.5b",
        },
    },
)

Qwen2p5_Coder_3B_it = ModelSpec(
    name="qwen-2.5-coder-3b-it",
    num_params=3e9,
    prompt_formatter=format_qwen_prompt,
    model_ids={
        "torch": {
            "bfloat16": "Qwen/Qwen2.5-Coder-3B-Instruct",
        },
        "mlx": {
            "bfloat16": "mlx-community/Qwen2.5-Coder-3B-Instruct-bf16",
            "int8": "mlx-community/Qwen2.5-Coder-3B-Instruct-8bit",
            "int4": "mlx-community/Qwen2.5-Coder-3B-Instruct-4bit",
        },
        "lmstudio": {
            "int4": "lmstudio-community/Qwen2.5-Coder-3B-Instruct-GGUF",
        },
        "ollama": {
            "int4": "qwen2.5-coder:3b",
        },
    },
)


Qwen3_0p6B_it = ModelSpec(
    name="qwen-3-0.6b-it",
    num_params=int(6e8),
    prompt_formatter=format_qwen_prompt,
    thinking=True,
    model_ids={
        "mlx": {
            "int4": "mlx-community/Qwen3-0.6B-4bit",
            "int8": "mlx-community/Qwen3-0.6B-8bit",
        },
        "lmstudio": {
            "int4": "lmstudio-community/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M.gguf",
            "int8": "lmstudio-community/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf",
        },
        "ollama": {
            "int4": "qwen3:0.6b",
        },
    },
)


Qwen3_8B_it = ModelSpec(
    name="qwen-3-8B-it",
    num_params=int(8e9),
    prompt_formatter=format_qwen_prompt,
    thinking=True,
    model_ids={
        "mlx": {
            "int4": "mlx-community/Qwen3-8B-4bit",
            "int8": "mlx-community/Qwen3-8B-8bit",
        },
        "lmstudio": {
            "int4": "lmstudio-community/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf",
            "int8": "lmstudio-community/Qwen3-8B-GGUF/Qwen3-8B-Q8_0.gguf",
        },
        "ollama": {
            "int4": "qwen3:8b",
        },
    },
    # Speculative decoding drafter (shares the Qwen3 vocab).
    draft_model_ids={
        "mlx": {"int4": "mlx-community/Qwen3-0.6B-4bit"},
    },
)


Qwen3_14B_it = ModelSpec(
    name="qwen-3-14B-it",
    num_params=int(14e9),
    prompt_formatter=format_qwen_prompt,
    thinking=True,
    model_ids={
        "mlx": {
            "int4": "mlx-community/Qwen3-14B-4bit",
            "int8": "mlx-community/Qwen3-14B-8bit",
        },
        "lmstudio": {
            "int4": "lmstudio-community/Qwen3-14B-GGUF/Qwen3-14B-Q4_K_M.gguf",
            "int8": "lmstudio-community/Qwen3-14B-GGUF/Qwen3-14B-Q8_0.gguf",
        },
        "ollama": {
            "int4": "qwen3:14b",
        },
    },
    # Speculative decoding drafter (shares the Qwen3 vocab).
    draft_model_ids={
        "mlx": {"int4": "mlx-community/Qwen3-0.6B-4bit"},
    },
)


# --- 128GB+ models (require >=128GB unified memory) ---

Qwen3_32B_it = ModelSpec(
    name="qwen-3-32B-it",
    num_params=int(32e9),
    prompt_formatter=format_qwen_prompt,
    model_ids={
        "mlx": {
            "int4": "mlx-community/Qwen3-32B-4bit",
            "int8": "mlx-community/Qwen3-32B-8bit",
            "bfloat16": "mlx-community/Qwen3-32B-bf16",
        },
    },
    # Speculative decoding drafter (shares the Qwen3 vocab).
    draft_model_ids={
        "mlx": {"int4": "mlx-community/Qwen3-0.6B-4bit"},
    },
)
