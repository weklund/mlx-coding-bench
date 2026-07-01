from typing import Any

from mtb.llm_benchmarks.models.base import ModelSpec

__all__ = [
    "Gemma3_1B_it",
    "Gemma3_4B_it",
    "Gemma3_1B_it_QAT",
    "Gemma3_4B_it_QAT",
    "Gemma3_12B_it_QAT",
    "Gemma3_27B_it",
]


def format_gemma_prompt(prompt: str) -> Any:
    """Gemma models expect a system prompt with data types."""
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


Gemma3_1B_it = ModelSpec(
    name="gemma-3-1b-it",
    num_params=1e9,
    prompt_formatter=format_gemma_prompt,
    model_ids={
        "torch": {
            "bfloat16": "google/gemma-3-1b-it",
        },
        "mlx": {
            "bfloat16": "mlx-community/gemma-3-1b-it-bf16",
            "int8": "mlx-community/gemma-3-1b-it-8bit",
            "int4": "mlx-community/gemma-3-1b-it-4bit",
        },
        "lmstudio": {
            "int3": "lmstudio-community/gemma-3-1B-it-GGUF/gemma-3-1B-it-Q3_K_L.gguf",
            "int4": "lmstudio-community/gemma-3-1B-it-GGUF/gemma-3-1B-it-Q4_K_M.gguf",
            "int6": "lmstudio-community/gemma-3-1B-it-GGUF/gemma-3-1B-it-Q6_K.gguf",
            "int8": "lmstudio-community/gemma-3-1B-it-GGUF/gemma-3-1b-it-Q8_0.gguf",
        },
        "ollama": {
            "int4": "gemma3:1b-it-Q4_K_M",
            "int8": "gemma3:1b-it-Q8_0",
        },
    },
)


Gemma3_4B_it = ModelSpec(
    name="gemma-3-4b-it",
    num_params=4e9,
    prompt_formatter=format_gemma_prompt,
    model_ids={
        "torch": {
            "bfloat16": "google/gemma-3-4b-it",
        },
        "mlx": {
            "bfloat16": "mlx-community/gemma-3-4b-it-bf16",
            "int8": "mlx-community/gemma-3-4b-it-8bit",
            "int4": "mlx-community/gemma-3-4b-it-4bit",
        },
        "lmstudio": {
            "int3": "lmstudio-community/gemma-3-4b-it-GGUF/gemma-3-4b-it-Q3_K_L.gguf",
            "int4": "lmstudio-community/gemma-3-4b-it-GGUF/gemma-3-4B-it-Q4_K_M.gguf",
            "int6": "lmstudio-community/gemma-3-4b-it-GGUF/gemma-3-4B-it-Q6_K.gguf",
            "int8": "lmstudio-community/gemma-3-4b-it-GGUF/gemma-3-4b-it-Q8_0.gguf",
        },
        "ollama": {
            "int4": "gemma3:4b-it-q4_K_M",
            "int8": "gemma3:4b-it-q8_0",
        },
    },
)


Gemma3_1B_it_QAT = ModelSpec(
    name="gemma-3-1b-it-qat",
    num_params=1e9,
    prompt_formatter=format_gemma_prompt,
    model_ids={
        "torch": {
            "int4": "google/gemma-3-1b-it-qat-q4_0-gguf",
        },
        "mlx": {
            "bfloat16": "mlx-community/gemma-3-1b-it-qat-bf16",
            "int8": "mlx-community/gemma-3-1b-it-qat-8bit",
            "int4": "mlx-community/gemma-3-1b-it-qat-4bit",
        },
        "lmstudio": {
            "int4": "lmstudio-community/gemma-3-1B-it-qat-GGUF/gemma-3-1B-it-QAT-Q4_0.gguf",
        },
        "ollama": {
            "int4": "gemma3:1b-qat",
        },
    },
)


Gemma3_4B_it_QAT = ModelSpec(
    name="gemma-3-4b-it-qat",
    num_params=4e9,
    prompt_formatter=format_gemma_prompt,
    model_ids={
        # TODO: hf repo provides no pytorch_model.bin for the 4bit quantized model
        # "torch": {
        #    "int4": "google/gemma-3-4b-it-qat-q4_0-gguf",
        # },
        "mlx": {
            "bfloat16": "mlx-community/gemma-3-4b-it-qat-bf16",
            "int8": "mlx-community/gemma-3-4b-it-qat-8bit",
            "int4": "mlx-community/gemma-3-4b-it-qat-4bit",
        },
        "lmstudio": {
            "int4": "lmstudio-community/gemma-3-4B-it-qat-GGUF/gemma-3-4B-it-QAT-Q4_0.gguf",
        },
        "ollama": {
            "int4": "gemma3:4b-it-qat",
        },
    },
)


Gemma3_12B_it_QAT = ModelSpec(
    name="gemma-3-12b-it-qat",
    num_params=12e9,
    prompt_formatter=format_gemma_prompt,
    model_ids={
        # TODO: hf repo provides no pytorch_model.bin for the 4bit quantized model
        # "torch": {
        #    "int4": "google/gemma-3-12b-it-qat-q4_0-gguf",
        # },
        "mlx": {
            "bfloat16": "mlx-community/gemma-3-12b-it-qat-bf16",
            "int8": "mlx-community/gemma-3-12b-it-qat-8bit",
            "int4": "mlx-community/gemma-3-12b-it-qat-4bit",
        },
        "lmstudio": {
            "int4": "lmstudio-community/gemma-3-12B-it-qat-GGUF/gemma-3-12B-it-QAT-Q4_0.gguf",
        },
        "ollama": {
            "int4": "gemma3:12b-it-qat",
        },
    },
    # Speculative decoding drafter (shares the Gemma vocab). Cross-family test:
    # Gemma-3's sliding-window attention may or may not expose a trimmable cache.
    draft_model_ids={
        "mlx": {"int4": "mlx-community/gemma-3-1b-it-qat-4bit"},
    },
)


# --- 128GB+ models (require >=128GB unified memory) ---

Gemma3_27B_it = ModelSpec(
    name="gemma-3-27b-it",
    num_params=27e9,
    prompt_formatter=format_gemma_prompt,
    model_ids={
        "mlx": {
            "int4": "mlx-community/gemma-3-27b-it-4bit",
            "int8": "mlx-community/gemma-3-27b-it-8bit",
            "bfloat16": "mlx-community/gemma-3-27b-it-bf16",
        },
    },
)
