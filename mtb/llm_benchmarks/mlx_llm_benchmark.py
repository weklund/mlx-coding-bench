import gc
from pathlib import Path

import mlx
import mlx.core as mx
import mlx_lm
import mlx_lm.tokenizer_utils

from mtb.hf_utils import verbose_download_model
from mtb.llm_benchmarks.base_llm_benchmark import BaseLLMBenchmark
from mtb.measurement import LlmBenchmarkMeasurement


class MlxLlmBenchmark(BaseLLMBenchmark):
    """Base class for LLM benchmarks in mlx."""

    framework = "mlx"

    def setup(self):
        """Set up the benchmark. Load the model, tokenizer."""
        self._device = {
            "cpu": mx.Device(mx.DeviceType.cpu),
            "metal": mx.Device(mx.DeviceType.gpu),
        }[self.backend]

        mx.set_default_device(self._device)

        if not Path(self.model_id).exists():
            verbose_download_model(self.model_id)
        model, tokenizer = mlx_lm.load(self.model_id)
        self.model: mlx.nn.Module = model
        self.tokenizer: mlx_lm.tokenizer_utils.TokenizerWrapper = tokenizer

        # Speculative decoding: load the drafter alongside the base model.
        self.draft_model = None
        if self.use_speculative and self.draft_model_id:
            if not Path(self.draft_model_id).exists():
                verbose_download_model(self.draft_model_id)
            draft_model, draft_tokenizer = mlx_lm.load(self.draft_model_id)
            # Drafter and target must share a vocabulary for spec-dec validity.
            if draft_tokenizer.vocab_size != tokenizer.vocab_size:
                raise ValueError(
                    f"Drafter vocab ({draft_tokenizer.vocab_size}) != target "
                    f"vocab ({tokenizer.vocab_size}) for "
                    f"'{self.draft_model_id}' -> '{self.model_id}'."
                )
            self.draft_model = draft_model

    def format_and_tokenize_prompt(self, prompt: str) -> mx.array:
        prompt = self.prompt_formatter(prompt)

        kwargs = dict(
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="mlx",
        )
        if self.thinking:
            kwargs["enable_thinking"] = True

        model_input = self.tokenizer.apply_chat_template(
            prompt,
            **kwargs,
        )
        # transformers >=5.x returns mx.array directly instead of a dict
        if isinstance(model_input, mx.array):
            return model_input[0]
        return model_input.input_ids[0]

    def run_once(self, prompt: str) -> LlmBenchmarkMeasurement:
        """Run the mlx benchmark once. Return measurements."""
        mx.reset_peak_memory()
        prompt_tokens = self.format_and_tokenize_prompt(prompt)

        # use stream_generate instead of generate, its response is more useful.
        # draft_model=None is a plain (standard) run; a drafter enables
        # speculative decoding (same output distribution, higher throughput).
        generation = ""
        for response in mlx_lm.stream_generate(
            self.model,
            self.tokenizer,
            max_tokens=self.max_num_tokens,
            prompt=prompt_tokens,
            draft_model=getattr(self, "draft_model", None),
        ):
            generation += response.text

        return LlmBenchmarkMeasurement(
            response=generation,
            prompt_tps=response.prompt_tps,
            prompt_time_sec=response.prompt_tokens / response.prompt_tps,
            generation_time_sec=response.generation_tokens / response.generation_tps,
            generation_tps=response.generation_tps,
            num_prompt_tokens=response.prompt_tokens,
            num_generated_tokens=response.generation_tokens,
            peak_memory_gib=response.peak_memory,
        )

    def teardown(self):
        """Teardown the benchmark."""

        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None

        if getattr(self, "draft_model", None) is not None:
            del self.draft_model
            self.draft_model = None

        mx.clear_cache()
        gc.collect()
