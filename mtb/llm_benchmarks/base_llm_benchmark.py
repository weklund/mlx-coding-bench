from typing import Any, Callable, Iterable

from mtb.measurement import LlmBenchmarkMeasurement
from mtb.system.memory import get_process_memory_gib


class BaseLLMBenchmark:
    """Base class for LLM benchmarks.

    Should implement:

      1. `setup`: Initialize the model and tokenizer. Prepare the prompt.
      2. `run_once`: Run the benchmark once.
      3. `teardown`: Cleanup.

    """

    # Name of the framework used for the benchmark
    framework = None

    def __init__(
        self,
        name: str,
        model_id: str,
        backend: str,
        dtype: str,
        prompt_formatter: Callable[[str], Any],
        max_num_tokens: int = 100,
        thinking: bool = False,
        draft_model_id: str = None,
        use_speculative: bool = False,
        prompt_profile: str = "generic",
    ):
        self.name = name
        self.model_id = model_id
        self.backend = backend
        self.dtype = dtype
        self.prompt_formatter = prompt_formatter
        self.max_num_tokens = max_num_tokens
        self.thinking = thinking
        # Speculative decoding: an optional small drafter that proposes tokens
        # the base model verifies in parallel. Only the mlx backend uses these.
        self.draft_model_id = draft_model_id
        self.use_speculative = use_speculative
        # Prompt profile ("generic" prose vs "code") — affects what the model
        # generates, which strongly affects speculative-decoding acceptance.
        self.prompt_profile = prompt_profile

        # track memory allocated by the process after this benchmark
        self.initial_process_memory_gib = get_process_memory_gib()

    @property
    def optimization(self) -> str:
        """Name of the optimization layered on the base model.

        "" means the standard/unoptimized config. Each optimization method
        (speculative decoding today; others later) reports its own name here so
        the runner and README can treat optimizations generically.
        """
        if self.use_speculative:
            return "speculative_decoding"
        return ""

    @property
    def optimization_detail(self) -> str:
        """Human-readable config for the active optimization (e.g. drafter id)."""
        if self.use_speculative and self.draft_model_id:
            return self.draft_model_id
        return ""

    def format_and_tokenize_prompt(self, prompt: str) -> Iterable:
        """Format and tokenize the prompt. Return a list, array or tensor of tokens."""
        raise NotImplementedError

    def get_num_prompt_tokens(self, user_prompt: str) -> int:
        """Get the number of tokens for a given user prompt."""
        tokens = self.tokenize(user_prompt)
        return len(tokens)

    def setup(self):
        """Set up the benchmark. Load the model, tokenizer."""
        raise NotImplementedError

    def run_once(self) -> LlmBenchmarkMeasurement:
        """Run the benchmark once. Return measurements."""
        raise NotImplementedError

    def teardown(self):
        """Teardown the benchmark."""
        raise NotImplementedError
