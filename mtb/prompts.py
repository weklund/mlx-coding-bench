import random

from mtb.llm_benchmarks.base_llm_benchmark import BaseLLMBenchmark


class PromptLengthUnreachable(ValueError):
    """Requested prompt length is smaller than the model's chat-template overhead.

    Some models (e.g. gpt-oss with the harmony format) wrap every prompt in a
    large fixed system preamble. When that overhead alone exceeds the requested
    number of prompt tokens, no user text can hit the target and the length must
    be skipped for that model.
    """


def find_prompt_for_llm_benchmark(
    num_tokens: int,
    benchmark: BaseLLMBenchmark,
):
    """Find a prompt of the desired number of tokens.

    As each benchmark tokenizes and formats prompts differently, we need to find
    prompt that has the right number of tokens, but that still makes sense for the
    model.

    Args:
        num_tokens: The number of tokens to find a prompt for.
        benchmark: The benchmark, which contains a tokenization function.

    Returns:
        A prompt with the correct number of tokens for the tokenizer and message template.

    """
    # The prompt profile controls what kind of text the model generates.
    # "generic" prose is close to worst-case acceptance for speculative
    # decoding; "code" is more representative of agentic-coding workloads.
    prompt_builder = (
        get_code_prompt
        if getattr(benchmark, "prompt_profile", "generic") == "code"
        else get_random_prompt
    )

    try:
        prompt_tokens = benchmark.format_and_tokenize_prompt("")
    except NotImplementedError:
        # benchmark does not implement tokenizer: use a very long prompt,
        # and rely on the context length parameter to limit the context.
        prompt = prompt_builder(text_length=5 * num_tokens)
        # sadly, this involves re-initializing the model: context length is a load-time parameter
        benchmark.max_context_length = num_tokens
        benchmark.teardown()
        benchmark.setup()
        return prompt

    # If the chat-template overhead alone already meets or exceeds the target,
    # no user text can bring the total down to num_tokens: skip this length.
    min_num_tokens = len(prompt_tokens)
    if num_tokens <= min_num_tokens:
        raise PromptLengthUnreachable(
            f"Chat-template overhead is {min_num_tokens} tokens, which is "
            f">= the requested prompt length {num_tokens}."
        )

    # initial guess: we need 1 character per token
    text_length = int(num_tokens)
    num_prompt_tokens = None

    # iteratively search for prompts until we hit one with length num_tokens
    while num_prompt_tokens != num_tokens:
        prompt = prompt_builder(text_length=text_length)
        prompt_tokens = benchmark.format_and_tokenize_prompt(prompt)
        num_prompt_tokens = len(prompt_tokens)

        if num_prompt_tokens < num_tokens:
            # prompt too short, increase length by at least 1 character
            text_length += max(1, int((num_tokens - num_prompt_tokens) / 2))
        elif num_prompt_tokens > num_tokens:
            # prompt too long, decrease length by at least 1 character
            text_length -= max(1, int((num_prompt_tokens - num_tokens) / 2))

    return prompt


def get_random_prompt(text_length: int) -> str:
    """Get a prompt of the specified length by generating some text.

    This randomizes initial tokens to avoid prompt caching. Note that the
    text_length is in characters, not tokens.

    We prefer if the prompt is an actual task, instead of just any random
    tokens. This is so we can check the model correctness, and hopefully
    avoid early stopping.

    As an example, the prompt could look like this:

        0, 3, 2. Write a story

    """
    task_string = "Write a story"
    if text_length < len(task_string):
        raise ValueError(
            f"Text_length {text_length} is too small for "
            f"task string '{task_string}' of length {len(task_string)}"
        )

    # add random prefix
    prefix_length = text_length - len(task_string) - 2
    prompt = "".join(str(random.randint(0, 10)) for _ in range(prefix_length)) + ". "

    # actual task
    prompt += task_string

    return prompt


# A realistic partial module used by the "code" prompt profile. Ending the
# prompt inside real code makes the model generate more code — predictable
# syntax and repeated identifiers give a small drafter a much higher acceptance
# rate than free-form prose, closer to agentic-coding use.
_CODE_HEADER = '''\
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class LRUCache:
    """A fixed-capacity least-recently-used cache."""

    capacity: int
    _store: Dict[int, int] = field(default_factory=dict)
    _order: List[int] = field(default_factory=list)

    def get(self, key: int) -> Optional[int]:
        if key not in self._store:
            return None
        self._order.remove(key)
        self._order.append(key)
        return self._store[key]

    def put(self, key: int, value: int) -> None:
        if key in self._store:
            self._order.remove(key)
        elif len(self._store) >= self.capacity:
            oldest = self._order.pop(0)
            del self._store[oldest]
        self._store[key] = value
        self._order.append(key)
'''


def get_code_prompt(text_length: int) -> str:
    """Get a realistic code-continuation prompt of the specified length.

    Same length contract as get_random_prompt (text_length in characters), but
    the content is a Python module the model is asked to extend, so generation
    is code. Speculative-decoding acceptance is highly content-dependent, so
    this profile measures spec-dec closer to real coding workloads.
    """
    instruction = (
        "# Extend the module below: add `peek()`, `__len__`, `clear()`, and a "
        "`stats()` method returning a dict of hits and misses. Keep the same "
        "style, add docstrings, then write a few unit tests. Continue the code:\n\n"
    )
    base = instruction + _CODE_HEADER
    if text_length <= len(base):
        return base[-text_length:]

    # Pad with varying comment lines (random numbers defeat prompt caching)
    # placed before the module, so the instruction + code stay at the end.
    pad_target = text_length - len(base)
    lines = []
    total = 0
    i = 0
    while total < pad_target:
        line = f"# fixture {random.randint(0, 10**9)}: seed row {i}\n"
        lines.append(line)
        total += len(line)
        i += 1
    pad = "".join(lines)[:pad_target]
    return pad + base
