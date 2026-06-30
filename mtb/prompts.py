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
    try:
        prompt_tokens = benchmark.format_and_tokenize_prompt("")
    except NotImplementedError:
        # benchmark does not implement tokenizer: use a very long prompt,
        # and rely on the context length parameter to limit the context.
        prompt = get_random_prompt(text_length=5 * num_tokens)
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
        prompt = get_random_prompt(text_length=text_length)
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
