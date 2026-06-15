"""Shared utility functions for quality benchmark evaluation.

These helpers are used across all category problem files and the runner.
"""

import re
from typing import List


def _strip_thinking(response: str) -> str:
    """Strip thinking/reasoning preambles from model responses.

    Handles three formats:
    1. <think>...</think> blocks (Claude-distilled models)
    2. Freeform "Thinking Process:" preambles (base Qwen 3.5 models)
    3. A bare trailing </think> with no opening tag: some chat templates
       (e.g. Qwen3.x with enable_thinking) prefill the opening <think> into
       the prompt, so the response begins mid-reasoning and only emits the
       closing </think> before the answer.

    Our checks should evaluate the actual answer only.
    """
    # Remove all <think>...</think> blocks (greedy, handles multiline)
    stripped = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    # Also handle unclosed <think> blocks (response truncated mid-thinking)
    stripped = re.sub(r"<think>.*", "", stripped, flags=re.DOTALL)
    # Handle a bare closing </think> whose opening tag was in the prompt:
    # drop everything up to and including the (last) </think>.
    if "</think>" in stripped:
        stripped = re.sub(r"^.*</think>", "", stripped, flags=re.DOTALL)

    # Handle freeform thinking preambles (e.g., "Thinking Process:" or
    # "Here's a thinking process that leads to the solution:")
    # These appear at the start and consist of numbered analysis steps,
    # followed by the actual answer after a clear section break.
    thinking_start = re.match(
        r"^(?:(?:Here\'s a )?[Tt]hinking [Pp]rocess.*?[:\n]|"
        r"(?:The user wants|Let me (?:think|analyze|break down|work through)).*?\n)",
        stripped,
    )
    if thinking_start:
        # Strategy: find the first "final answer" marker after the thinking.
        # Models produce headers like "*Revised Draft:*", "**Rewrite:**",
        # "Here's the implementation:", etc. We take everything from that
        # marker to the end, which includes the answer plus any follow-up
        # explanation — this is fine because the answer content dominates.
        #
        # For models that draft multiple attempts (Draft -> Revised Draft),
        # we prefer "final/revised" markers over plain "draft" markers.
        # First try to find a "final" marker; fall back to any marker.
        final_pattern = re.compile(
            r"\n\s*"
            r"(?:\*{1,2}|#{1,3}\s*)?"
            r"(?:Final Answer|Final Polish|Revised Draft|Final|"
            r"Here\'s (?:the|my) (?:implementation|solution|rewrite|answer))"
            r"(?:\*{1,2})?"
            r"\s*[:：\n]",
            re.IGNORECASE,
        )
        any_answer_pattern = re.compile(
            r"\n\s*"
            r"(?:\*{1,2}|#{1,3}\s*)?"
            r"(?:Answer|Solution|Result|Output|Rewrite|Rewritten|Draft|"
            r"Summary|Response|Implementation|"
            r"Here\'s (?:the|my)|Here is (?:the|my)|"
            r"Non-technical|Simplified|Plain)"
            r"(?:\*{1,2})?"
            r"\s*[:：\n]",
            re.IGNORECASE,
        )
        # Prefer final markers (last occurrence), fall back to first general marker
        match = None
        for m in final_pattern.finditer(stripped):
            match = m  # Take the last "final" marker
        if not match:
            match = any_answer_pattern.search(stripped)  # Take first general marker
        if match:
            remainder = stripped[match.end() :].strip()
            if len(remainder) > 20:  # Sanity check: must have substantial content
                stripped = remainder

    return stripped.strip()


def _contains_any(response: str, targets: List[str]) -> bool:
    """Check if response contains any of the target strings (case-insensitive)."""
    response_lower = response.lower()
    return any(t.lower() in response_lower for t in targets)


def _extract_number(response: str) -> float | None:
    """Extract the last number from a response (models often reason then give final answer)."""
    numbers = re.findall(r"-?\d+\.?\d*", response)
    if numbers:
        return float(numbers[-1])
    return None
