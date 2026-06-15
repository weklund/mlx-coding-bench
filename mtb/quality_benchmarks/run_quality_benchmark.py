"""Run quality evaluation benchmarks to measure reasoning accuracy across quantizations."""

import json
import math
import time
from pathlib import Path
from typing import List, Optional

import pandas as pd
from tqdm import tqdm

from mtb.llm_benchmarks.models.base import ModelSpec
from mtb.quality_benchmarks.eval_problems import EVAL_PROBLEMS, EvalProblem
from mtb.quality_benchmarks.sandbox import SandboxResult, execute_code
from mtb.quality_benchmarks.tool_call_parser import parse_tool_calls
from mtb.quality_benchmarks.utils import _strip_thinking


def _build_test_code(problem: EvalProblem, code: str) -> str:
    """Build executable code combining extracted model code + test assertions.

    For problems with test_cases, combines the model's implementation with
    assertion-based test cases for sandbox execution.

    Args:
        problem: The evaluation problem with function_signature and test_cases.
        code: The extracted code from the model's response.

    Returns:
        A string of Python code ready for sandbox execution.
    """
    func_name = problem.function_signature.split("(")[0].replace("def ", "")
    lines = [code, ""]
    for i, tc in enumerate(problem.test_cases):
        inp = tc["input"]
        expected = tc["expected_output"]
        # Use approximate comparison for floats
        if isinstance(expected, float):
            lines.append(
                f"assert abs({func_name}({inp}) - {expected!r}) < 1e-6, "
                f"'Test case {i} failed'"
            )
        else:
            lines.append(
                f"assert {func_name}({inp}) == {expected!r}, " f"'Test case {i} failed'"
            )
    lines.append("print('ALL TESTS PASSED')")
    return "\n".join(lines)


def _evaluate_with_sandbox(problem: EvalProblem, response: str) -> tuple:
    """Evaluate a code-execution problem by running model code in sandbox.

    Extracts code from the model response, combines it with test cases,
    and executes in the sandbox. Handles timeouts and errors gracefully.

    Args:
        problem: The coding problem with function_signature and test_cases.
        response: The raw model response containing code.

    Returns:
        Tuple of (passed: bool, sandbox_result: SandboxResult).
    """
    from mtb.quality_benchmarks.sandbox import _strip_markdown_fences

    # Strip the thinking section first. Verbose thinking models (e.g. Qwen3.6)
    # draft indented code blocks mid-reasoning; without this, the fence
    # extractor would grab an early draft from the thinking instead of the
    # clean final answer after </think>.
    response = _strip_thinking(response)

    # Extract code from response (strip markdown fences)
    extracted_code = _strip_markdown_fences(response)

    # Build test code combining implementation + assertions
    test_code = _build_test_code(problem, extracted_code)

    # Execute in sandbox with timeout
    result = execute_code(test_code, timeout=15)

    return result.success, result


def _has_code_execution(problem: EvalProblem) -> bool:
    """Check if a problem should use code execution evaluation.

    A problem uses sandbox execution when it has both test_cases and
    function_signature set (i.e., it's a coding problem converted to
    code-execution-based evaluation).

    Args:
        problem: The evaluation problem to check.

    Returns:
        True if the problem should use sandbox execution.
    """
    return problem.test_cases is not None and problem.function_signature is not None


def _is_tool_calling(problem: EvalProblem) -> bool:
    """Check if a problem is a tool calling evaluation problem.

    Tool calling problems have category 'tool_calling' and are evaluated
    using structured tool call parsing followed by check().

    Args:
        problem: The evaluation problem to check.

    Returns:
        True if the problem is a tool calling problem.
    """
    return problem.category == "tool_calling"


def _evaluate_tool_calling(problem: EvalProblem, response: str) -> tuple:
    """Evaluate a tool calling problem with structured parsing and logging.

    Parses the model response through the tool call parser for debugging,
    then passes the original response to the problem's check() function
    (which internally also uses the parser for validation).

    Args:
        problem: The tool calling problem.
        response: The raw model response.

    Returns:
        Tuple of (passed: bool, parsed_tool_calls_json: str).
        The parsed_tool_calls_json is a JSON string of extracted tool calls
        for debugging/logging, or "null" if no tool calls were found.
    """
    # Parse tool calls for debugging/logging
    parsed = parse_tool_calls(response)
    if parsed:
        parsed_json = json.dumps(
            [{"name": tc.name, "arguments": tc.arguments} for tc in parsed],
            default=str,
        )
    else:
        parsed_json = "null"

    # Use the problem's check function for pass/fail determination
    # (check functions already use the parser internally)
    passed = problem.check(response)

    return passed, parsed_json


def run_quality_benchmark(
    model_spec: ModelSpec,
    framework: str,
    backend: str,
    dtype: str,
    output_path: Path,
    problems: Optional[List[EvalProblem]] = None,
    num_runs: int = 3,
    cooldown_time_fraction: float = 0.05,
    max_tokens_multiplier: float = 1.0,
) -> pd.DataFrame:
    """Run quality evaluation for a single model+dtype configuration.

    Each problem is run num_runs times to account for sampling variance.
    A problem "passes" if it passes in the majority of runs.

    For problems with test_cases (code-execution problems), the model's
    generated code is extracted and executed in a sandbox against the test
    cases. For all other problems, the check() callback is used for
    pattern-match evaluation.

    Args:
        model_spec: The model specification.
        framework: Framework identifier (e.g. "mlx").
        backend: Backend identifier (e.g. "metal").
        dtype: Data type identifier (e.g. "int4", "int8", "bfloat16").
        output_path: Path to save results CSV.
        problems: List of eval problems. Defaults to EVAL_PROBLEMS.
        num_runs: Number of times to run each problem (for majority vote).
        cooldown_time_fraction: Fraction of elapsed time to sleep between problems.

    Returns:
        DataFrame with results.

    """
    if problems is None:
        problems = EVAL_PROBLEMS

    from mtb.llm_benchmarks.run_llm_benchmark import create_benchmark

    benchmark = create_benchmark(
        model_spec=model_spec,
        framework=framework,
        backend=backend,
        dtype=dtype,
        max_num_tokens=512,  # will be overridden per problem
    )

    try:
        benchmark.setup()

        results = []
        for problem in tqdm(
            problems, desc=f"{model_spec.name} {dtype}", position=1, leave=False
        ):
            # Scale the per-problem token cap. Verbose thinking models (e.g.
            # Qwen3.6) need a larger ceiling to close </think> and emit the
            # answer within budget; generation still stops at EOS, so a higher
            # ceiling only costs time on problems that would otherwise truncate.
            benchmark.max_num_tokens = int(problem.max_tokens * max_tokens_multiplier)

            prompt = problem.prompt
            is_code_exec = _has_code_execution(problem)
            is_tool_call = _is_tool_calling(problem)

            run_passes = []
            run_responses = []
            run_gen_times = []
            run_gen_tps = []
            run_gen_tokens = []
            # Track last sandbox result for execution detail columns
            last_sandbox_result: Optional[SandboxResult] = None
            # Track last parsed tool calls for debugging
            last_parsed_tool_calls: Optional[str] = None

            start_time = time.perf_counter()
            for run_idx in range(num_runs):
                measurement = benchmark.run_once(prompt=prompt)
                response = measurement.response

                if is_code_exec:
                    # Code execution: extract code, run in sandbox
                    try:
                        passed, sandbox_result = _evaluate_with_sandbox(
                            problem, response
                        )
                        last_sandbox_result = sandbox_result
                    except Exception:
                        # Sandbox failure should not crash the runner
                        passed = False
                        last_sandbox_result = SandboxResult(
                            success=False,
                            stdout="",
                            stderr="sandbox evaluation error",
                            error_type="runtime_error",
                            return_code=-1,
                            execution_time_sec=0.0,
                        )
                elif is_tool_call:
                    # Tool calling: parse response for structured tool calls,
                    # then use check() for pass/fail
                    try:
                        passed, parsed_json = _evaluate_tool_calling(problem, response)
                        last_parsed_tool_calls = parsed_json
                    except Exception:
                        # Tool call evaluation failure should not crash the runner
                        passed = False
                        last_parsed_tool_calls = "null"
                else:
                    # Pattern matching: use check() callback
                    passed = problem.check(response)

                run_passes.append(passed)
                run_responses.append(response)
                run_gen_times.append(measurement.generation_time_sec)
                run_gen_tps.append(measurement.generation_tps)
                run_gen_tokens.append(measurement.num_generated_tokens)

            elapsed = time.perf_counter() - start_time
            time.sleep(cooldown_time_fraction * elapsed)

            # Majority vote
            pass_count = sum(run_passes)
            majority_pass = pass_count > num_runs / 2

            # Average speed metrics across runs
            avg_gen_time = sum(run_gen_times) / len(run_gen_times)
            avg_gen_tps = sum(run_gen_tps) / len(run_gen_tps)
            avg_gen_tokens = sum(run_gen_tokens) / len(run_gen_tokens)

            result = dict(
                model=model_spec.name,
                framework=framework,
                backend=backend,
                dtype=dtype,
                category=problem.category,
                problem=problem.name,
                pass_count=pass_count,
                num_runs=num_runs,
                passed=majority_pass,
                avg_generation_time_sec=round(avg_gen_time, 2),
                avg_generation_tps=round(avg_gen_tps, 1),
                avg_tokens_generated=round(avg_gen_tokens, 0),
                # Store the first response for debugging
                sample_response=run_responses[0][:500],
                # Execution detail columns (NaN for non-code-execution problems)
                execution_stdout=(
                    last_sandbox_result.stdout[:500]
                    if last_sandbox_result is not None
                    else float("nan")
                ),
                execution_stderr=(
                    last_sandbox_result.stderr[:500]
                    if last_sandbox_result is not None
                    else float("nan")
                ),
                execution_exit_code=(
                    last_sandbox_result.return_code
                    if last_sandbox_result is not None
                    else float("nan")
                ),
                # Parsed tool calls for debugging (NaN for non-tool-calling problems)
                parsed_tool_calls=(
                    last_parsed_tool_calls
                    if last_parsed_tool_calls is not None
                    else float("nan")
                ),
            )
            results.append(result)

        benchmark.teardown()

    except Exception as e:
        print(f"\n  Exception for '{model_spec.name}' {dtype}: {e}")
        return pd.DataFrame()

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(
        output_path,
        index=False,
        mode="a",
        header=(not output_path.exists()),
    )

    return df
