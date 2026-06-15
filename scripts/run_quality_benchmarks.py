"""Run quality evaluation benchmarks across models and quantizations.

Usage:
    uv run python scripts/run_quality_benchmarks.py
    uv run python scripts/run_quality_benchmarks.py --difficulty hard
    uv run python scripts/run_quality_benchmarks.py --difficulty all
    uv run python scripts/run_quality_benchmarks.py --run_only_benchmarks '["qwen-3.5-4b"]'
    uv run python scripts/run_quality_benchmarks.py --dtypes '["int4","int8"]' --num_runs 5
"""

from pathlib import Path
from typing import Iterable, List, Optional, Union

import fire
import pandas as pd
from tqdm import tqdm

import mtb
import mtb.llm_benchmarks
from mtb.file_io import create_benchmark_output_dir
from mtb.llm_benchmarks.models.base import ModelSpec
from mtb.quality_benchmarks import (
    EVAL_PROBLEMS,
    EXPERT_EVAL_PROBLEMS,
    HARD_EVAL_PROBLEMS,
    TOOL_CALLING_PROBLEMS,
    run_quality_benchmark,
)
from mtb.quality_benchmarks.eval_problem import EvalProblem
from mtb.quality_benchmarks.scoring import compute_weighted_score
from mtb.select_benchmarks import filter_llm_benchmarks

DEFAULT_OUTPUT_ROOT = mtb.REPO_ROOT / "measurements" / "quality_benchmarks"


def main(
    output_root: Union[str, Path] = DEFAULT_OUTPUT_ROOT,
    difficulty: str = "easy",
    dtypes: Iterable[str] = ("int4", "int8", "bfloat16"),
    num_runs: int = 3,
    cooldown_time_fraction: float = 0.05,
    max_tokens_multiplier: float = 1.0,
    hf_cache_dir: Optional[str] = mtb.DEFAULT_HF_HOME,
    *,
    run_only_benchmarks: Optional[Iterable[str]] = None,
    run_mlx_metal: bool = True,
    run_ollama_metal: bool = False,
    run_anthropic_api: bool = False,
    use_variants: bool = False,
    num_variants: int = 3,
):
    """Run quality evaluation benchmarks.

    Args:
        output_root: Root directory for output.
        difficulty: Problem difficulty - "easy", "hard", or "all".
        dtypes: Data types to evaluate.
        num_runs: Number of runs per problem for majority voting.
        cooldown_time_fraction: Cooldown between problems.
        max_tokens_multiplier: Scale factor applied to every problem's
            max_tokens cap. Use >1.0 for verbose thinking models that need a
            larger ceiling to finish reasoning and emit the answer within
            budget (default 1.0 = unchanged).
        hf_cache_dir: HuggingFace cache directory.
        run_only_benchmarks: Optional list of model names to run.
        run_mlx_metal: Whether to run MLX with Metal backend.
        run_ollama_metal: Whether to run Ollama with Metal backend.
        run_anthropic_api: Whether to run Anthropic API models (requires ANTHROPIC_API_KEY).
        use_variants: Replace problems that have generate_variant() with
            freshly generated variants. Helps resist benchmark contamination
            by changing concrete values (numbers, names, constraints) while
            preserving problem structure. Each parameterized problem generates
            num_variants variants, each running as a separate row in the output.
            Non-parameterized problems are kept unchanged.
        num_variants: Number of variants to generate per parameterized problem
            when use_variants is enabled (default 3).

    """
    from mtb.hf_utils import set_hf_home

    set_hf_home(path=hf_cache_dir, enable_hf_progressbar=False)

    # Select problem set based on difficulty
    if difficulty == "easy":
        problems = EVAL_PROBLEMS
    elif difficulty == "hard":
        problems = HARD_EVAL_PROBLEMS
    elif difficulty == "expert":
        problems = EXPERT_EVAL_PROBLEMS
    elif difficulty == "tool_calling":
        problems = TOOL_CALLING_PROBLEMS
    elif difficulty == "all":
        problems = (
            EVAL_PROBLEMS
            + HARD_EVAL_PROBLEMS
            + EXPERT_EVAL_PROBLEMS
            + TOOL_CALLING_PROBLEMS
        )
    else:
        raise ValueError(
            f"Unknown difficulty '{difficulty}', must be 'easy', 'hard', 'expert', 'tool_calling', or 'all'"
        )

    # Apply variant generation for contamination resistance
    if use_variants:
        variant_problems = []
        for p in problems:
            if p.generate_variant is not None:
                for i in range(1, num_variants + 1):
                    variant = p.generate_variant()
                    # Give each variant a distinct name
                    variant = EvalProblem(
                        category=variant.category,
                        name=f"{p.name}_variant_{i}",
                        prompt=variant.prompt,
                        check=variant.check,
                        max_tokens=variant.max_tokens,
                        function_signature=variant.function_signature,
                        test_cases=variant.test_cases,
                        generate_variant=variant.generate_variant,
                    )
                    variant_problems.append(variant)
            else:
                variant_problems.append(p)
        problems = variant_problems

    model_specs: List[ModelSpec] = list(mtb.llm_benchmarks.MODEL_SPECS)

    benchmarks_to_run = filter_llm_benchmarks(
        model_specs=model_specs,
        dtypes=dtypes,
        run_only_benchmarks=run_only_benchmarks,
        run_mlx_metal=run_mlx_metal,
        run_torch_mps=False,
        run_torch_cpu=False,
        run_torch_cuda=False,
        run_mlx_cpu=False,
        run_lmstudio_metal=False,
        run_lmstudio_mlx=False,
        run_ollama_metal=run_ollama_metal,
        run_anthropic_api=run_anthropic_api,
    )

    output_dir = create_benchmark_output_dir(
        output_root=output_root,
        benchmark_settings=dict(
            difficulty=difficulty,
            num_runs=num_runs,
            dtypes=list(dtypes),
            run_only_benchmarks=run_only_benchmarks,
            max_tokens_multiplier=max_tokens_multiplier,
        ),
    )
    output_path = output_dir / "quality_results.csv"
    print(f"\nOutput directory: '{output_dir}'")
    print(f"Difficulty: {difficulty} ({len(problems)} problems x {num_runs} runs each)")
    if max_tokens_multiplier != 1.0:
        print(f"Token cap multiplier: {max_tokens_multiplier}x (raised for verbose thinking models)")
    print(f"Problems: {[p.name for p in problems]}\n")

    with tqdm(benchmarks_to_run, position=0) as iterator:
        for config in iterator:
            model_spec = config["model_spec"]
            iterator.set_description(
                f"{model_spec.name}, {config['framework']}+{config['backend']}, "
                f"dtype={config['dtype']}"
            )

            run_quality_benchmark(
                model_spec=model_spec,
                framework=config["framework"],
                backend=config["backend"],
                dtype=config["dtype"],
                output_path=output_path,
                problems=problems,
                num_runs=num_runs,
                cooldown_time_fraction=cooldown_time_fraction,
                max_tokens_multiplier=max_tokens_multiplier,
            )

    # Print summary
    if output_path.exists():
        print_summary(output_path)

    print(f"\nResults saved to: {output_path}")


def print_summary(output_path: Union[str, Path]) -> None:
    """Print a comprehensive benchmark summary with weighted and raw scores.

    Reads the CSV at output_path and prints:
      - Overall weighted score and raw pass rate per model+dtype
      - Per-category weighted breakdown
      - Per-category raw pass counts
      - Tool calling subcategory breakdowns
      - Failed problems list

    Args:
        output_path: Path to the quality_results.csv file.
    """
    output_path = Path(output_path)
    df = pd.read_csv(output_path)

    print("\n" + "=" * 70)
    print("QUALITY BENCHMARK RESULTS")
    print("=" * 70)

    # ---------------------------------------------------------------
    # Weighted Score + Raw Pass Rate per model+dtype
    # ---------------------------------------------------------------
    model_dtype_groups = df.groupby(["model", "dtype"])

    weighted_rows = []
    for (model, dtype), group_df in model_dtype_groups:
        # Build results dict for compute_weighted_score
        results = {}
        for _, row in group_df.iterrows():
            results[row["problem"]] = bool(row["passed"])

        scores = compute_weighted_score(results)
        total = len(group_df)
        passed = int(group_df["passed"].sum())

        weighted_rows.append(
            {
                "model": model,
                "dtype": dtype,
                "weighted_score": f"{scores['weighted_score'] * 100:.1f}%",
                "raw_pass_rate": f"{scores['raw_pass_rate'] * 100:.1f}% ({passed}/{total})",
            }
        )

    ws_df = pd.DataFrame(weighted_rows)

    print("\nWeighted Score (per model+dtype):")
    ws_pivot = ws_df.pivot(index="model", columns="dtype", values="weighted_score")
    print(ws_pivot.to_string())

    print("\nRaw Pass Rate (per model+dtype):")
    raw_pivot = ws_df.pivot(index="model", columns="dtype", values="raw_pass_rate")
    print(raw_pivot.to_string())

    # ---------------------------------------------------------------
    # Per-category weighted breakdown
    # ---------------------------------------------------------------
    print("\nCategory Weighted Breakdown:")
    for (model, dtype), group_df in model_dtype_groups:
        results = {}
        for _, row in group_df.iterrows():
            results[row["problem"]] = bool(row["passed"])

        scores = compute_weighted_score(results)
        cat_scores = scores.get("category_scores", {})

        if cat_scores:
            print(f"\n  {model} ({dtype}):")
            for cat_name in sorted(cat_scores.keys()):
                cat_score = cat_scores[cat_name]
                # Also show raw counts for this category
                cat_df = group_df[group_df["category"] == cat_name]
                cat_passed = int(cat_df["passed"].sum())
                cat_total = len(cat_df)
                print(
                    f"    {cat_name.replace('_', ' ').title():30s} "
                    f"Weighted: {cat_score * 100:.1f}%  "
                    f"Raw: {cat_passed}/{cat_total}"
                )

    # ---------------------------------------------------------------
    # By category (raw pass counts)
    # ---------------------------------------------------------------
    for category in df["category"].unique():
        cat_df = df[df["category"] == category]
        cat_summary = (
            cat_df.groupby(["model", "dtype"])
            .agg(passed=("passed", "sum"), total=("passed", "count"))
            .reset_index()
        )
        cat_summary["score"] = cat_summary.apply(
            lambda r: f"{int(r['passed'])}/{int(r['total'])}", axis=1
        )
        cat_pivot = cat_summary.pivot(index="model", columns="dtype", values="score")
        print(f"\n{category.replace('_', ' ').title()}:")
        print(cat_pivot.to_string())

    # ---------------------------------------------------------------
    # Tool calling subcategory breakdowns
    # ---------------------------------------------------------------
    tc_df = df[df["category"] == "tool_calling"]
    if not tc_df.empty:
        _SUBCATEGORY_MAP = {
            "ts_": "Tool Selection",
            "aa_": "Argument Accuracy",
            "mt_": "Multi-Tool",
            "ec_": "Edge Cases",
            "fc_": "Format Compliance",
        }

        def _get_subcategory(name: str) -> str:
            for prefix, label in _SUBCATEGORY_MAP.items():
                if name.startswith(prefix):
                    return label
            return "Other"

        tc_df = tc_df.copy()
        tc_df["subcategory"] = tc_df["problem"].apply(_get_subcategory)
        print("\nTool Calling Subcategory Breakdown:")
        for subcat in _SUBCATEGORY_MAP.values():
            sub_df = tc_df[tc_df["subcategory"] == subcat]
            if sub_df.empty:
                continue
            sub_summary = (
                sub_df.groupby(["model", "dtype"])
                .agg(passed=("passed", "sum"), total=("passed", "count"))
                .reset_index()
            )
            sub_summary["score"] = sub_summary.apply(
                lambda r: f"{int(r['passed'])}/{int(r['total'])}", axis=1
            )
            sub_pivot = sub_summary.pivot(
                index="model", columns="dtype", values="score"
            )
            print(f"\n  {subcat}:")
            print("  " + sub_pivot.to_string().replace("\n", "\n  "))

    # ---------------------------------------------------------------
    # Show any failures
    # ---------------------------------------------------------------
    failures = df[~df["passed"]]
    if not failures.empty:
        print(f"\nFailed problems ({len(failures)}):")
        for _, row in failures.iterrows():
            print(
                f"  {row['model']} {row['dtype']}: {row['problem']} "
                f"({row['pass_count']}/{row['num_runs']} runs)"
            )


if __name__ == "__main__":
    fire.Fire(main)
