# Contributing to mlx-coding-bench

There are two main ways to contribute: submitting benchmark results for new hardware, or adding a new model.

You will need:
 - [`uv`](https://github.com/astral-sh/uv) to manage dependencies, available as [homebrew](https://formulae.brew.sh/formula/uv)

First, [fork the repo](https://github.com/weklund/mlx-coding-bench/fork) and set up a local environment:
```
git clone git@github.com:<your-username>/mlx-coding-bench.git
cd mlx-coding-bench
make setup
```

Check installation:
```
make test
```

### Adding a measurement

1. Run a speed benchmark for a specific model:
   ```
   uv run python scripts/run_llm_benchmarks.py \
       --run_only_benchmarks '["qwen-3.6-27b"]' \
       --dtypes '["int4","int8"]' \
       --num_iterations 3
   ```
   This creates a timestamped CSV in `measurements/llm_benchmarks/<hardware>/`.

2. Run quality benchmarks (takes ~40 min for MoE models, longer for large dense):
   ```
   uv run python scripts/run_quality_benchmarks.py \
       --difficulty all \
       --run_only_benchmarks '["qwen-3.6-27b"]' \
       --dtypes '["int4"]' \
       --num_runs 3
   ```

3. Update the README table to include your results:
   ```
   uv run python scripts/update_readme_table.py
   ```

4. Commit and submit a PR:
   ```
   git add measurements/ README.md
   git commit -m "Add benchmarks for <model> on <hardware>"
   git push
   ```

### Adding a model

Models live in `mtb/llm_benchmarks/models/`. Each model family has its own file.

1. Create or update a model file with a `ModelSpec`:
   ```python
   from mtb.llm_benchmarks.models.base import ModelSpec

   MyModel = ModelSpec(
       name="my-model-7b",
       num_params=7e9,
       prompt_formatter=format_my_model_prompt,
       model_ids={
           "mlx": {
               "int4": "mlx-community/my-model-7b-4bit",
               "int8": "mlx-community/my-model-7b-8bit",
           },
       },
   )
   ```

2. Register it in `mtb/llm_benchmarks/__init__.py` (import + add to `MODEL_SPECS`).

3. Verify: `uv run python -c "from mtb.llm_benchmarks import MODEL_SPECS; print(len(MODEL_SPECS))"`

See [CLAUDE.md](CLAUDE.md) for detailed `ModelSpec` field documentation and common issues.
