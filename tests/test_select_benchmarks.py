import pytest

from mtb.llm_benchmarks.models.base import ModelSpec
from mtb.select_benchmarks import filter_benchmarks


def _make_spec(name: str) -> ModelSpec:
    return ModelSpec(name=name, num_params=1e9, prompt_formatter=lambda p: [])


@pytest.fixture()
def model_specs():
    return [
        _make_spec("qwen-3-8b-it"),
        _make_spec("gemma-4-e2b-it"),
        _make_spec("phi-4-mini"),
    ]


def test_filter_benchmarks_list(model_specs):
    filtered = filter_benchmarks(model_specs, ["qwen", "phi"])
    assert len(filtered) == 2
    assert filtered[0].name == "qwen-3-8b-it"
    assert filtered[1].name == "phi-4-mini"


def test_filter_benchmarks_str(model_specs):
    filtered = filter_benchmarks(model_specs, "gemma")
    assert len(filtered) == 1
    assert filtered[0].name == "gemma-4-e2b-it"


def test_filter_benchmarks_no_match(model_specs):
    with pytest.raises(ValueError, match="No benchmarks to run"):
        filter_benchmarks(model_specs, ["nonexistent"])
