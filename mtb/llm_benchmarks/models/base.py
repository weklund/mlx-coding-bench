from dataclasses import dataclass, field
from typing import Callable, Dict, Optional


@dataclass
class ModelSpec:
    # identifier for the model
    name: str
    # number of parameters in billions
    num_params: float
    # Function that formats the prompt
    prompt_formatter: Callable
    # model_id for each framework and dtype
    model_ids: Dict[str, Dict[str, str]] = field(default_factory=dict)
    # whether this model uses a thinking/reasoning chat template
    thinking: bool = False
    # Optional drafter model_ids for speculative decoding, same shape as
    # model_ids. A drafter must share this model's tokenizer/vocabulary.
    # Drafters are small and dtype-agnostic, so int4 is typically listed.
    draft_model_ids: Dict[str, Dict[str, str]] = field(default_factory=dict)

    def has_model_id(self, framework: str, dtype: str) -> bool:
        """Check if we have a model_id for the framework and dtype."""
        return framework in self.model_ids and dtype in self.model_ids[framework]

    def get_draft_model_id(self, framework: str, dtype: str) -> Optional[str]:
        """Return the speculative-decoding drafter id for framework/dtype.

        Drafters are small and largely dtype-agnostic, so we fall back to the
        int4 drafter (or any listed) if the exact dtype isn't specified.
        Returns None if this model has no drafter configured.
        """
        framework_drafters = self.draft_model_ids.get(framework, {})
        if not framework_drafters:
            return None
        return (
            framework_drafters.get(dtype)
            or framework_drafters.get("int4")
            or next(iter(framework_drafters.values()))
        )
