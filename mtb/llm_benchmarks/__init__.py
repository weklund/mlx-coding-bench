from mtb.llm_benchmarks.models.nemotron import (
    Nemotron3_Nano_4B,
    Nemotron_Nano_9B_v2,
    Nemotron_Cascade2_30B_A3B,
)
from mtb.llm_benchmarks.models.deepseek import (
    Deepseek_R1_0528_Qwen3_8B,
    Deepseek_R1_Distill_Qwen_7B,
)
from mtb.llm_benchmarks.models.gemma import (
    Gemma3_1B_it,
    Gemma3_1B_it_QAT,
    Gemma3_4B_it,
    Gemma3_4B_it_QAT,
    Gemma3_12B_it_QAT,
    Gemma3_27B_it,
)
from mtb.llm_benchmarks.models.glm import (
    GLM4p7_Flash,
)
from mtb.llm_benchmarks.models.lfm import (
    LFM2_24B_A2B,
    LFM2p5_8B_A1B,
)
from mtb.llm_benchmarks.models.gemma4 import (
    Gemma4_E2B_it,
    Gemma4_E4B_it,
    Gemma4_12B_it,
    Gemma4_26B_A4B_it,
    Gemma4_31B_it,
)
from mtb.llm_benchmarks.models.llama import (
    Llama4_Scout_17B_16E_it,
    Llama3p3_70B_it,
)
from mtb.llm_benchmarks.models.qwen import (
    Qwen2p5_0p5B_it,
    Qwen2p5_3B_it,
    Qwen2p5_Coder_0p5B_it,
    Qwen2p5_Coder_3B_it,
    Qwen3_0p6B_it,
    Qwen3_8B_it,
    Qwen3_14B_it,
    Qwen3_32B_it,
)
from mtb.llm_benchmarks.models.qwen3_coder import (
    Qwen3_Coder_30B_A3B,
)
from mtb.llm_benchmarks.models.phi import (
    Phi4,
    Phi4_Reasoning_Plus,
    Phi4_Mini,
)
from mtb.llm_benchmarks.models.claude import (
    Claude_Opus_4_6,
)
from mtb.llm_benchmarks.models.qwen35 import (
    Qwen3p5_0p8B,
    Qwen3p5_2B,
    Qwen3p5_4B,
    Qwen3p5_9B,
    Qwen3p5_27B,
    Qwen3p5_35B_A3B,
    Qwen3p5_27B_Claude_Opus_Distilled,
)
from mtb.llm_benchmarks.models.qwen36 import (
    Qwen3p6_27B,
    Qwen3p6_35B_A3B,
)

MODEL_SPECS = [
    # deepseek
    Deepseek_R1_Distill_Qwen_7B,
    Deepseek_R1_0528_Qwen3_8B,
    # gemma
    Gemma3_1B_it,
    Gemma3_1B_it_QAT,
    Gemma3_4B_it,
    Gemma3_4B_it_QAT,
    Gemma3_12B_it_QAT,
    # qwen
    Qwen2p5_0p5B_it,
    Qwen2p5_3B_it,
    Qwen2p5_Coder_0p5B_it,
    Qwen2p5_Coder_3B_it,
    Qwen3_0p6B_it,
    Qwen3_8B_it,
    Qwen3_14B_it,
    # nemotron
    Nemotron3_Nano_4B,
    Nemotron_Nano_9B_v2,
    Nemotron_Cascade2_30B_A3B,
    # qwen 3.5
    Qwen3p5_0p8B,
    Qwen3p5_2B,
    Qwen3p5_4B,
    Qwen3p5_9B,
    Qwen3p5_27B,
    Qwen3p5_35B_A3B,
    Qwen3p5_27B_Claude_Opus_Distilled,
    # qwen 3.6
    Qwen3p6_27B,
    Qwen3p6_35B_A3B,
    # agentic / coding — MoE models
    Qwen3_Coder_30B_A3B,
    GLM4p7_Flash,
    LFM2_24B_A2B,
    LFM2p5_8B_A1B,
    # phi
    Phi4_Mini,
    Phi4,
    Phi4_Reasoning_Plus,
    # gemma 4 — fast / on-device
    Gemma4_E2B_it,
    Gemma4_E4B_it,
    # gemma 4 — mid
    Gemma4_12B_it,
    # gemma 4 — reasoning / agentic
    Gemma4_26B_A4B_it,
    Gemma4_31B_it,
    # --- 128GB+ only models ---
    Gemma3_27B_it,
    Qwen3_32B_it,
    Llama4_Scout_17B_16E_it,
    Llama3p3_70B_it,
    # --- API models ---
    Claude_Opus_4_6,
]
