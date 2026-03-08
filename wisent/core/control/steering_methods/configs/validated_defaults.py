"""Empirically validated defaults.

Contains results proven by large-scale empirical studies.
Each entry documents the study that produced it.

Steering method parameters remain empty until per-method
optimization runs produce proven configs. Auto-computed
parameters (sensor_layer, steering_layers) and boolean flags
(normalize, use_caa_init) are handled by inline defaults on
SteeringMethodParameter, not here.
"""

# ------------------------------------------------------------------ #
# Extraction strategy: chat_last                                      #
# ------------------------------------------------------------------ #
# Source: strategy_analysis_results/results.json
# Study: four models x ~two hundred eleven benchmarks x seven
# strategies x all layers. Models: Llama-1B-Instruct, Qwen3-8B,
# Llama2-7b-chat, gpt-oss-20b.
# chat_last ranks first on every metric (linear acc, steering acc,
# signal breadth) across all model x benchmark combinations.
VALIDATED_EXTRACTION_STRATEGY = "chat_last"


# ------------------------------------------------------------------ #
# Steering method defaults (empty until proven)                        #
# ------------------------------------------------------------------ #
VALIDATED_METHOD_DEFAULTS: dict[str, dict] = {}
