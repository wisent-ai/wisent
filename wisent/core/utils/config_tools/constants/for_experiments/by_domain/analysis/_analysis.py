"""Statistical analysis, recommendation, and intervention constants."""
from wisent.core.utils.config_tools.constants.for_experiments.by_domain.analysis.infra import *  # noqa: F401,F403

# Default exponential decay factor for the chat_weighted activation extraction
# strategy. Chosen empirically to bias toward earlier answer tokens without
# fully collapsing to chat_first. Matches the convention used in
# cluster_benchmarks_activations.compute_directions_for_strategy.
CHAT_WEIGHTED_DECAY_DEFAULT = 0.1

