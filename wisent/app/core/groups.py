"""Command groupings for the Wisent Gradio interface.

Maps all CLI commands into semantic groups for tabbed navigation.
"""

from dataclasses import dataclass, field

_PA = "wisent.core.utils.config_tools.parser_arguments"


@dataclass
class CommandInfo:
    """Metadata for a single CLI command."""

    name: str
    help_text: str
    parser_module: str
    parser_func: str


@dataclass
class CommandGroup:
    """A group of related CLI commands."""

    label: str
    description: str
    commands: list = field(default_factory=list)


def _ci(name, help_text, subpath, func):
    return CommandInfo(name, help_text, f"{_PA}.{subpath}", func)


def get_command_groups() -> list:
    """Return all command groups with their commands."""
    generation = CommandGroup(
        label="Generation",
        description="Generate contrastive pairs, steering vectors, and responses",
        commands=[
            _ci("generate-pairs", "Generate synthetic contrastive pairs",
                "generation.pairs", "setup_generate_pairs_parser"),
            _ci("generate-pairs-from-task", "Generate pairs from lm-eval task",
                "generation.pairs", "setup_generate_pairs_from_task_parser"),
            _ci("generate-vector-from-task", "Generate vector from task (full pipeline)",
                "generation.vectors", "setup_generate_vector_from_task_parser"),
            _ci("generate-vector-from-synthetic", "Generate vector from synthetic pairs",
                "generation.vectors", "setup_generate_vector_from_synthetic_parser"),
            _ci("generate-vector", "Generate steering vectors from pairs",
                "generation.vectors", "setup_generate_vector_parser"),
            _ci("generate-responses", "Generate model responses to task questions",
                "generation.pairs", "setup_generate_responses_parser"),
            _ci("synthetic", "Run synthetic contrastive pair pipeline",
                "training", "setup_synthetic_parser"),
        ],
    )
    steering = CommandGroup(
        label="Steering",
        description="Create, combine, discover, and visualize steering vectors",
        commands=[
            _ci("create-steering-vector", "Create steering vectors from enriched pairs",
                "configuration", "setup_create_steering_object_parser"),
            _ci("multi-steer", "Combine multiple vectors at inference time",
                "other.steering", "setup_multi_steer_parser"),
            _ci("discover-steering", "Discover optimal steering directions",
                "other.steering", "setup_discover_steering_parser"),
            _ci("steering-viz", "Visualize steering effect on activation space",
                "other.steering", "setup_steering_viz_parser"),
            _ci("verify-steering", "Verify steered model activations are aligned",
                "evaluation", "setup_verify_steering_parser"),
            _ci("compare-steering", "Compare steering objects across traits",
                "analysis", "setup_compare_steering_parser"),
        ],
    )
    evaluation = CommandGroup(
        label="Evaluation",
        description="Evaluate responses, refusal, and quality scores",
        commands=[
            _ci("evaluate-responses", "Evaluate responses using embedded evaluator",
                "evaluation", "setup_evaluate_responses_parser"),
            _ci("evaluate-refusal", "Evaluate model refusal rate on harmful prompts",
                "evaluation", "setup_evaluate_refusal_parser"),
        ],
    )
    optimization = CommandGroup(
        label="Optimization",
        description="Optimize classification, steering, weights, and sample sizes",
        commands=[
            _ci("optimize", "Run all optimizations",
                "optimization", "setup_optimize_all_parser"),
            _ci("optimize-classification", "Optimize classification parameters",
                "optimization.steering", "setup_classification_optimizer_parser"),
            _ci("optimize-steering", "Optimize steering parameters",
                "optimization.steering", "setup_steering_optimizer_parser"),
            _ci("optimize-sample-size", "Find optimal training sample size",
                "optimization.steering", "setup_sample_size_optimizer_parser"),
            _ci("optimize-weights", "Optimize weight modification parameters",
                "optimization.weights", "setup_optimize_weights_parser"),
            _ci("optimization-cache", "Manage cached optimization results",
                "optimization.weights", "setup_optimization_cache_parser"),
            _ci("find-best-method", "Find the best steering method for a benchmark",
                "other.steering", "setup_find_best_method_parser"),
        ],
    )
    analysis = CommandGroup(
        label="Analysis",
        description="Diagnostics, geometry, linearity, and clustering",
        commands=[
            _ci("diagnose-pairs", "Diagnose and analyze contrastive pairs",
                "analysis.diagnostics", "setup_diagnose_pairs_parser"),
            _ci("diagnose-vectors", "Diagnose and analyze steering vectors",
                "analysis.diagnostics", "setup_diagnose_vectors_parser"),
            _ci("check-linearity", "Check if a representation is linear",
                "analysis", "setup_check_linearity_parser"),
            _ci("cluster-benchmarks", "Cluster benchmarks by direction similarity",
                "analysis", "setup_cluster_benchmarks_parser"),
            _ci("geometry-search", "Search for unified goodness direction",
                "analysis", "setup_geometry_search_parser"),
            _ci("zwiad", "Run Zwiad geometry analysis",
                "analysis", "setup_zwiad_parser"),
        ],
    )
    config = CommandGroup(
        label="Configuration",
        description="Model configuration, inference settings, and tasks",
        commands=[
            _ci("inference-config", "View and update inference settings",
                "configuration", "setup_inference_config_parser"),
            _ci("tasks", "Run evaluation tasks",
                "other.utilities", "setup_tasks_parser"),
        ],
    )
    weights = CommandGroup(
        label="Weights",
        description="Weight modification and activation collection",
        commands=[
            _ci("modify-weights", "Permanently modify model weights",
                "other", "setup_modify_weights_parser"),
            _ci("get-activations", "Collect activations from contrastive pairs",
                "training", "setup_get_activations_parser"),
            _ci("train-unified-goodness", "Train single goodness vector from benchmarks",
                "training", "setup_train_unified_goodness_parser"),
        ],
    )
    other = CommandGroup(
        label="Agent",
        description="Autonomous agent interaction",
        commands=[
            _ci("agent", "Interact with autonomous agent",
                "other", "setup_agent_parser"),
        ],
    )
    return [generation, steering, evaluation, optimization,
            analysis, config, weights, other]
