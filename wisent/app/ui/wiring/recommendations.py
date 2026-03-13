"""Wizard presets and recommendation text for each subgoal."""

RECOMMENDATIONS = {
    "Generate synthetic contrastive pairs from a custom trait": (
        "generate-pairs",
        "Generates contrastive pairs where the model answers the same "
        "prompts with and without a specified trait.",
    ),
    "Generate pairs from an lm-eval benchmark task": (
        "generate-pairs-from-task",
        "Extracts contrastive pairs from an lm-eval harness task.",
    ),
    "Generate model responses to evaluation questions": (
        "generate-responses",
        "Generates model responses, optionally with steering applied.",
    ),
    "Run the full synthetic pipeline end-to-end": (
        "synthetic",
        "Runs the complete synthetic pipeline: prompts, responses, "
        "and enriched pairs.",
    ),
    "From an lm-eval task (full pipeline)": (
        "generate-vector-from-task",
        "Full pipeline: extracts pairs, collects activations, and "
        "trains a steering vector from a benchmark.",
    ),
    "From synthetic contrastive pairs": (
        "generate-vector-from-synthetic",
        "Full pipeline from synthetic data: generates pairs, collects "
        "activations, and trains a steering vector.",
    ),
    "From existing enriched pairs": (
        "create-steering-vector",
        "Creates a steering vector from already-enriched contrastive "
        "pairs with activations.",
    ),
    "Discover the best steering direction automatically": (
        "discover-steering",
        "Searches for the optimal steering direction by trying multiple "
        "methods, layers, and hyperparameters.",
    ),
    "Combine multiple steering vectors at inference": (
        "multi-steer",
        "Applies multiple steering vectors simultaneously during "
        "generation with individual strengths.",
    ),
    "Visualize how steering affects activation space": (
        "steering-viz",
        "Creates PCA/UMAP plots comparing steered vs unsteered "
        "activations.",
    ),
    "Verify steered activations are aligned correctly": (
        "verify-steering",
        "Verifies that a steered model produces activations aligned "
        "with the intended direction.",
    ),
    "Compare steering objects across traits": (
        "compare-steering",
        "Compares multiple steering objects: cosine similarity, overlap, "
        "and interference between directions.",
    ),
    "Evaluate response quality with embedded evaluator": (
        "evaluate-responses",
        "Evaluates generated responses using NLI, embedding similarity, "
        "or other built-in evaluators.",
    ),
    "Evaluate model refusal rate on harmful prompts": (
        "evaluate-refusal",
        "Measures how often a model refuses to answer harmful prompts.",
    ),
    "Run all optimizations at once": (
        "optimize",
        "Runs the complete optimization suite: classification, steering, "
        "and evaluation.",
    ),
    "Optimize classification thresholds": (
        "optimize-classification",
        "Tunes MLP classifier hyperparameters for separating positive "
        "and negative activations.",
    ),
    "Optimize steering parameters (method, layer, strength)": (
        "optimize-steering",
        "Searches for the best method, target layer, and strength via "
        "Bayesian optimization.",
    ),
    "Find optimal training sample size": (
        "optimize-sample-size",
        "Determines the minimum number of contrastive pairs needed.",
    ),
    "Optimize weight modification parameters": (
        "optimize-weights",
        "Optimizes parameters for permanent weight modification.",
    ),
    "Manage cached optimization results": (
        "optimization-cache",
        "View, clear, or export cached optimization results.",
    ),
    "Find the best steering method for a benchmark": (
        "find-best-method",
        "Trials every steering method on a benchmark and ranks them.",
    ),
    "Diagnose contrastive pair quality": (
        "diagnose-pairs",
        "Analyzes pairs for quality issues: semantic divergence, length "
        "imbalance, duplication, and consistency.",
    ),
    "Diagnose steering vector quality": (
        "diagnose-vectors",
        "Analyzes a trained steering vector: separability, stability, "
        "and interference.",
    ),
    "Check if a representation is linear": (
        "check-linearity",
        "Tests whether a concept is represented linearly in model "
        "activations.",
    ),
    "Cluster benchmarks by direction similarity": (
        "cluster-benchmarks",
        "Groups benchmarks whose steering directions are similar.",
    ),
    "Search for unified goodness direction": (
        "geometry-search",
        "Searches for a single direction that improves performance "
        "across multiple benchmarks.",
    ),
    "Run full Zwiad geometry analysis": (
        "zwiad",
        "Runs comprehensive geometry analysis: layer sensitivity, "
        "method comparison, and similarity heatmaps.",
    ),
    "Permanently modify model weights with steering": (
        "modify-weights",
        "Permanently applies steering to model weights and saves "
        "a new model.",
    ),
    "Collect activations from contrastive pairs": (
        "get-activations",
        "Extracts hidden-state activations from a model for each "
        "contrastive pair.",
    ),
    "Train a unified goodness vector from benchmarks": (
        "train-unified-goodness",
        "Trains a single steering vector from multiple benchmark "
        "tasks capturing a general goodness direction.",
    ),
    "View and update inference settings": (
        "inference-config",
        "View or modify inference configuration: temperature, top-p, "
        "max tokens, and generation parameters.",
    ),
    "Run evaluation tasks": (
        "tasks",
        "Lists or runs evaluation tasks from the lm-eval harness.",
    ),
}

PRESETS = (
    (
        "Create a model that is undetectable by AI humanizers",
        "generate-vector-from-synthetic",
        "Generate a steering vector from synthetic pairs that shifts "
        "the model's writing style to be more natural and human-like.",
    ),
    (
        "Steer the model to be left-wing",
        "generate-vector-from-synthetic",
        "Generate a steering vector from synthetic contrastive pairs "
        "that shifts the model's political alignment.",
    ),
    (
        "Analyze the structure of the model's happiness representation",
        "zwiad",
        "Run full Zwiad geometry analysis to understand how happiness "
        "is encoded across the model's layers and dimensions.",
    ),
)
