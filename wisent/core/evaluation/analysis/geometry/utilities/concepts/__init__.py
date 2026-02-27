"""Concept analysis, naming, and visualization package."""
from .analysis import (
    detect_multiple_concepts,
    split_by_concepts,
    analyze_concept_independence,
    compute_concept_coherence,
    compute_concept_stability,
    decompose_into_concepts,
    compute_concept_linear_separability,
    get_pair_concept_assignments,
    find_mixed_pairs,
    get_pure_concept_pairs,
    analyze_concept_structure,
    recommend_per_concept_steering,
    detect_with_hdbscan,
    detect_with_coarse_fine_search,
    detect_concepts_multilayer,
)
from .visualization import (
    create_concept_overview_figure,
    create_per_concept_figure,
    create_all_concept_figures,
    create_layer_accuracy_heatmap,
    create_inter_concept_similarity_heatmap,
)
from .naming import (
    name_concepts,
    find_optimal_layer_per_concept,
    decompose_and_name_concepts_with_labels,
    decompose_and_name_concepts,
    format_naming_prompt,
    get_wisent_model,
    call_local_llm,
    parse_llm_response,
    name_concept_with_llm,
    name_all_concepts_with_llm,
)
