"""Activation extraction and inference strategy implementations."""

from .extraction_strategy import (
    ExtractionStrategy,
    ExtractionComponent,
    tokenizer_has_chat_template,
    extract_activation,
    add_extraction_strategy_args,
    add_extraction_component_args,
)
from .text_builders import build_extraction_texts
from .classifier_inference_strategy import (
    ClassifierInferenceStrategy,
    extract_inference_activation,
    get_inference_score,
    get_recommended_inference_strategy,
    add_classifier_inference_strategy_args,
)

__all__ = [
    'ExtractionStrategy',
    'ExtractionComponent',
    'tokenizer_has_chat_template',
    'build_extraction_texts',
    'extract_activation',
    'add_extraction_strategy_args',
    'add_extraction_component_args',
    'ClassifierInferenceStrategy',
    'extract_inference_activation',
    'get_inference_score',
    'get_recommended_inference_strategy',
    'add_classifier_inference_strategy_args',
]
