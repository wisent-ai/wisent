"""Data models for evaluation results."""
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any


@dataclass
class GenerationResult:
    """Result from a single generation."""
    layer: str
    strength: float
    aggregation_method: str
    question: str
    baseline_response: str
    steered_response: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "layer": self.layer,
            "strength": self.strength,
            "aggregation_method": self.aggregation_method,
            "question": self.question,
            "baseline_response": self.baseline_response,
            "steered_response": self.steered_response,
        }


@dataclass
class EvaluationResult:
    """Result from evaluating a single generation."""
    layer: str
    strength: float
    aggregation_method: str
    question: str
    baseline_response: str
    steered_response: str

    # Metric scores
    differentiation_score: Optional[float] = None
    coherence_score: Optional[float] = None
    trait_alignment_score: Optional[float] = None
    open_traits: Optional[List[str]] = None
    choose_result: Optional[str] = None
    overall_score: Optional[float] = None

    # Raw judge responses
    judge_responses: Dict[str, str] = field(default_factory=dict)

    # Extracted explanations
    explanations: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "layer": self.layer,
            "strength": self.strength,
            "aggregation_method": self.aggregation_method,
            "question": self.question,
            "baseline_response": self.baseline_response,
            "steered_response": self.steered_response,
            "differentiation_score": self.differentiation_score,
            "coherence_score": self.coherence_score,
            "trait_alignment_score": self.trait_alignment_score,
            "open_traits": self.open_traits,
            "choose_result": self.choose_result,
            "overall_score": self.overall_score,
            "judge_responses": self.judge_responses,
            "explanations": self.explanations
        }


@dataclass
class ConfigStatistics:
    """Aggregated statistics for a configuration."""
    layer: str
    strength: float
    aggregation_method: str
    avg_overall_score: Optional[float] = None
    avg_differentiation_score: Optional[float] = None
    avg_coherence_score: Optional[float] = None
    avg_trait_alignment_score: Optional[float] = None
    choose_correct: int = 0
    choose_incorrect: int = 0
    choose_equal: int = 0
    choose_total: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "layer": self.layer,
            "strength": self.strength,
            "aggregation_method": self.aggregation_method,
            "avg_overall_score": self.avg_overall_score,
            "avg_differentiation_score": self.avg_differentiation_score,
            "avg_coherence_score": self.avg_coherence_score,
            "avg_trait_alignment_score": self.avg_trait_alignment_score,
            "choose_correct": self.choose_correct,
            "choose_incorrect": self.choose_incorrect,
            "choose_equal": self.choose_equal,
            "choose_total": self.choose_total
        }
