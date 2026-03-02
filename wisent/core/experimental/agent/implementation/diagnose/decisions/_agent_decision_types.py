from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
import re
import asyncio
import time
import sys
import os

from wisent.core.utils.config_tools.constants import (
    CLASSIFIER_NUM_EPOCHS, CLASSIFIER_BATCH_SIZE, DEFAULT_CLASSIFIER_LR,
    CLASSIFIER_EARLY_STOPPING_PATIENCE, CLASSIFIER_HIDDEN_DIM,
)

# Add the lm-harness-integration path for benchmark selection
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'lm-harness-integration'))

from .classifier_marketplace import ClassifierMarketplace, ClassifierListing, ClassifierCreationEstimate
from ..budget import get_budget_manager, track_task_performance, ResourceType

@dataclass
class TaskAnalysis:
    """Analysis of what classifiers might be needed for a task."""
    prompt_content: str
    relevant_benchmarks: List[Dict[str, Any]] = None  # Selected benchmarks for training and steering

@dataclass
class ClassifierDecision:
    """A decision about whether to use an existing classifier or create a new one."""
    benchmark_name: str
    action: str  # "use_existing", "create_new", "skip"
    selected_classifier: Optional[ClassifierListing] = None
    creation_estimate: Optional[ClassifierCreationEstimate] = None
    reasoning: str = ""
    confidence: float = 0.0

@dataclass
class SingleClassifierDecision:
    """Decision about creating one combined classifier from multiple benchmarks."""
    benchmark_names: List[str]
    action: str  # "use_existing", "create_new", "skip"
    selected_classifier: Optional[ClassifierListing] = None
    creation_estimate: Optional[ClassifierCreationEstimate] = None
    reasoning: str = ""
    confidence: float = 0.0

@dataclass
class ClassifierParams:
    """Model-determined classifier parameters."""
    optimal_layer: int  # 8-20: Based on semantic complexity needed
    classification_threshold: float  # 0.1-0.9: Based on quality strictness required
    training_samples: int  # 10-50: Based on complexity and time constraints
    classifier_type: str  # logistic/svm/neural: Based on data characteristics
    reasoning: str = ""
    model_name: str = "unknown"  # Model name for matching existing classifiers
    
    # Additional classifier configuration parameters
    aggregation_method: str = "last_token"  # last_token/mean/max for activation aggregation
    token_aggregation: str = "average"  # average/final/first/max/min for token score aggregation
    num_epochs: int = CLASSIFIER_NUM_EPOCHS
    batch_size: int = CLASSIFIER_BATCH_SIZE
    learning_rate: float = DEFAULT_CLASSIFIER_LR
    early_stopping_patience: int = CLASSIFIER_EARLY_STOPPING_PATIENCE
    hidden_dim: int = CLASSIFIER_HIDDEN_DIM

@dataclass
class SteeringParams:
    """Model-determined steering parameters."""
    steering_method: str  # CAA: Contrastive Activation Addition
    initial_strength: float  # 0.1-2.0: How aggressive to start
    increment: float  # 0.1-0.5: How much to increase per failed attempt
    maximum_strength: float  # 0.5-3.0: Upper limit to prevent over-steering
    method_specific_params: Dict[str, Any] = None  # Method-specific parameters
    reasoning: str = ""

@dataclass
class QualityResult:
    """Result of quality evaluation."""
    score: float  # Classifier prediction score
    acceptable: bool  # Model judgment if quality is acceptable
    reasoning: str = ""

@dataclass
class QualityControlledResponse:
    """Final response with complete metadata."""
    response_text: str
    final_quality_score: float
    attempts_needed: int
    classifier_params_used: ClassifierParams
    steering_params_used: Optional[SteeringParams] = None
    quality_progression: List[float] = None  # Quality scores for each attempt
    total_time_seconds: float = 0.0

class AgentClassifierDecisionSystem:
    """
    Intelligent system that helps the agent make autonomous decisions about
    which classifiers to use based on task analysis and cost-benefit considerations.
    """
