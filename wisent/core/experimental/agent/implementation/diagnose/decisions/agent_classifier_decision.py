from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
import re
import asyncio
import time
import sys
import os

# Add the lm-harness-integration path for benchmark selection
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'lm-harness-integration'))

from .classifier_marketplace import ClassifierMarketplace, ClassifierListing, ClassifierCreationEstimate
from ..budget import get_budget_manager, track_task_performance, ResourceType

from wisent.core.utils.config_tools.constants import AGENT_RESOURCE_BUDGET_MINUTES, MAX_BENCHMARKS_SINGLE

from wisent.core.experimental.agent.diagnose._agent_decision_types import (
    TaskAnalysis, ClassifierDecision, SingleClassifierDecision,
    ClassifierParams, SteeringParams, QualityResult, QualityControlledResponse,
)
from wisent.core.experimental.agent.diagnose._agent_decision_creation import ClassifierCreationMixin
from wisent.core.experimental.agent.diagnose._agent_decision_pipeline import ClassifierPipelineMixin

class AgentClassifierDecisionSystem(ClassifierCreationMixin, ClassifierPipelineMixin):
    """
    Intelligent system that helps the agent make autonomous decisions about
    which classifiers to use based on task analysis and cost-benefit considerations.
    """
    
    def __init__(self, marketplace: ClassifierMarketplace):
        self.marketplace = marketplace
        self.decision_history: List[ClassifierDecision] = []
        
    def analyze_task_requirements(self, prompt: str, context: str = "", 
                                 priority: Optional[str] = None, fast_only: bool = False, 
                                 time_budget_minutes: float = AGENT_RESOURCE_BUDGET_MINUTES, max_benchmarks: int = MAX_BENCHMARKS_SINGLE) -> TaskAnalysis:
        """
        Analyze a task/prompt to select relevant benchmarks for training and steering.
        
        Args:
            prompt: The prompt or task to analyze
            context: Additional context about the task
            priority: Priority level for benchmark selection
            fast_only: Only use fast benchmarks
            time_budget_minutes: Time budget for benchmark selection
            max_benchmarks: Maximum number of benchmarks to select
            prefer_fast: Prefer fast benchmarks
            
        Returns:
            Analysis with relevant benchmarks for direct use
        """
        print(f"🔍 Analyzing task requirements for prompt...")
        
        # Get relevant benchmarks for the prompt using priority-aware selection
        existing_model = getattr(self.marketplace, 'model', None)
        relevant_benchmarks = self._get_relevant_benchmarks_for_prompt(
            prompt, 
            existing_model=existing_model,
            priority=priority,
            fast_only=fast_only,
            time_budget_minutes=time_budget_minutes,
            max_benchmarks=max_benchmarks
        )
        print(f"   📊 Found {len(relevant_benchmarks)} relevant benchmarks")
        
        return TaskAnalysis(
            prompt_content=prompt,
            relevant_benchmarks=relevant_benchmarks
        )
    
    def _get_relevant_benchmarks_for_prompt(self, prompt: str, existing_model=None, 
                                           priority: Optional[str] = None, fast_only: bool = False, 
                                           time_budget_minutes: float = AGENT_RESOURCE_BUDGET_MINUTES, max_benchmarks: int = MAX_BENCHMARKS_SINGLE) -> List[Dict[str, Any]]:
        """Get relevant benchmarks for the prompt using the intelligent selection system with priority awareness."""
        try:
            # Import the benchmark selection function from the correct location
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'lm-harness-integration'))
            from populate_tasks import get_relevant_benchmarks_for_prompt
            
            # Use priority-aware selection with provided parameters
            relevant_benchmarks = get_relevant_benchmarks_for_prompt(
                prompt=prompt, 
                max_benchmarks=max_benchmarks, 
                existing_model=existing_model,
                priority=priority,
                fast_only=fast_only,
                time_budget_minutes=time_budget_minutes
            )
            
            return relevant_benchmarks
        except Exception as e:
            print(f"   ⚠️ Failed to get relevant benchmarks: {e}")
            # Fallback to basic high-priority benchmarks
            return [
                {'benchmark': 'mmlu', 'explanation': 'General knowledge benchmark', 'relevance_score': 1, 'priority': 'high', 'loading_time': 9.5},
                {'benchmark': 'truthfulqa_mc1', 'explanation': 'Truthfulness benchmark', 'relevance_score': 2, 'priority': 'high', 'loading_time': 11.2},
                {'benchmark': 'hellaswag', 'explanation': 'Commonsense reasoning benchmark', 'relevance_score': 3, 'priority': 'high', 'loading_time': 12.8}
            ]
    

    

    
