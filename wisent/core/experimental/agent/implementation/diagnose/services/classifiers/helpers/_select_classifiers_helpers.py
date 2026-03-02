"""Selection helpers for ClassifierSelector."""

from typing import List, Dict, Any, Optional

from wisent.core.utils.config_tools.constants import CLASSIFIER_MIN_PERFORMANCE_SCORE, DEFAULT_MAX_CLASSIFIERS_SELECT, DISPLAY_TOP_N_TINY


class ClassifierSelectorHelpersMixin:
    """Mixin providing selection methods for ClassifierSelector."""

    def select_classifiers(self, criteria: SelectionCriteria) -> List[Dict[str, Any]]:
        """
        Select the best classifiers based on criteria.
        
        Args:
            criteria: Selection criteria
            
        Returns:
            List of classifier configurations ready for use
        """
        print(f"🎯 Selecting classifiers for: {criteria.required_issue_types}")
        
        # Ensure we've discovered classifiers
        if not self.discovered_classifiers:
            self.discover_classifiers()
        
        selected_classifiers = []
        
        # For each required issue type, find the best classifier
        for issue_type in criteria.required_issue_types:
            best_classifier = self._find_best_classifier_for_issue_type(
                issue_type, criteria
            )
            
            if best_classifier:
                config = {
                    "path": best_classifier.path,
                    "layer": best_classifier.layer,
                    "issue_type": best_classifier.issue_type,
                    "threshold": best_classifier.threshold
                }
                selected_classifiers.append(config)
                print(f"   ✅ Selected for {issue_type}: {os.path.basename(best_classifier.path)} "
                      f"(layer {best_classifier.layer}, score: {best_classifier.performance_score:.3f})")
            else:
                print(f"   ❌ No classifier found for {issue_type}")
                raise NoSuitableClassifierError(issue_type=issue_type)
        
        # Add additional high-performing classifiers if space allows
        if len(selected_classifiers) < criteria.max_classifiers:
            self._add_supplementary_classifiers(selected_classifiers, criteria)
        
        print(f"   📊 Final selection: {len(selected_classifiers)} classifiers")
        return selected_classifiers
    
    def _find_best_classifier_for_issue_type(
        self, 
        issue_type: str, 
        criteria: SelectionCriteria
    ) -> Optional[ClassifierInfo]:
        """
        Find the best classifier for a specific issue type.
        
        Args:
            issue_type: The issue type to find a classifier for
            criteria: Selection criteria
            
        Returns:
            Best matching classifier or None
        """
        candidates = []
        
        for classifier in self.discovered_classifiers:
            # Check if it matches the issue type (exact or partial match)
            if (classifier.issue_type == issue_type or 
                issue_type in classifier.issue_type or
                classifier.issue_type in issue_type):
                
                # Check performance threshold
                if classifier.performance_score >= criteria.min_performance_score:
                    
                    # Check layer preferences
                    if (criteria.preferred_layers is None or 
                        classifier.layer in criteria.preferred_layers):
                        
                        # Check model compatibility
                        if self._is_model_compatible(classifier, criteria.model_name):
                            candidates.append(classifier)
        
        # Return the best candidate (highest performance score)
        return max(candidates, key=lambda x: x.performance_score) if candidates else None
    
    def _is_model_compatible(self, classifier: ClassifierInfo, model_name: Optional[str]) -> bool:
        """
        Check if classifier is compatible with the specified model.
        
        Args:
            classifier: Classifier information
            model_name: Target model name
            
        Returns:
            True if compatible
        """
        if not model_name:
            return True
        
        # Check metadata for model compatibility
        classifier_model = classifier.metadata.get('model_name', '')
        
        if not classifier_model:
            return True  # No model info available, assume compatible
        
        # Extract model family (e.g., "llama", "mistral")
        target_family = self._extract_model_family(model_name)
        classifier_family = self._extract_model_family(classifier_model)
        
        return target_family == classifier_family
    
    def _extract_model_family(self, model_name: str) -> str:
        """Extract model family from model name."""
        model_name = model_name.lower()
        
        if 'llama' in model_name:
            return 'llama'
        elif 'mistral' in model_name:
            return 'mistral'
        elif 'gemma' in model_name:
            return 'gemma'
        elif 'qwen' in model_name:
            return 'qwen'
        else:
            return 'unknown'
    
    def _add_supplementary_classifiers(
        self, 
        selected_classifiers: List[Dict[str, Any]], 
        criteria: SelectionCriteria
    ):
        """
        Add supplementary high-performing classifiers if space allows.
        
        Args:
            selected_classifiers: Currently selected classifiers (modified in place)
            criteria: Selection criteria
        """
        selected_paths = {config["path"] for config in selected_classifiers}
        
        for classifier in self.discovered_classifiers:
            if len(selected_classifiers) >= criteria.max_classifiers:
                break
                
            if (classifier.path not in selected_paths and
                classifier.performance_score >= criteria.min_performance_score):
                
                config = {
                    "path": classifier.path,
                    "layer": classifier.layer,
                    "issue_type": classifier.issue_type,
                    "threshold": classifier.threshold
                }
                selected_classifiers.append(config)
                selected_paths.add(classifier.path)
                print(f"   ➕ Added supplementary: {os.path.basename(classifier.path)} "
                      f"({classifier.issue_type}, score: {classifier.performance_score:.3f})")
    
    def get_classifier_summary(self) -> str:
        """
        Get a summary of discovered classifiers.
        
        Returns:
            Formatted summary string
        """
        if not self.discovered_classifiers:
            return "No classifiers discovered yet. Run discover_classifiers() first."
        
        summary = f"\n📊 Classifier Discovery Summary\n"
        summary += f"{'='*50}\n"
        summary += f"Total Classifiers: {len(self.discovered_classifiers)}\n\n"
        
        # Group by issue type
        by_issue_type = {}
        for classifier in self.discovered_classifiers:
            issue_type = classifier.issue_type
            if issue_type not in by_issue_type:
                by_issue_type[issue_type] = []
            by_issue_type[issue_type].append(classifier)
        
        for issue_type, classifiers in by_issue_type.items():
            summary += f"{issue_type.upper()}: {len(classifiers)} classifiers\n"
            for classifier in sorted(classifiers, key=lambda x: x.performance_score, reverse=True)[:DISPLAY_TOP_N_TINY]:
                summary += f"  • {os.path.basename(classifier.path)} "
                summary += f"(layer {classifier.layer}, score: {classifier.performance_score:.3f})\n"
            summary += "\n"
        
        return summary


def auto_select_classifiers_for_agent(
    model_name: str,
    required_issue_types: List[str] = None,
    search_paths: List[str] = None,
    max_classifiers: int = DEFAULT_MAX_CLASSIFIERS_SELECT,
    min_performance: float = CLASSIFIER_MIN_PERFORMANCE_SCORE
) -> List[Dict[str, Any]]:
    """
    Auto-select the best classifiers for an autonomous agent.
    
    Args:
        model_name: Name of the model being used
        required_issue_types: List of required issue types to detect
        search_paths: Custom search paths for classifiers
        max_classifiers: Maximum number of classifiers to select
        min_performance: Minimum performance score required
        
    Returns:
        List of classifier configurations ready for use
    """
    # Default issue types for comprehensive analysis
    if required_issue_types is None:
        required_issue_types = [
            "hallucination",
            "quality", 
            "harmful",
            "bias"
        ]
    
    selector = ClassifierSelector(search_paths)
    
    criteria = SelectionCriteria(
        required_issue_types=required_issue_types,
        max_classifiers=max_classifiers,
        min_performance_score=min_performance,
        model_name=model_name
    )
    
    return selector.select_classifiers(criteria)
