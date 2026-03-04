"""
Classifier Selection System for Autonomous Agent

This module handles:
- Auto-discovery of existing trained classifiers
- Intelligent selection of classifiers based on task requirements
- Performance-based classifier ranking and filtering
- Model-specific classifier matching
"""

import os
import glob
import json
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from ...model_persistence import ModelPersistence
from wisent.core.utils.infra_tools.errors import NoSuitableClassifierError


from wisent.core.experimental.agent.diagnose.classifiers._select_classifiers_helpers import ClassifierSelectorHelpersMixin, auto_select_classifiers_for_agent  # noqa: F401

@dataclass
class ClassifierInfo:
    """Information about a discovered classifier."""
    path: str
    layer: int
    issue_type: str
    threshold: float
    metadata: Dict[str, Any]
    performance_score: float = 0.0


@dataclass
class SelectionCriteria:
    """Criteria for selecting classifiers."""
    required_issue_types: List[str]
    preferred_layers: Optional[List[int]] = None
    min_performance_score: float = 0.0
    max_classifiers: int = None
    model_name: Optional[str] = None
    task_type: Optional[str] = None


class ClassifierSelector(ClassifierSelectorHelpersMixin):
    """Intelligent classifier selection system."""

    def __init__(self, search_paths: List[str] = None, f1_weight: float = None, accuracy_weight: float = None, bonus_sample_denom: float = None, bonus_max: float = None, recency_days: int = None, recency_bonus: float = None):
        """Initialize the classifier selector."""
        if f1_weight is None:
            raise ValueError("f1_weight is required for ClassifierSelector")
        if accuracy_weight is None:
            raise ValueError("accuracy_weight is required for ClassifierSelector")
        if bonus_sample_denom is None:
            raise ValueError("bonus_sample_denom is required for ClassifierSelector")
        if bonus_max is None:
            raise ValueError("bonus_max is required for ClassifierSelector")
        if recency_days is None:
            raise ValueError("recency_days is required for ClassifierSelector")
        if recency_bonus is None:
            raise ValueError("recency_bonus is required for ClassifierSelector")
        self._f1_weight = f1_weight
        self._accuracy_weight = accuracy_weight
        self._bonus_sample_denom = bonus_sample_denom
        self._bonus_max = bonus_max
        self._recency_days = recency_days
        self._recency_bonus = recency_bonus
        self.search_paths = search_paths or [
            "./models",
            "./optimization_results", 
            "./trained_classifiers",
            "./examples/models",
            "."  # Current directory
        ]
        self.discovered_classifiers: List[ClassifierInfo] = []
    
    def discover_classifiers(self) -> List[ClassifierInfo]:
        """
        Auto-discover all available trained classifiers.
        
        Returns:
            List of discovered classifier information
        """
        print("🔍 Discovering available classifiers...")
        
        self.discovered_classifiers = []
        
        for search_path in self.search_paths:
            if not os.path.exists(search_path):
                continue
                
            print(f"   Searching in: {search_path}")
            
            # Search for various classifier file patterns
            patterns = [
                "**/*_classifier.pkl",
                "**/*classifier*.pkl", 
                "**/classifier_layer_*.pkl",
                "**/trained_classifier_*.pkl",
                "**/*_layer_*.pkl"
            ]
            
            for pattern in patterns:
                classifier_files = glob.glob(os.path.join(search_path, pattern), recursive=True)
                
                for filepath in classifier_files:
                    classifier_info = self._analyze_classifier_file(filepath)
                    if classifier_info:
                        self.discovered_classifiers.append(classifier_info)
        
        # Remove duplicates based on path
        unique_classifiers = {}
        for classifier in self.discovered_classifiers:
            unique_classifiers[classifier.path] = classifier
        self.discovered_classifiers = list(unique_classifiers.values())
        
        print(f"   ✅ Discovered {len(self.discovered_classifiers)} classifiers")
        
        # Sort by performance score (highest first)
        self.discovered_classifiers.sort(key=lambda x: x.performance_score, reverse=True)
        
        return self.discovered_classifiers
    
    def _analyze_classifier_file(self, filepath: str) -> Optional[ClassifierInfo]:
        """
        Analyze a classifier file and extract information.
        
        Args:
            filepath: Path to classifier file
            
        Returns:
            ClassifierInfo if valid, None otherwise
        """
        try:
            # Extract layer and issue type from filename
            layer, issue_type = self._parse_classifier_filename(filepath)
            
            # Load metadata if available
            metadata = self._load_classifier_metadata(filepath)
            
            # Calculate performance score
            performance_score = self._calculate_performance_score(metadata)
            
            # Determine threshold
            if 'detection_threshold' not in metadata:
                raise KeyError(f"Classifier metadata at {filepath} missing required 'detection_threshold' key")
            threshold = metadata['detection_threshold']
            
            return ClassifierInfo(
                path=filepath,
                layer=layer,
                issue_type=issue_type,
                threshold=threshold,
                metadata=metadata,
                performance_score=performance_score
            )
            
        except Exception as e:
            print(f"      ⚠️ Failed to analyze {filepath}: {e}")
            return None
    
    def _parse_classifier_filename(self, filepath: str) -> Tuple[int, str]:
        """
        Parse classifier filename to extract layer and issue type.
        
        Args:
            filepath: Path to classifier file
            
        Returns:
            Tuple of (layer, issue_type)
        """
        filename = os.path.basename(filepath)
        
        # Pattern: classifier_layer_X_*.pkl
        if "classifier_layer_" in filename:
            parts = filename.split("_")
            layer_idx = parts.index("layer") + 1 if "layer" in parts else 2
            if layer_idx < len(parts):
                layer = int(parts[layer_idx])
                issue_type = "_".join(parts[:parts.index("layer")])
                return layer, issue_type
        
        # Pattern: trained_classifier_*_layer_X.pkl
        elif "trained_classifier_" in filename and "_layer_" in filename:
            layer_part = filename.split("_layer_")[-1]
            layer = int(layer_part.split(".")[0])
            issue_type = filename.split("trained_classifier_")[1].split("_layer_")[0]
            return layer, issue_type
        
        # Pattern: issue_type_classifier.pkl or issue_type_model_classifier.pkl
        elif "_classifier" in filename:
            parts = filename.replace("_classifier.pkl", "").split("_")
            # Layer not specified in filename - set to None (caller must handle)
            layer = None
            issue_type = "_".join(parts[:-1]) if len(parts) > 1 else parts[0]
            return layer, issue_type
        
        # Fallback: extract from path structure
        else:
            path_parts = Path(filepath).parts
            layer = None
            issue_type = "unknown"
            
            # Look for layer information in path
            for part in path_parts:
                if "layer" in part.lower():
                    try:
                        layer = int(part.split("_")[-1])
                    except:
                        pass
                        
            return layer, issue_type
    
    def _load_classifier_metadata(self, filepath: str) -> Dict[str, Any]:
        """
        Load classifier metadata if available.
        
        Args:
            filepath: Path to classifier file
            
        Returns:
            Metadata dictionary
        """
        metadata = {}
        
        try:
            # Try to load classifier file to get metadata
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                
            if isinstance(data, dict):
                metadata = data.get('metadata', {})
                
        except Exception as e:
            # Skip corrupted files
            print(f"      ⚠️ Skipping corrupted classifier file {filepath}: {e}")
            pass
        
        # Look for associated metadata files
        metadata_paths = [
            filepath.replace('.pkl', '_metadata.json'),
            filepath.replace('.pkl', '.json'),
            os.path.join(os.path.dirname(filepath), 'metadata.json')
        ]
        
        for metadata_path in metadata_paths:
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        file_metadata = json.load(f)
                        metadata.update(file_metadata)
                        break
                except Exception:
                    continue
        
        return metadata
    
    def _calculate_performance_score(self, metadata: Dict[str, Any]) -> float:
        """
        Calculate a performance score for the classifier.
        
        Args:
            metadata: Classifier metadata
            
        Returns:
            Performance score (0.0 to 1.0)
        """
        score = 0.0
        
        # Base score from F1 or accuracy
        f1_score = metadata.get('f1', metadata.get('training_f1', 0.0))
        accuracy = metadata.get('accuracy', metadata.get('training_accuracy', 0.0))
        
        if f1_score > 0:
            score += f1_score * self._f1_weight
        elif accuracy > 0:
            score += accuracy * self._accuracy_weight
        
        # Bonus for larger training sets
        training_samples = metadata.get('training_samples', 0)
        if training_samples > 0:
            sample_bonus = min(training_samples / self._bonus_sample_denom, self._bonus_max)
            score += sample_bonus
        
        # Bonus for recent training
        if 'created_at' in metadata:
            try:
                from datetime import datetime
                created_at = datetime.fromisoformat(metadata['created_at'])
                days_old = (datetime.now() - created_at).days
                if days_old < self._recency_days:
                    score += self._recency_bonus
            except:
                pass
        
        return min(score, 1.0)  # Cap at 1.0
    
