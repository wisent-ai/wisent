from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import os
import json
import pickle
import re
from datetime import datetime
import numpy as np

from wisent.core.utils import resolve_default_device
from wisent.core.utils.config_tools.constants import (
    MARKETPLACE_F1_WEIGHT, MARKETPLACE_ACCURACY_WEIGHT,
    MARKETPLACE_DATA_QUALITY_WEIGHT, MARKETPLACE_DATA_QUALITY_DENOM,
    MARKETPLACE_RECENCY_WEIGHT, BUDGET_RECENCY_WINDOW_DAYS,
    CLASSIFIER_THRESHOLD,
)
from ._marketplace_helpers import (
    ClassifierCreationEstimate,
    MarketplaceEstimationMixin,
)

@dataclass
class ClassifierListing:
    """A classifier available in the marketplace."""
    path: str
    layer: int
    issue_type: str
    threshold: float
    quality_score: float  # 0.0 to 1.0, higher is better
    training_samples: int
    model_family: str
    created_at: str
    training_time_seconds: float
    metadata: Dict[str, Any]

    def to_config(self) -> Dict[str, Any]:
        """Convert to classifier config format."""
        return {
            "path": self.path,
            "layer": self.layer,
            "issue_type": self.issue_type,
            "threshold": self.threshold
        }


class ClassifierMarketplace(MarketplaceEstimationMixin):
    """
    A marketplace interface for classifiers that gives the agent full autonomy
    to discover, evaluate, and create classifiers based on its needs.
    """

    def __init__(self, model, search_paths: List[str] = None, layer: int = None):
        self.model = model
        self.layer = layer
        self.search_paths = search_paths or [
            "./models/",
            "./classifiers/",
            "./wisent/models/",
            "./wisent/classifiers/",
            "./wisent/core/classifiers/"
        ]
        self.available_classifiers: List[ClassifierListing] = []
        self._training_time_cache = {}

    def discover_available_classifiers(self) -> List[ClassifierListing]:
        """
        Discover all available classifiers and return them as marketplace listings.

        Returns:
            List of classifier listings with quality scores and metadata
        """
        print("Discovering available classifiers in marketplace...")

        self.available_classifiers = []

        for search_path in self.search_paths:
            if not os.path.exists(search_path):
                continue

            if "wisent/core/classifiers" in search_path:
                import glob
                pattern = os.path.join(search_path, "**", "*.pkl")
                classifier_files = glob.glob(pattern, recursive=True)
                for filepath in classifier_files:
                    listing = self._create_classifier_listing(filepath)
                    if listing:
                        self.available_classifiers.append(listing)
            else:
                for filename in os.listdir(search_path):
                    if filename.endswith('.pkl'):
                        filepath = os.path.join(search_path, filename)
                        listing = self._create_classifier_listing(filepath)
                        if listing:
                            self.available_classifiers.append(listing)

        self.available_classifiers.sort(key=lambda x: x.quality_score, reverse=True)

        print(f"   Found {len(self.available_classifiers)} classifiers in marketplace")
        return self.available_classifiers

    def _create_classifier_listing(self, filepath: str) -> Optional[ClassifierListing]:
        """Create a marketplace listing for a classifier file."""
        try:
            metadata = self._load_metadata(filepath)
            layer, issue_type = self._parse_filename(filepath)
            quality_score = self._calculate_quality_score(metadata)

            threshold = metadata.get('threshold', CLASSIFIER_THRESHOLD)
            training_samples = metadata.get('training_samples', 0)
            model_family = self._extract_model_family(metadata.get('model_name', ''))
            created_at = metadata.get('created_at', datetime.now().isoformat())
            training_time = metadata.get('training_time_seconds', 0.0)

            return ClassifierListing(
                path=filepath,
                layer=layer,
                issue_type=issue_type,
                threshold=threshold,
                quality_score=quality_score,
                training_samples=training_samples,
                model_family=model_family,
                created_at=created_at,
                training_time_seconds=training_time,
                metadata=metadata
            )

        except Exception as e:
            print(f"   Could not create listing for {filepath}: {e}")
            return None

    def _load_metadata(self, filepath: str) -> Dict[str, Any]:
        """Load metadata for a classifier."""
        json_path = filepath.replace('.pkl', '.json')
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    return json.load(f)
            except:
                pass

        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                if isinstance(data, dict) and 'metadata' in data:
                    return data['metadata']
                elif hasattr(data, 'metadata'):
                    return data.metadata
        except:
            pass

        return {}

    def _parse_filename(self, filepath: str) -> Tuple[int, str]:
        """Parse layer and issue type from filename."""
        filename = os.path.basename(filepath).lower()

        if "wisent/core/classifiers" in filepath:
            path_parts = filepath.split(os.sep)

            if len(path_parts) >= 2:
                benchmark_name = path_parts[-2]
                layer_match = re.search(r'layer_(\d+)\.pkl', filename)
                layer = int(layer_match.group(1)) if layer_match else self.layer
                issue_type = f"quality_{benchmark_name}"
                return layer, issue_type

        filename = os.path.basename(filepath).lower()

        layer = self.layer
        for part in filename.replace('_', ' ').replace('-', ' ').split():
            if part.startswith('l') and part[1:].isdigit():
                layer = int(part[1:])
                break
            elif part.startswith('layer') and len(part) > 5:
                try:
                    layer = int(part[5:])
                    break
                except:
                    pass
            elif 'layer' in filename:
                match = re.search(r'layer[_\s]*(\d+)', filename)
                if match:
                    layer = int(match.group(1))
                    break

        issue_type = self._get_model_issue_type(filename)

        return layer, issue_type

    def _get_model_issue_type(self, filename: str) -> str:
        """Extract issue type from filename using model decisions."""
        prompt = f"""What AI safety issue type is this classifier filename related to?

Filename: {filename}

Common issue types include:
- hallucination (false information, factual errors)
- quality (output quality, coherence)
- harmful (toxic content, safety violations)
- bias (unfairness, discrimination)
- coherence (logical consistency)

Respond with just the issue type (one word):"""

        try:
            response = self.model.generate(prompt, layer_index=self.layer)
            issue_type = response.strip().lower()

            match = re.search(r'(hallucination|quality|harmful|bias|coherence|unknown)', issue_type)
            if match:
                return match.group(1)
            return "unknown"
        except:
            return "unknown"

    def _calculate_quality_score(self, metadata: Dict[str, Any]) -> float:
        """Calculate a comprehensive quality score for the classifier."""
        score = 0.0

        f1_score = metadata.get('f1', metadata.get('training_f1', 0.0))
        accuracy = metadata.get('accuracy', metadata.get('training_accuracy', 0.0))

        if f1_score > 0:
            score += f1_score * MARKETPLACE_F1_WEIGHT
        if accuracy > 0:
            score += accuracy * MARKETPLACE_ACCURACY_WEIGHT

        training_samples = metadata.get('training_samples', 0)
        if training_samples > 0:
            data_quality = min(training_samples / MARKETPLACE_DATA_QUALITY_DENOM, 1.0) * MARKETPLACE_DATA_QUALITY_WEIGHT
            score += data_quality

        try:
            created_at = datetime.fromisoformat(metadata.get('created_at', ''))
            days_old = (datetime.now() - created_at).days
            recency_score = max(0, (BUDGET_RECENCY_WINDOW_DAYS - days_old) / BUDGET_RECENCY_WINDOW_DAYS) * MARKETPLACE_RECENCY_WEIGHT
            score += recency_score
        except:
            pass

        return min(score, 1.0)

    def _extract_model_family(self, model_name: str) -> str:
        """Extract model family from model name using model decisions."""
        if not model_name:
            return "unknown"

        prompt = f"""What model family is this model name from?

Model name: {model_name}

Common families include: llama, mistral, gemma, qwen, gpt, claude, other

Respond with just the family name (one word):"""

        try:
            response = self.model.generate(prompt, layer_index=self.layer)
            family = response.strip().lower()

            match = re.search(r'(llama|mistral|gemma|qwen|gpt|claude|other|unknown)', family)
            if match:
                return match.group(1)
            return "unknown"
        except:
            return "unknown"
