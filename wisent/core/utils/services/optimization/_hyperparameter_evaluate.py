"""Evaluation logic and layer detection for HyperparameterOptimizer."""

import logging
from typing import Dict, List, Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from wisent.core.utils.infra_tools.errors import NoActivationDataError, InsufficientDataError
from wisent.core.utils.config_tools.constants import JSON_INDENT

logger = logging.getLogger(__name__)


def detect_model_layers(model) -> int:
    """Detect the number of layers in a model."""
    try:
        if hasattr(model, 'hf_model'):
            hf_model = model.hf_model
        else:
            hf_model = model
        if hasattr(hf_model, 'config'):
            config = hf_model.config
            layer_attrs = ['num_hidden_layers', 'n_layer', 'num_layers', 'n_layers']
            for attr in layer_attrs:
                if hasattr(config, attr):
                    layer_count = getattr(config, attr)
                    if isinstance(layer_count, int) and layer_count > 0:
                        logger.info(f"Detected {layer_count} layers from config.{attr}")
                        return layer_count
        if hasattr(hf_model, 'model') and hasattr(hf_model.model, 'layers'):
            layer_count = len(hf_model.model.layers)
            logger.info(f"Detected {layer_count} layers from model.layers")
            return layer_count
        elif hasattr(hf_model, 'transformer') and hasattr(hf_model.transformer, 'h'):
            layer_count = len(hf_model.transformer.h)
            logger.info(f"Detected {layer_count} layers from transformer.h")
            return layer_count
        elif hasattr(hf_model, 'encoder') and hasattr(hf_model.encoder, 'layer'):
            layer_count = len(hf_model.encoder.layer)
            logger.info(f"Detected {layer_count} layers from encoder.layer")
            return layer_count
        layer_count = 0
        for name, _ in hf_model.named_modules():
            if any(pattern in name for pattern in ['.layers.', '.h.', '.layer.']):
                for part in name.split('.'):
                    if part.isdigit():
                        layer_num = int(part)
                        layer_count = max(layer_count, layer_num + 1)
        if layer_count > 0:
            logger.info(f"Detected {layer_count} layers from module names")
            return layer_count
        raise ValueError("num_layers must be specified: could not detect layer count from model")
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"num_layers must be specified: error detecting layer count ({e})")


def get_default_layer_range(total_layers: int, use_all: bool = True) -> List[int]:
    """Get a reasonable default layer range for optimization."""
    if use_all:
        return list(range(total_layers))
    else:
        start_layer = max(0, total_layers // 4)
        end_layer = min(total_layers, (3 * total_layers) // 4)
        return list(range(start_layer, end_layer))


class HyperparameterEvaluateMixin:
    """Mixin providing _evaluate_combination, save_results, from_config_dict."""

    def _evaluate_combination(
        self,
        model,
        train_pair_set,
        test_pair_set,
        layer: int,
        aggregation: str,
        prompt_construction_strategy: str,
        token_targeting_strategy: str,
        threshold: float,
        classifier_type: str,
        device: str = None,
        *,
        architecture_module_limit: int,
    ) -> Dict[str, Any]:
        """Evaluate a single hyperparameter combination."""
        import torch
        from .activations.activations_collector import ActivationCollector
        from .activations.extraction_strategy import ExtractionStrategy

        aggregation_map = {
            'average': ExtractionStrategy.CHAT_MEAN,
            'first': ExtractionStrategy.CHAT_FIRST,
            'last': ExtractionStrategy.CHAT_LAST,
            'max': ExtractionStrategy.CHAT_MAX_NORM,
        }
        agg_strategy = aggregation_map.get(aggregation, ExtractionStrategy.CHAT_MEAN)

        prompt_strategy_map = {
            'multiple_choice': ExtractionStrategy.MC_BALANCED,
            'role_playing': ExtractionStrategy.ROLE_PLAY,
            'direct_completion': ExtractionStrategy.CHAT_LAST,
            'instruction_following': ExtractionStrategy.CHAT_LAST,
            'chat_template': ExtractionStrategy.CHAT_LAST,
        }
        prompt_strategy = prompt_strategy_map.get(
            prompt_construction_strategy, ExtractionStrategy.CHAT_LAST
        )

        collector = ActivationCollector(model=model, architecture_module_limit=architecture_module_limit)
        layer_str = str(layer)
        train_pos_acts = []
        train_neg_acts = []

        for pair in train_pair_set.pairs:
            updated_pair = collector.collect_for_pair(
                pair, layers=[layer_str], aggregation=agg_strategy,
                return_full_sequence=False, normalize_layers=False,
                prompt_strategy=prompt_strategy
            )
            if updated_pair.positive_response.layers_activations and layer_str in updated_pair.positive_response.layers_activations:
                act = updated_pair.positive_response.layers_activations[layer_str]
                if act is not None:
                    if isinstance(act, torch.Tensor):
                        train_pos_acts.append(act.detach().cpu().numpy().flatten())
                    else:
                        train_pos_acts.append(np.array(act).flatten())
            if updated_pair.negative_response.layers_activations and layer_str in updated_pair.negative_response.layers_activations:
                act = updated_pair.negative_response.layers_activations[layer_str]
                if act is not None:
                    if isinstance(act, torch.Tensor):
                        train_neg_acts.append(act.detach().cpu().numpy().flatten())
                    else:
                        train_neg_acts.append(np.array(act).flatten())

        if not train_pos_acts or not train_neg_acts:
            raise NoActivationDataError()

        X_train = np.array(train_pos_acts + train_neg_acts)
        y_train = np.array([0] * len(train_pos_acts) + [1] * len(train_neg_acts))

        from .classifiers.models.logistic import LogisticClassifier
        from .classifiers.models.mlp import MLPClassifier

        if classifier_type == 'mlp':
            classifier = MLPClassifier(threshold=threshold, device=device)
        else:
            classifier = LogisticClassifier(threshold=threshold, device=device)
        classifier.fit(X_train, y_train)

        predictions, true_labels, prob_scores = self._collect_test_predictions(
            collector, test_pair_set, layer_str, agg_strategy,
            prompt_strategy, classifier, threshold
        )

        if len(predictions) == 0:
            raise InsufficientDataError(reason="No valid predictions generated")

        accuracy = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions, zero_division=0)
        precision = precision_score(true_labels, predictions, zero_division=0)
        recall = recall_score(true_labels, predictions, zero_division=0)
        try:
            auc = roc_auc_score(true_labels, prob_scores) if len(set(true_labels)) > 1 else 0.0
        except:
            auc = 0.0

        return {
            'layer': layer, 'aggregation': aggregation,
            'prompt_construction_strategy': prompt_construction_strategy,
            'token_targeting_strategy': token_targeting_strategy,
            'threshold': threshold, 'classifier_type': classifier_type,
            'accuracy': accuracy, 'f1': f1, 'precision': precision,
            'recall': recall, 'auc': auc,
            'num_train_samples': len(train_pos_acts) + len(train_neg_acts),
            'num_test_samples': len(predictions)
        }

    def _collect_test_predictions(self, collector, test_pair_set, layer_str,
                                   agg_strategy, prompt_strategy, classifier, threshold):
        """Collect predictions on test set."""
        import torch
        predictions, true_labels, prob_scores = [], [], []

        for pair in test_pair_set.pairs:
            updated_pair = collector.collect_for_pair(
                pair, layers=[layer_str], aggregation=agg_strategy,
                return_full_sequence=False, normalize_layers=False,
                prompt_strategy=prompt_strategy
            )
            pos_act = self._extract_act(updated_pair.positive_response, layer_str)
            neg_act = self._extract_act(updated_pair.negative_response, layer_str)

            if pos_act is not None:
                pos_prob = classifier.predict_proba([pos_act])
                if isinstance(pos_prob, list):
                    pos_prob = pos_prob[0]
                predictions.append(1 if pos_prob > threshold else 0)
                true_labels.append(0)
                prob_scores.append(pos_prob)
            if neg_act is not None:
                neg_prob = classifier.predict_proba([neg_act])
                if isinstance(neg_prob, list):
                    neg_prob = neg_prob[0]
                predictions.append(1 if neg_prob > threshold else 0)
                true_labels.append(1)
                prob_scores.append(neg_prob)

        return predictions, true_labels, prob_scores

    @staticmethod
    def _extract_act(response, layer_str):
        """Extract activation from response for a layer."""
        import torch
        if response.layers_activations and layer_str in response.layers_activations:
            act = response.layers_activations[layer_str]
            if act is not None:
                if isinstance(act, torch.Tensor):
                    return act.detach().cpu().numpy().flatten()
                return np.array(act).flatten()
        return None

    @staticmethod
    def from_config_dict(config_dict: Dict[str, Any]):
        """Create optimizer from configuration dictionary."""
        from .hyperparameter_optimizer import OptimizationConfig, HyperparameterOptimizer
        config = OptimizationConfig(**config_dict)
        return HyperparameterOptimizer(config)

    def save_results(self, result, filepath: str):
        """Save optimization results to file."""
        import json
        result_dict = {
            'best_hyperparameters': {
                'layer': result.best_layer,
                'aggregation': result.best_aggregation,
                'prompt_construction_strategy': result.best_prompt_construction_strategy,
                'token_targeting_strategy': result.best_token_targeting_strategy,
                'threshold': result.best_threshold,
                'classifier_type': result.best_classifier_type
            },
            'best_score': result.best_score,
            'best_metrics': result.best_metrics,
            'optimization_config': {
                'layer_range': self.config.layer_range,
                'aggregation_methods': self.config.aggregation_methods,
                'prompt_construction_strategies': self.config.prompt_construction_strategies,
                'token_targeting_strategies': self.config.token_targeting_strategies,
                'threshold_range': self.config.threshold_range,
                'classifier_types': self.config.classifier_types,
                'metric': self.config.metric
            },
            'all_results': result.all_results
        }
        with open(filepath, 'w') as f:
            json.dump(result_dict, f, indent=JSON_INDENT)
        logger.info(f"Optimization results saved to {filepath}")
