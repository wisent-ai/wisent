"""Perplexity evaluation for LM-Eval-Harness Ground Truth."""

import logging
import numpy as np
import torch
from typing import Any, Dict

from wisent.core.activations.activations import Activations
from wisent.core.layer import Layer
from wisent.core.models import get_generate_kwargs

logger = logging.getLogger(__name__)


def calculate_perplexity(model, text: str) -> float:
    """Calculate perplexity of text using the model."""
    try:
        prepared = model.prepare_activations(text)
        outputs, inputs = prepared["outputs"], prepared["inputs"]
        input_ids = inputs["input_ids"]
        logits = outputs.logits
        log_probs = torch.log_softmax(logits, dim=-1)
        if input_ids.shape[1] > 1:
            target_ids = input_ids[0, 1:]
            prediction_logits = log_probs[0, :-1, :]
            token_log_probs = prediction_logits.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
            avg_log_prob = token_log_probs.mean().item()
            perplexity = np.exp(-avg_log_prob)
        else:
            perplexity = float("inf")
        return perplexity
    except Exception as e:
        logger.error(f"Error calculating perplexity: {e}")
        return float("inf")


def evaluate_perplexity(evaluator, classifier, task_name: str, num_samples: int, model, layer: int,
                         token_aggregation: str = "average") -> Dict[str, Any]:
    """Evaluate classifier using perplexity approach."""
    try:
        logger.info(f"PERPLEXITY EVALUATION: {task_name}")
        task_data = model.load_lm_eval_task(task_name, shots=0, limit=num_samples)
        docs, _ = model.split_task_data(task_data, split_ratio=1.0)
        if not docs:
            return evaluator._error_result(f"No documents retrieved from task: {task_name}")
        logger.info(f"Retrieved {len(docs)} documents from {task_name}")
        perplexity_results = []
        for i, doc in enumerate(docs):
            try:
                if task_name == "wikitext":
                    text = doc.get("page", doc.get("text", ""))
                    if not text:
                        continue
                    perplexity = calculate_perplexity(model, text)
                    classification_score = None
                    try:
                        layer_obj = Layer(index=layer, type="transformer")
                        activation_text = text[:1000] if len(text) > 1000 else text
                        activation_tensor = model.extract_activations(activation_text, layer_obj)
                        activation_method = evaluator._map_token_aggregation_to_activation_method(token_aggregation)
                        activation_obj = Activations(tensor=activation_tensor, layer=layer_obj, aggregation_strategy=activation_method)
                        if classifier is not None:
                            features = activation_obj.extract_features_for_classifier()
                            try:
                                prediction_proba = classifier.predict_proba([features.cpu().numpy()])
                                if isinstance(prediction_proba, (list, tuple)) and len(prediction_proba) > 0:
                                    classification_score = float(prediction_proba[0])
                                else:
                                    classification_score = float(prediction_proba)
                                if hasattr(classification_score, "__len__") and not isinstance(classification_score, str):
                                    classification_score = float(classification_score[0])
                            except Exception:
                                predictions = classifier.predict([features.cpu().numpy()])
                                classification_score = float(predictions[0]) if len(predictions) > 0 else 0.5
                        else:
                            classification_score = 0.5
                    except Exception as e:
                        logger.error(f"Error classifying WikiText document: {e}")
                        classification_score = None
                    perplexity_results.append({"document_idx": i, "text_preview": text[:200] + "..." if len(text) > 200 else text,
                        "text_length": len(text), "perplexity": perplexity, "classifier_score": classification_score})
                    continue
                if hasattr(task_data, "doc_to_text"):
                    prompt = task_data.doc_to_text(doc)
                else:
                    prompt = str(doc.get("question", doc.get("text", "")))
                choices = []
                if hasattr(task_data, "doc_to_choice"):
                    choices = [task_data.doc_to_choice(doc, choice_idx) for choice_idx in range(len(doc.get("choices", [])))]
                elif "choices" in doc:
                    choices = doc["choices"]
                else:
                    gen_kwargs = get_generate_kwargs(max_new_tokens=100, temperature=0.1, do_sample=False)
                    generated_response, _ = model.generate(prompt=prompt, layer_index=layer, **gen_kwargs)
                    choices = [generated_response]
                choice_perplexities = []
                for choice_idx, choice in enumerate(choices):
                    try:
                        full_text = f"{prompt} {choice}"
                        perplexity = calculate_perplexity(model, full_text)
                        choice_perplexities.append({"choice_idx": choice_idx, "choice_text": choice, "perplexity": perplexity})
                    except Exception as e:
                        logger.error(f"Error calculating perplexity for choice {choice_idx}: {e}")
                        continue
                ground_truth_idx = None
                if hasattr(task_data, "doc_to_target"):
                    ground_truth = task_data.doc_to_target(doc)
                    try:
                        ground_truth_idx = int(ground_truth)
                    except:
                        ground_truth_idx = None
                elif "answer" in doc:
                    ground_truth_idx = doc["answer"]
                if choice_perplexities:
                    best_choice = min(choice_perplexities, key=lambda x: x["perplexity"])
                    classification_score = None
                    try:
                        layer_obj = Layer(index=layer, type="transformer")
                        activation_tensor = model.extract_activations(best_choice["choice_text"], layer_obj)
                        activation_method = evaluator._map_token_aggregation_to_activation_method(token_aggregation)
                        activation_obj = Activations(tensor=activation_tensor, layer=layer_obj, aggregation_strategy=activation_method)
                        features = activation_obj.extract_features_for_classifier()
                        try:
                            prediction_proba = classifier.predict_proba([features.cpu().numpy()])
                            if isinstance(prediction_proba, (list, tuple)) and len(prediction_proba) > 0:
                                classification_score = prediction_proba[0]
                            else:
                                classification_score = prediction_proba
                            if hasattr(classification_score, "__len__") and not isinstance(classification_score, str):
                                classification_score = classification_score[0] if len(classification_score) > 0 else 0.5
                            classification_score = float(classification_score)
                        except Exception:
                            predictions = classifier.predict([features.cpu().numpy()])
                            classification_score = float(predictions[0]) if len(predictions) > 0 else 0.5
                    except Exception as e:
                        logger.error(f"Error classifying best choice: {e}")
                    perplexity_results.append({"question": prompt, "choices": choice_perplexities,
                        "best_choice_idx": best_choice["choice_idx"], "best_choice_text": best_choice["choice_text"],
                        "best_choice_perplexity": best_choice["perplexity"], "ground_truth_idx": ground_truth_idx,
                        "classifier_score": classification_score,
                        "perplexity_correct": best_choice["choice_idx"] == ground_truth_idx if ground_truth_idx is not None else None})
            except Exception as e:
                logger.error(f"Error processing doc {i}: {e}")
                continue
        total_samples = len(perplexity_results)
        if task_name == "wikitext":
            perplexities = [r["perplexity"] for r in perplexity_results if r["perplexity"] != float("inf")]
            avg_perplexity = sum(perplexities) / len(perplexities) if perplexities else float("inf")
            classifier_scores = [r["classifier_score"] for r in perplexity_results if r["classifier_score"] is not None]
            avg_classifier_score = sum(classifier_scores) / len(classifier_scores) if classifier_scores else None
            perplexity_accuracy = 1.0 if avg_perplexity < 100 else 0.0
            correct_perplexity = sum(1 for r in perplexity_results if r["perplexity"] < 100)
        else:
            correct_perplexity = sum(1 for r in perplexity_results if r.get("perplexity_correct") == True)
            perplexity_accuracy = correct_perplexity / total_samples if total_samples > 0 else 0.0
            classifier_scores = [r["classifier_score"] for r in perplexity_results if r["classifier_score"] is not None]
            avg_classifier_score = sum(classifier_scores) / len(classifier_scores) if classifier_scores else None
        result_dict = {"ground_truth": "EVALUATED", "method_used": "lm-eval-harness-perplexity",
            "confidence": perplexity_accuracy, "details": f"Calculated perplexity for {total_samples} samples",
            "task_name": task_name, "evaluation_method": "perplexity", "perplexity_accuracy": perplexity_accuracy,
            "average_classifier_score": avg_classifier_score, "total_samples": total_samples,
            "correct_perplexity": correct_perplexity, "perplexity_results": perplexity_results[:10]}
        if task_name == "wikitext":
            result_dict["average_perplexity"] = avg_perplexity
            result_dict["details"] = f"Calculated perplexity for {total_samples} WikiText documents, avg perplexity: {avg_perplexity:.3f}"
        return result_dict
    except Exception as e:
        logger.error(f"Error in perplexity evaluation: {e}")
        return evaluator._error_result(f"Perplexity evaluation error: {e!s}")
