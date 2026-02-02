"""Text generation evaluation for LM-Eval-Harness Ground Truth."""

import logging
from typing import Any, Dict

from wisent.core.activations.activations import Activations
from wisent.core.layer import Layer
from wisent.core.models import get_generate_kwargs

logger = logging.getLogger(__name__)


def evaluate_text_generation(evaluator, classifier, task_name: str, num_samples: int, model, layer: int,
                              token_aggregation: str = "average") -> Dict[str, Any]:
    """Evaluate classifier using text generation approach."""
    try:
        logger.info(f"TEXT GENERATION EVALUATION: {task_name}")
        if evaluator._is_task_interface_task(task_name):
            docs, task_data = evaluator._load_task_interface_data(task_name, num_samples)
        else:
            task_data = model.load_lm_eval_task(task_name, shots=0, limit=num_samples)
            docs, _ = model.split_task_data(task_data, split_ratio=1.0)
        if not docs:
            return evaluator._error_result(f"No documents retrieved from task: {task_name}")
        logger.info(f"Retrieved {len(docs)} documents from {task_name}")
        generated_responses = []
        for i, doc in enumerate(docs):
            try:
                if hasattr(task_data, "doc_to_text"):
                    question = task_data.doc_to_text(doc)
                else:
                    question = str(doc.get("question", doc.get("text", "")))
                logger.debug(f"Generating response for: {question}...")
                gen_kwargs = get_generate_kwargs(max_new_tokens=150, temperature=0.1, do_sample=False)
                generated_response, _ = model.generate(prompt=question, layer_index=layer, **gen_kwargs)
                if task_name.startswith("hle") or task_name in ["math500", "math", "hendrycks_math"]:
                    ground_truth = doc.get("answer", "")
                elif task_name.startswith("aime"):
                    ground_truth = str(doc.get("Answer", "") or doc.get("answer", ""))
                elif task_name == "drop":
                    ground_truth = doc.get("answer", {})
                elif hasattr(task_data, "doc_to_target"):
                    ground_truth = task_data.doc_to_target(doc)
                else:
                    ground_truth = str(doc.get("answer", doc.get("target", "")))
                generated_responses.append({"question": question, "generated_response": generated_response,
                                           "ground_truth": ground_truth, "doc": doc})
                logger.debug(f"Generated: {generated_response}...")
            except Exception as e:
                logger.error(f"Error generating response for doc {i}: {e}")
                continue
        logger.info(f"Evaluating {len(generated_responses)} generated responses using lm-eval metrics...")
        evaluation_results = evaluator.evaluate_with_lm_eval_metrics(task_name, generated_responses, task_data)
        classification_results = []
        for response_data in generated_responses:
            try:
                layer_obj = Layer(index=layer, type="transformer")
                activation_tensor = model.extract_activations(response_data["generated_response"], layer_obj)
                activation_method = evaluator._map_token_aggregation_to_activation_method(token_aggregation)
                activation_obj = Activations(tensor=activation_tensor, layer=layer_obj, aggregation_strategy=activation_method)
                features = activation_obj.extract_features_for_classifier()
                try:
                    features_numpy = features.cpu().numpy()
                    prediction_proba = classifier.predict_proba([features_numpy])
                    if isinstance(prediction_proba, (list, tuple)):
                        prediction = prediction_proba[0] if len(prediction_proba) > 0 else 0.5
                    else:
                        prediction = prediction_proba
                    if hasattr(prediction, "__len__") and not isinstance(prediction, str):
                        prediction = prediction[0] if len(prediction) > 0 else 0.5
                    prediction = float(prediction)
                except Exception as proba_error:
                    logger.warning(f"predict_proba failed: {proba_error}, trying predict...")
                    try:
                        predictions = classifier.predict([features.cpu().numpy()])
                        prediction = float(predictions[0]) if len(predictions) > 0 else 0.5
                    except Exception as predict_error:
                        logger.error(f"Both predict_proba and predict failed: {predict_error}")
                        prediction = 0.5
                classification_results.append({"question": response_data["question"],
                    "generated_response": response_data["generated_response"],
                    "classifier_score": prediction, "classifier_truthful": prediction < 0.5})
            except Exception as e:
                import traceback
                logger.error(f"Error classifying generated response: {e}")
                logger.error(f"Full traceback: {traceback.format_exc()}")
                classification_results.append({"question": response_data["question"],
                    "generated_response": response_data["generated_response"],
                    "classifier_score": 0.5, "classifier_truthful": False, "error": str(e)})
                continue
        return {"ground_truth": "EVALUATED", "method_used": "lm-eval-harness-text-generation",
                "confidence": evaluation_results.get("accuracy", 0.0),
                "details": f"Generated and evaluated {len(generated_responses)} responses using lm-eval metrics",
                "task_name": task_name, "evaluation_method": "text-generation",
                "lm_eval_metrics": evaluation_results, "classification_results": classification_results,
                "total_samples": len(generated_responses)}
    except Exception as e:
        logger.error(f"Error in text generation evaluation: {e}")
        return evaluator._error_result(f"Text generation evaluation error: {e!s}")
