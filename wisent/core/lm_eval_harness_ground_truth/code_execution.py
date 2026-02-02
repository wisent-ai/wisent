"""Code execution evaluation for LM-Eval-Harness Ground Truth."""

import logging
from typing import Any, Dict

from wisent.core.activations.activations import Activations
from wisent.core.layer import Layer
from wisent.core.models import get_generate_kwargs
from wisent.core.utils import get_all_docs_from_task, create_deterministic_split

logger = logging.getLogger(__name__)


def evaluate_generic_code_execution(evaluator, classifier, task_name: str, num_samples: int, model, layer: int,
                                     token_aggregation: str = "average") -> Dict[str, Any]:
    """Evaluate generic code execution tasks (non-BigCode) like LiveCodeBench."""
    try:
        logger.info(f"GENERIC CODE EXECUTION EVALUATION: {task_name}")
        from ..secure_code_evaluator import SecureCodeEvaluator
        secure_evaluator = SecureCodeEvaluator()
        task_data = model.load_lm_eval_task(task_name, shots=0, limit=num_samples)
        all_docs, split_counts = get_all_docs_from_task(task_data)
        if all_docs:
            _, docs = create_deterministic_split(all_docs, task_name)
            logger.info(f"Using {len(docs)} test docs from unified split (total: {len(all_docs)}, original splits: {split_counts})")
        else:
            docs, _ = model.split_task_data(task_data, split_ratio=1.0)
        if not docs:
            return evaluator._error_result(f"No documents retrieved from task: {task_name}")
        logger.info(f"Retrieved {len(docs)} documents from {task_name}")
        generated_codes, evaluation_results = [], []
        for i, doc in enumerate(docs):
            try:
                if hasattr(task_data, "doc_to_text"):
                    prompt = task_data.doc_to_text(doc)
                else:
                    question = doc.get("question_content", doc.get("text", ""))
                    starter_code = doc.get("starter_code", "")
                    prompt = f"{question}\n\n{starter_code}" if starter_code else question
                logger.debug(f"Generating code for sample {i + 1}/{len(docs)}...")
                gen_kwargs = get_generate_kwargs(max_new_tokens=500, temperature=0.1, do_sample=False)
                generated_code, _ = model.generate(prompt=prompt, layer_index=layer, **gen_kwargs)
                generated_codes.append(generated_code)
                eval_result = secure_evaluator.evaluate_response(task_name, doc, generated_code)
                evaluation_results.append(eval_result)
                logger.debug(f"Evaluation result: {'PASSED' if eval_result.get('passed', False) else 'FAILED'}")
            except Exception as e:
                logger.error(f"Error processing sample {i}: {e}")
                generated_codes.append("")
                evaluation_results.append({"passed": False, "error": str(e), "success": False})
        total_passed = sum(1 for r in evaluation_results if r.get("passed", False))
        accuracy = total_passed / len(evaluation_results) if evaluation_results else 0.0
        logger.info(f"CODE EXECUTION COMPLETED: {total_passed}/{len(evaluation_results)} passed ({accuracy:.2%})")
        secure_evaluator.cleanup()
        return {"ground_truth": "EVALUATED", "method_used": f"generic-code-execution-{task_name}",
                "confidence": accuracy, "accuracy": accuracy,
                "details": f"Executed and evaluated {len(generated_codes)} code samples",
                "task_name": task_name, "evaluation_method": "code-execution",
                "total_samples": len(generated_codes), "passed_samples": total_passed,
                "evaluation_results": evaluation_results}
    except Exception as e:
        logger.error(f"Error in generic code execution evaluation: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return evaluator._error_result(f"Generic code execution evaluation error: {e!s}")


def evaluate_code_execution(evaluator, classifier, task_name: str, num_samples: int, model, layer: int,
                             token_aggregation: str = "average") -> Dict[str, Any]:
    """Evaluate classifier using code execution approach for BigCode tasks."""
    try:
        logger.debug(f"CODE EXECUTION EVALUATION: {task_name}")
        from ..bigcode_integration import get_bigcode_evaluator, is_bigcode_task, load_bigcode_task
        from ..secure_code_evaluator import SecureCodeEvaluator
        if not is_bigcode_task(task_name):
            if SecureCodeEvaluator.is_code_execution_task(task_name):
                logger.info(f"Task {task_name} is a non-BigCode code execution task")
                return evaluate_generic_code_execution(evaluator, classifier, task_name, num_samples, model, layer, token_aggregation)
            logger.warning(f"Task {task_name} is not a code execution task, falling back to text generation")
            from .text_generation import evaluate_text_generation
            return evaluate_text_generation(evaluator, classifier, task_name, num_samples, model, layer, token_aggregation)
        bigcode_task = load_bigcode_task(task_name, limit=num_samples)
        logger.info(f"Loaded BigCode task {task_name} with {len(bigcode_task)} samples")
        generated_codes = []
        for i, sample in enumerate(bigcode_task.get_samples()):
            try:
                prompt = bigcode_task.doc_to_text(sample)
                logger.debug(f"Generating code for sample {i + 1}/{len(bigcode_task)}...")
                gen_kwargs = get_generate_kwargs(max_new_tokens=300, temperature=0.1, do_sample=False)
                generated_code, _ = model.generate(prompt=prompt, layer_index=layer, **gen_kwargs)
                generated_codes.append(generated_code)
                logger.debug(f"Generated: {generated_code[:100]}...")
            except Exception as e:
                logger.error(f"Error generating code for sample {i}: {e}")
                generated_codes.append("")
        logger.info(f"Evaluating {len(generated_codes)} generated code samples...")
        docker_executor = None
        try:
            from ..docker import OptimizedDockerExecutor
            docker_executor = OptimizedDockerExecutor()
        except Exception as e:
            logger.warning(f"Docker executor not available: {e}")
        evaluator_obj = get_bigcode_evaluator(docker_executor)
        generations_for_eval = [[code] for code in generated_codes]
        evaluation_results = evaluator_obj.evaluate(bigcode_task, generations_for_eval, k_values=[1])
        pass_rate = evaluation_results.get("pass_at_k", {}).get("pass@1", 0.0)
        logger.info(f"Code execution pass@1: {pass_rate:.2%}")
        classification_results = []
        for i, code in enumerate(generated_codes):
            try:
                layer_obj = Layer(index=layer, type="transformer")
                activation_tensor = model.extract_activations(code, layer_obj)
                activation_method = evaluator._map_token_aggregation_to_activation_method(token_aggregation)
                activation_obj = Activations(tensor=activation_tensor, layer=layer_obj, aggregation_strategy=activation_method)
                features = activation_obj.extract_features_for_classifier()
                features_numpy = features.cpu().numpy()
                try:
                    prediction_proba = classifier.predict_proba([features_numpy])
                    if isinstance(prediction_proba, (list, tuple)) and len(prediction_proba) > 0:
                        prediction = float(prediction_proba[0])
                    else:
                        prediction = float(prediction_proba)
                except:
                    predictions = classifier.predict([features_numpy])
                    prediction = float(predictions[0]) if len(predictions) > 0 else 0.5
                code_passed = False
                if i < len(evaluation_results.get("execution_results", [])):
                    sample_results = evaluation_results["execution_results"][i].get("results", [])
                    if sample_results:
                        code_passed = sample_results[0].get("passed", False)
                classification_results.append({"classifier_score": prediction, "code_passed": code_passed, "code_snippet": code[:200]})
            except Exception as e:
                logger.error(f"Error classifying generated code {i}: {e}")
                classification_results.append({"classifier_score": 0.5, "code_passed": False, "error": str(e)})
        correct_predictions = sum(1 for r in classification_results if (r["classifier_score"] > 0.5 and r["code_passed"]) or (r["classifier_score"] <= 0.5 and not r["code_passed"]))
        classifier_accuracy = correct_predictions / len(classification_results) if classification_results else 0.0
        return {"ground_truth": "CODE_EXECUTION", "method_used": "bigcode-evaluation",
                "confidence": classifier_accuracy, "pass_rate": pass_rate,
                "classifier_accuracy": classifier_accuracy, "total_samples": len(generated_codes),
                "passing_samples": int(pass_rate * len(generated_codes)),
                "details": f"Pass@1: {pass_rate:.2%}, Classifier accuracy: {classifier_accuracy:.2%}",
                "task_name": task_name, "evaluation_method": "code-execution",
                "execution_results": evaluation_results}
    except Exception as e:
        logger.error(f"Error in code execution evaluation: {e}")
        import traceback
        traceback.print_exc()
        return {"ground_truth": "ERROR", "method_used": "code-execution-error", "confidence": 0.0,
                "details": f"Code execution evaluation failed: {e!s}",
                "task_name": task_name, "evaluation_method": "code-execution"}
