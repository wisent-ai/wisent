"""
Data loading and processing utilities for comprehensive evaluation.
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import LMEvalHarnessGroundTruth for intelligent evaluation (same approach as CLI)
from ..lm_eval_harness_ground_truth import LMEvalHarnessGroundTruth

# Import task interface for dynamic task loading
from ..task_interface import get_task

logger = logging.getLogger(__name__)


def load_dataset_samples(dataset_name: str, limit: int) -> List[Dict]:
    """Load samples from a dataset using the unified task interface."""
    logger.info(f"Loading {limit} samples from {dataset_name}...")

    try:
        # Use the unified task interface to get any registered task
        task = get_task(dataset_name, limit=limit)
        samples = task.load_data(limit=limit)

        logger.info(f"Loaded {len(samples)} samples from {dataset_name} via {task.__class__.__name__}")
        return samples

    except Exception as e:
        logger.error(f"Failed to load {dataset_name}: {e}")
        # Provide helpful error message with available tasks
        try:
            from ...task_interface import list_tasks

            available_tasks = list_tasks()
            logger.error(f"Available tasks: {available_tasks}")
        except:
            pass
        raise


def extract_activations_with_hook(
    model, tokenizer, texts: List[str], layer: int, batch_size: int, max_length: int, device: torch.device
) -> np.ndarray:
    """Extract activations from a specific layer using hooks."""
    activations = []

    def hook_fn(module, input, output):
        # Handle different output formats (some layers return tuples)
        if isinstance(output, tuple):
            hidden_states = output[0]  # First element is usually hidden states
        else:
            hidden_states = output

        # Extract last token activations (typical for causal LM)
        if len(hidden_states.shape) == 3:  # [batch, seq, hidden]
            last_token_acts = hidden_states[:, -1, :].detach().cpu().numpy()
            activations.extend(last_token_acts)

    # Register hook
    if hasattr(model, "transformer"):  # GPT-style models
        target_layer = model.transformer.h[layer]
    elif hasattr(model, "model"):  # Some other architectures
        target_layer = model.model.layers[layer]
    else:
        raise ValueError("Unknown model architecture")

    handle = target_layer.register_forward_hook(hook_fn)

    try:
        # Process texts in batches
        for i in tqdm(range(0, len(texts), batch_size), desc=f"Extracting activations (layer {layer})"):
            batch_texts = texts[i : i + batch_size]

            inputs = tokenizer(
                batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length
            ).to(device)

            with torch.no_grad():
                _ = model(**inputs)

    finally:
        handle.remove()

    return np.array(activations)


def generate_benchmark_predictions(
    model,
    tokenizer,
    samples: List[Dict],
    batch_size: int,
    max_length: int,
    device: torch.device,
    task_name: str,
    max_new_tokens: int,
) -> Tuple[List[str], List[str]]:
    """Generate model predictions for benchmark evaluation using task extractor with batching."""
    predictions = []
    ground_truths = []

    # Get the task and its extractor
    task = get_task(task_name)
    extractor = task.get_extractor()

    # First, extract all questions and answers
    questions = []
    answers = []

    for sample in samples:
        qa_pair = extractor.extract_qa_pair(sample, task)
        if not qa_pair:
            logger.warning(f"Skipping sample - extractor couldn't extract QA pair: {sample.keys()}")
            continue
        questions.append(qa_pair["formatted_question"])
        answers.append(qa_pair["correct_answer"])

    # Process in batches
    for i in tqdm(range(0, len(questions), batch_size), desc="Generating benchmark predictions"):
        batch_questions = questions[i : i + batch_size]
        batch_answers = answers[i : i + batch_size]

        # Tokenize batch
        inputs = tokenizer(
            batch_questions, return_tensors="pt", padding=True, truncation=True, max_length=max_length
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=True, pad_token_id=tokenizer.eos_token_id
            )

        # Extract generated text for each item in batch
        for j, output in enumerate(outputs):
            input_length = inputs["input_ids"][j].shape[0]
            generated = tokenizer.decode(output[input_length:], skip_special_tokens=True)
            generated = generated.strip()
            predictions.append(generated)

        # Add ground truths
        ground_truths.extend(batch_answers)

    return predictions, ground_truths


def create_probe_training_data(
    model,
    tokenizer,
    samples: List[Dict],
    layer: int,
    batch_size: int,
    max_length: int,
    device: torch.device,
    task_name: str,
    max_new_tokens: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create training data for probes: activations -> correctness labels using task extractor with batched generation."""
    texts = []
    labels = []

    # Get the task and its extractor
    task = get_task(task_name)
    extractor = task.get_extractor()

    # Pre-extract all questions and answers for batched generation
    questions = []
    correct_answers = []

    for sample in samples:
        qa_pair = extractor.extract_qa_pair(sample, task)
        if not qa_pair:
            continue
        questions.append(qa_pair["formatted_question"])
        correct_answers.append(qa_pair["correct_answer"])

    # Generate predictions in batches
    generated_answers = []

    for i in tqdm(range(0, len(questions), batch_size), desc=f"Generating probe data (layer {layer})"):
        batch_questions = questions[i : i + batch_size]

        # Tokenize batch
        inputs = tokenizer(
            batch_questions, return_tensors="pt", padding=True, truncation=True, max_length=max_length
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=True, pad_token_id=tokenizer.eos_token_id
            )

        # Extract generated text for each item in batch
        for j, output in enumerate(outputs):
            input_length = inputs["input_ids"][j].shape[0]
            generated = tokenizer.decode(output[input_length:], skip_special_tokens=True)
            generated = generated.strip()
            generated_answers.append(generated)

    # Now process each question-answer pair for probe training data
    evaluator = LMEvalHarnessGroundTruth(task_name)

    for question, correct_answer, generated in zip(questions, correct_answers, generated_answers):
        # Create examples with model's actual prediction
        correct_text = f"{question} {correct_answer}"
        incorrect_text = f"{question} {generated}"

        texts.extend([correct_text, incorrect_text])

        # Evaluate if prediction is correct using LMEvalHarnessGroundTruth
        try:
            # Create response data format expected by _evaluate_with_lm_eval_metrics
            response_data = [
                {
                    "generated_response": generated,
                    "ground_truth": correct_answer,
                    "question": "evaluation_question",  # Required field for evaluation
                }
            ]

            # Use the same evaluation logic as CLI
            eval_results = evaluator._evaluate_with_lm_eval_metrics(task_name, response_data, None)

            # Extract the result - accuracy > 0 means at least one correct
            is_correct = eval_results.get("accuracy", 0.0) > 0.0

        except Exception as e:
            logger.warning(f"LMEvalHarnessGroundTruth failed, using exact match fallback: {e}")
            is_correct = generated.strip().lower() == correct_answer.strip().lower()

        labels.extend([1, 1 if is_correct else 0])

    # Extract activations
    activations = extract_activations_with_hook(model, tokenizer, texts, layer, batch_size, max_length, device)

    return activations, np.array(labels)


def load_model_and_tokenizer(model_name: str, device: torch.device):
    """Load model and tokenizer with proper configuration."""
    logger.info(f"Loading model {model_name} (ONCE)...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Configure tokenizer for decoder-only models
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Set left padding for decoder-only models (required for correct generation)
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map={"": 0} if device.type == "cuda" else None,
        low_cpu_mem_usage=True,
    )

    model.eval()

    # Log memory usage
    if torch.cuda.is_available():
        memory_gb = torch.cuda.memory_allocated() / 1024**3
        logger.info(f"✓ Model loaded on {device}, GPU memory: {memory_gb:.2f} GB")

    return model, tokenizer


def free_model_memory(model, tokenizer):
    """Free model memory after activation extraction."""
    logger.info("🧹 Freeing model memory...")
    del model
    del tokenizer
    import gc

    gc.collect()
    torch.cuda.empty_cache()

    if torch.cuda.is_available():
        memory_gb = torch.cuda.memory_allocated() / 1024**3
        logger.info(f"GPU memory after cleanup: {memory_gb:.2f} GB")


def get_task_contrastive_pairs(samples: List[Dict], task_name: str) -> List[Dict]:
    """Extract contrastive pairs from samples using the task's extractor."""
    contrastive_pairs = []

    # Get the task and its extractor
    task = get_task(task_name)
    extractor = task.get_extractor()

    for sample in samples:
        # Use the task's extractor to get contrastive pair
        pair = extractor.extract_contrastive_pair(sample, task)
        if pair:
            contrastive_pairs.append(pair)

    logger.info(f"Extracted {len(contrastive_pairs)} contrastive pairs from {len(samples)} samples")
    return contrastive_pairs
