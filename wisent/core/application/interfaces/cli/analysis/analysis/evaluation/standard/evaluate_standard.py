"""Standard benchmark evaluation loop for evaluate-responses command."""

from wisent.core.constants import DISPLAY_TRUNCATION_EVAL


def evaluate_standard(
    args, input_data, responses, task_name,
    evaluation_results, task_results, evaluator, task_docs,
    evaluation_type="generate_until",
):
    """Run standard evaluation loop for benchmark responses."""
    for idx, response_data in enumerate(responses, 1):
        if 'error' in response_data:
            if args.verbose:
                print(f"Question {idx}: Skipped (generation error)")
            evaluation_results.append({
                **response_data,
                "evaluation": {
                    "error": "Generation failed"
                }
            })
            continue

        try:
            generated_response = response_data.get('generated_response', '')
            prompt = response_data.get('prompt', '')

            # First check if positive_reference is available in the response data
            # This is the expected answer that was already extracted during generation
            positive_reference = response_data.get('positive_reference')
            correct_answers = response_data.get('correct_answers')
            incorrect_answers = response_data.get('incorrect_answers')

            # If we have positive_reference, use it directly without needing to match task docs
            if positive_reference is not None:
                # Use the positive_reference as the expected answer
                if args.verbose:
                    print(f"Question {idx}: Using positive_reference as expected answer")

                # Evaluate using selected evaluator
                result = evaluator.evaluate(
                    generated_response,
                    positive_reference,
                    correct_answers=correct_answers,
                    incorrect_answers=incorrect_answers
                )

                is_correct = (result.ground_truth == "TRUTHFUL")

                # Store result
                task_results.append({
                    'acc': 1.0 if is_correct else 0.0,
                    'confidence': result.confidence
                })

                if args.verbose:
                    print(f"Question {idx}:")
                    print(f"   Prompt: {prompt[:DISPLAY_TRUNCATION_EVAL]}...")
                    print(f"   Expected: {str(positive_reference)[:DISPLAY_TRUNCATION_EVAL]}...")
                    print(f"   Generated: {generated_response[:DISPLAY_TRUNCATION_EVAL]}...")
                    print(f"   Ground truth: {result.ground_truth}")
                    print(f"   Confidence: {result.confidence:.3f}")
                    print(f"   Result: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}\n")

                evaluation_results.append({
                    **response_data,
                    "evaluation": {
                        "expected_answer": positive_reference,
                        "ground_truth": result.ground_truth,
                        "confidence": result.confidence,
                        "details": result.details,
                        "correct": is_correct
                    }
                })
                continue

            # Fall back to finding matching task doc by question text
            task_doc = None
            if task_docs:
                for doc in task_docs:
                    doc_question = doc.get('question', '').strip()
                    if doc_question and doc_question in prompt:
                        task_doc = doc
                        break

            if not task_doc:
                if args.verbose:
                    print(f"Question {idx}: Could not match to task doc (no positive_reference available)")
                evaluation_results.append({
                    **response_data,
                    "evaluation": {
                        "error": "Could not match to task document and no positive_reference available"
                    }
                })
                continue

            # Get expected answer based on evaluation type
            if evaluation_type == "multiple_choice":
                # Get all choice texts and gold index
                gold_idx = None
                choice_texts = []

                if 'mc1_targets' in task_doc:
                    # truthfulqa_mc1 format
                    labels = task_doc['mc1_targets']['labels']
                    gold_idx = labels.index(1)
                    choice_texts = task_doc['mc1_targets']['choices']
                elif 'choices' in task_doc:
                    # arc_easy, piqa, etc. format
                    answer_key = task_doc.get('answerKey', 'A')
                    gold_idx = ord(answer_key) - ord('A')
                    if isinstance(task_doc['choices'], dict):
                        choice_texts = task_doc['choices']['text']
                    else:
                        choice_texts = task_doc['choices']
                elif 'gold' in task_doc:
                    # Some tasks have gold directly
                    gold_idx = task_doc['gold']
                    choice_texts = task.doc_to_choice(task_doc)
                else:
                    if args.verbose:
                        print(f"Question {idx}: Unknown multiple-choice format")
                    evaluation_results.append({
                        **response_data,
                        "evaluation": {
                            "error": "Unknown task format"
                        }
                    })
                    continue

                # Use F1Evaluator to match response to best choice
                best_score = 0.0
                best_choice_idx = None

                for i, choice_text in enumerate(choice_texts):
                    result = evaluator.evaluate(generated_response, choice_text)
                    if result.confidence > best_score:
                        best_score = result.confidence
                        best_choice_idx = i

                # Check if correct
                is_correct = (best_choice_idx == gold_idx)

                # Store result
                task_results.append({
                    'acc': 1.0 if is_correct else 0.0,
                    'f1_score': best_score
                })

                if args.verbose:
                    doc_question = task_doc.get('question', '')
                    print(f"Question {idx}:")
                    print(f"   Question: {doc_question[:DISPLAY_TRUNCATION_EVAL]}...")
                    print(f"   Predicted choice: {best_choice_idx} (F1: {best_score:.3f})")
                    print(f"   Correct choice: {gold_idx}")
                    print(f"   Result: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}\n")

                evaluation_results.append({
                    **response_data,
                    "evaluation": {
                        "predicted_choice_idx": best_choice_idx,
                        "predicted_choice_text": choice_texts[best_choice_idx] if best_choice_idx is not None else None,
                        "correct_choice_idx": gold_idx,
                        "correct_choice_text": choice_texts[gold_idx],
                        "f1_score": best_score,
                        "correct": is_correct
                    }
                })

            elif evaluation_type == "generate_until":
                # Get expected answer
                expected = None
                if 'answer' in task_doc:
                    expected = task_doc['answer']
                elif 'answers' in task_doc:
                    expected = task_doc['answers']
                elif 'target' in task_doc:
                    expected = task_doc['target']
                else:
                    if args.verbose:
                        print(f"Question {idx}: No expected answer found")
                    evaluation_results.append({
                        **response_data,
                        "evaluation": {
                            "error": "No expected answer in task document"
                        }
                    })
                    continue

                # Evaluate using selected evaluator
                result = evaluator.evaluate(generated_response, expected)

                is_correct = (result.ground_truth == "TRUTHFUL")

                # Store result
                task_results.append({
                    'acc': 1.0 if is_correct else 0.0,
                    'confidence': result.confidence
                })

                if args.verbose:
                    doc_question = task_doc.get('question', '')
                    print(f"Question {idx}:")
                    print(f"   Question: {doc_question[:DISPLAY_TRUNCATION_EVAL]}...")
                    print(f"   Ground truth: {result.ground_truth}")
                    print(f"   Confidence: {result.confidence:.3f}")
                    print(f"   Result: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}\n")

                evaluation_results.append({
                    **response_data,
                    "evaluation": {
                        "ground_truth": result.ground_truth,
                        "confidence": result.confidence,
                        "details": result.details,
                        "correct": is_correct
                    }
                })

            else:
                # Other evaluation types (loglikelihood_rolling, etc.)
                if args.verbose:
                    print(f"Question {idx}: Evaluation type {evaluation_type} not fully implemented")
                evaluation_results.append({
                    **response_data,
                    "evaluation": {
                        "error": f"Evaluation type {evaluation_type} not implemented"
                    }
                })

        except Exception as e:
            print(f"   ❌ Error evaluating question {idx}: {e}")
            import traceback
            traceback.print_exc()
            evaluation_results.append({
                **response_data,
                "evaluation": {
                    "error": str(e)
                }
            })

