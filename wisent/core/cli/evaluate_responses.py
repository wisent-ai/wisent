"""Evaluate responses command execution logic."""

import json
import os
import sys
from pathlib import Path


def execute_evaluate_responses(args):
    """
    Execute the evaluate-responses command.

    Evaluates generated responses using benchmark-specific evaluators.
    Routes to appropriate evaluator based on task type from task-evaluator.json.
    """
    from lm_eval.tasks import TaskManager
    from wisent.core.evaluators.benchmark_specific import (
        GenerationEvaluator,
        ExactMatchEvaluator,
        F1Evaluator,
        PerplexityEvaluator
    )

    print(f"\n{'='*80}")
    print(f"📊 EVALUATING GENERATED RESPONSES")
    print(f"{'='*80}")
    print(f"   Input: {args.input}")
    print(f"{'='*80}\n")

    # Load input file
    print(f"📂 Loading generated responses...")
    try:
        with open(args.input, 'r') as f:
            input_data = json.load(f)

        # Handle both dict format (with 'responses' key) and direct list format
        if isinstance(input_data, list):
            responses = input_data
            task_name = args.task if args.task else "generic"  # Default to generic for list format
        else:
            responses = input_data.get('responses', [])
            task_name = args.task if args.task else input_data.get('task', "generic")

        if not task_name:
            print(f"   ❌ Task name not found in input file and not provided via --task")
            sys.exit(1)
        print(f"   ✓ Loaded {len(responses)} responses")
        print(f"   Task: {task_name}\n")
    except Exception as e:
        print(f"   ❌ Failed to load input file: {e}")
        sys.exit(1)

    # Load task-evaluator mapping
    print(f"📋 Loading task evaluation config...")
    try:
        # Find task-evaluator.json in project root
        current_dir = Path(__file__).resolve()
        project_root = None
        for parent in current_dir.parents:
            task_eval_file = parent / 'task-evaluator.json'
            if task_eval_file.exists():
                project_root = parent
                break

        if not project_root:
            raise FileNotFoundError("Could not find task-evaluator.json")

        with open(project_root / 'task-evaluator.json', 'r') as f:
            task_evaluator_map = json.load(f)

        # Get task evaluation type
        task_config = task_evaluator_map['tasks'].get(task_name)
        if not task_config:
            raise ValueError(f"Task {task_name} not found in task-evaluator.json")

        evaluation_type = task_config['evaluation_type']
        primary_metric = task_config['primary_metric']

        print(f"   ✓ Task evaluation type: {evaluation_type}")
        print(f"   ✓ Primary metric: {primary_metric}\n")
    except Exception as e:
        print(f"   ❌ Could not load task config: {e}")
        sys.exit(1)

    # Load task to get ground truth (skip for docker_execution and personalization)
    task_docs = None
    task = None
    if evaluation_type not in ["docker_execution", "personalization"]:
        print(f"📚 Loading task data...")
        try:
            tm = TaskManager()
            task_dict = tm.load_task_or_group(task_name)
            task = task_dict[task_name]

            # Get validation docs
            task_docs = list(task.validation_docs())

            print(f"   ✓ Loaded {len(task_docs)} task documents\n")
        except Exception as e:
            print(f"   ❌ Could not load task: {e}")
            sys.exit(1)

    # Select evaluator based on evaluation type
    print(f"🔧 Selecting evaluator for {evaluation_type} task...")
    if evaluation_type == "docker_execution":
        # Handle coding tasks with Docker execution
        from wisent.core.evaluators.benchmark_specific.coding.metrics.evaluator import (
            CodingEvaluator,
            EvaluatorConfig
        )
        from wisent.core.evaluators.benchmark_specific.coding.providers.livecodebench import LiveCodeBenchProvider

        print(f"   Using: CodingEvaluator (Docker sandbox execution)\n")

        # Get Docker config from task config
        docker_config = task_config.get('docker_config', {})
        provider_name = task_config.get('provider', 'livecodebench')

        # This will be handled separately - set evaluator to None for now
        evaluator = None

    elif evaluation_type == "multiple_choice":
        evaluator = F1Evaluator()
        print(f"   Using: F1Evaluator (compares response to choice texts)\n")
    elif evaluation_type == "generate_until":
        if primary_metric == "exact_match":
            evaluator = ExactMatchEvaluator()
            print(f"   Using: ExactMatchEvaluator (extracts and compares answers)\n")
        elif primary_metric in ["em", "f1"]:
            evaluator = F1Evaluator()
            print(f"   Using: F1Evaluator (token-level comparison)\n")
        else:
            evaluator = GenerationEvaluator()
            print(f"   Using: GenerationEvaluator (extracts and compares answers)\n")
    elif evaluation_type == "loglikelihood_rolling":
        evaluator = PerplexityEvaluator()
        print(f"   Using: PerplexityEvaluator (perplexity computation)\n")
    elif evaluation_type == "personalization":
        from wisent.core.evaluators.benchmark_specific import PersonalizationEvaluator
        evaluator = PersonalizationEvaluator()
        print(f"   Using: PersonalizationEvaluator (personality trait evaluation)\n")
    else:
        evaluator = F1Evaluator()
        print(f"   Using: F1Evaluator (default fallback)\n")

    # Evaluate responses
    print(f"🎯 Evaluating responses...\n")
    evaluation_results = []
    task_results = []  # For aggregation

    # Handle docker_execution separately - actual Docker execution
    if evaluation_type == "docker_execution":
        from wisent.core.evaluators.benchmark_specific.coding.providers.livecodebench.provider import LiveCodeBenchProvider
        from wisent.core.evaluators.benchmark_specific.coding.metrics.evaluator import CodingEvaluator, EvaluatorConfig, _make_schema
        from wisent.core.evaluators.benchmark_specific.coding.safe_docker.recipes import RECIPE_REGISTRY
        from wisent.core.evaluators.benchmark_specific.coding.output_sanitizer.python_sanitizer import PythonStandardizer
        from wisent.core.evaluators.benchmark_specific.coding.output_sanitizer.cpp_sanitizer import CppStandardizer
        from wisent.core.evaluators.benchmark_specific.coding.output_sanitizer.java_sanitizer import JavaStandardizer

        _SANITIZERS = {
            "python": PythonStandardizer(),
            "cpp":    CppStandardizer(),
            "java":   JavaStandardizer(),
        }

        print(f"🐳 Running Docker execution evaluation...")

        # Get provider configuration
        provider_name = task_config.get('provider', 'livecodebench')

        # Initialize provider
        if provider_name == 'livecodebench':
            # Extract language from responses or use default
            language = input_data.get('language', 'python')
            release_version = input_data.get('release_version', 'all')

            provider = LiveCodeBenchProvider(
                language=language,
                release_version=release_version,
                limit=None  # Load all problems
            )
            print(f"   ✓ LiveCodeBench provider loaded")
        else:
            print(f"   ❌ Unknown provider: {provider_name}")
            sys.exit(1)

        # Create mapping of problem_id to generated code
        code_map = {}
        for response_data in responses:
            problem_id = response_data.get('problem_id')
            generated_code = response_data.get('generated_response', '')
            if problem_id:
                code_map[problem_id] = generated_code

        print(f"   ✓ Loaded {len(code_map)} generated solutions\n")

        # Track which problems we want to evaluate
        problems_to_evaluate = set(code_map.keys())

        # Create model_fn that returns pre-generated code
        def model_fn(task):
            problem_id = task.options.get('problem_id', '')
            if problem_id in code_map:
                # Return code as solution file
                return {'solution.py': code_map[problem_id]}
            else:
                # No solution available for this problem - return empty
                return None

        # Configure evaluator
        config = EvaluatorConfig(
            image=docker_config.get('image', 'coding/sandbox:polyglot-1.0'),
            time_limit_s=docker_config.get('time_limit_s', 8),
            cpu_limit_s=docker_config.get('cpu_limit_s', 3),
            mem_limit_mb=docker_config.get('mem_limit_mb', 768),
            self_repair=False  # No self-repair for evaluation
        )

        print(f"🐳 Docker Configuration:")
        print(f"   Image: {config.image}")
        print(f"   Time limit: {config.time_limit_s}s")
        print(f"   CPU limit: {config.cpu_limit_s}s")
        print(f"   Memory limit: {config.mem_limit_mb}MB\n")

        # Run evaluation
        evaluator = CodingEvaluator(provider, model_fn, cfg=config)

        print(f"🎯 Executing code in Docker sandbox...\n")

        evaluated_count = 0
        skipped_count = 0

        # Iterate through provider tasks manually to track problem_ids
        for idx, task in enumerate(provider.iter_tasks()):
            problem_id = task.options.get('problem_id', f'unknown_{idx}')

            # Skip problems without solutions
            if problem_id not in problems_to_evaluate:
                skipped_count += 1
                continue

            # Get the generated code
            files = model_fn(task)
            if files is None:
                skipped_count += 1
                continue

            # Merge with task files (test files)
            files = {**task.files, **files}

            # Optionally sanitize
            if config.pre_sanitize:
                schema = _make_schema(task)
                sanitizer = _SANITIZERS.get(task.language)
                if sanitizer:
                    raw = files.get(schema.file_name) or files.get("__raw__")
                    if raw:
                        out = sanitizer.normalize(raw, schema)
                        files = {**files, schema.file_name: out.files.get(schema.file_name, raw)}

            # Run the code in Docker
            recipe = RECIPE_REGISTRY[task.language]
            job = recipe.make_job(**task.options,
                                  time_limit_s=config.time_limit_s,
                                  cpu_limit_s=config.cpu_limit_s,
                                  mem_limit_mb=config.mem_limit_mb)
            result_obj = evaluator.exec.run(files, job)

            evaluated_count += 1

            # Check if passed
            passed = (result_obj.status == "ok")

            result = {
                'problem_id': problem_id,
                'passed': passed,
                'status': result_obj.status,
                'elapsed': result_obj.elapsed
            }

            evaluation_results.append(result)
            task_results.append({
                'pass_rate': 1.0 if passed else 0.0
            })

            if args.verbose:
                status_icon = '✅' if passed else '❌'
                elapsed_time = result_obj.elapsed
                print(f"{status_icon} {problem_id}: {result_obj.status} ({elapsed_time:.2f}s)")
        print(f"\n   ✓ Evaluated {evaluated_count} problems (skipped {skipped_count})")

        # Aggregate results
        aggregated_metrics = {}
        if task_results:
            pass_rate = sum(r['pass_rate'] for r in task_results) / len(task_results)
            aggregated_metrics['pass_rate'] = pass_rate
            aggregated_metrics['total_passed'] = sum(r['pass_rate'] for r in task_results)
            aggregated_metrics['total_problems'] = len(task_results)

        # Save results
        print(f"\n💾 Saving evaluation results...")
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        output_data = {
            "input_file": args.input,
            "task": input_data.get('task'),
            "model": input_data.get('model'),
            "evaluation_type": evaluation_type,
            "evaluator_used": "CodingEvaluator",
            "aggregated_metrics": aggregated_metrics,
            "num_evaluated": len(task_results),
            "num_total": len(responses),
            "evaluations": evaluation_results
        }

        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"   ✓ Results saved to: {args.output}\n")
        print(f"{'='*80}")
        print(f"✅ EVALUATION COMPLETE")
        print(f"{'='*80}")
        print(f"   Total problems: {len(task_results)}")
        print(f"   Passed: {int(aggregated_metrics.get('total_passed', 0))}")
        print(f"   Failed: {len(task_results) - int(aggregated_metrics.get('total_passed', 0))}")
        print(f"   Pass rate: {aggregated_metrics.get('pass_rate', 0):.2%}")
        print(f"{'='*80}\n")
        return

    # Handle personalization separately - LLM-as-judge evaluation
    if evaluation_type == "personalization":
        print(f"🎭 Running personality trait evaluation...")

        # Extract judge model from task config if available
        judge_model = task_config.get('judge_model', evaluator.default_judge_model)
        use_mock = task_config.get('use_mock', False)

        # Check if trait is specified via CLI
        if hasattr(args, 'trait') and args.trait:
            print(f"   Target trait: {args.trait} (from CLI)")

        if use_mock:
            print(f"   ⚠️  Using mock evaluation (no API calls)\n")
        else:
            print(f"   Judge model: {judge_model}\n")

        print(f"   Evaluating {len(responses)} responses...\n")

        evaluated_count = 0
        trait_scores = []

        for idx, response_data in enumerate(responses, 1):
            if 'error' in response_data:
                if args.verbose:
                    print(f"Response {idx}: Skipped (generation error)")
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

                # Extract trait information from CLI argument or response_data
                # CLI argument takes precedence
                if hasattr(args, 'trait') and args.trait:
                    trait = args.trait
                    trait_description = f'The trait: {trait}'
                else:
                    trait = response_data.get('trait', 'unknown')
                    trait_description = response_data.get('trait_description', f'The trait: {trait}')

                    # If trait info is in a nested dict
                    if isinstance(response_data.get('expected'), dict):
                        trait = response_data['expected'].get('trait', trait)
                        trait_description = response_data['expected'].get('trait_description', trait_description)

                # Call evaluator
                eval_result = evaluator.evaluate(
                    response=generated_response,
                    expected={
                        'trait': trait,
                        'trait_description': trait_description
                    },
                    prompt=prompt,
                    judge_model=judge_model,
                    use_mock=use_mock
                )

                evaluated_count += 1

                # Extract metrics from meta
                trait_score = eval_result.meta.get('trait_score', 0)
                intensity = eval_result.meta.get('intensity', 'unknown')

                trait_scores.append(trait_score)

                # Store result
                result = {
                    'response_id': response_data.get('id', idx),
                    'trait': trait,
                    'trait_score': trait_score,
                    'intensity': intensity,
                    'ground_truth': eval_result.ground_truth,
                    'confidence': eval_result.confidence,
                    'explanation': eval_result.meta.get('explanation', ''),
                    'judge_model': eval_result.meta.get('judge_model', judge_model)
                }

                evaluation_results.append(result)
                task_results.append({
                    'trait_score': trait_score,
                    'confidence': eval_result.confidence
                })

                if args.verbose:
                    score_icon = '✅' if trait_score >= 7 else ('⚠️' if trait_score >= 4 else '❌')
                    print(f"{score_icon} Response {idx} ({trait}): {trait_score}/10 ({intensity})")

            except Exception as e:
                logger.exception(f"Error evaluating response {idx}: {e}")
                evaluation_results.append({
                    **response_data,
                    "evaluation": {
                        "error": str(e)
                    }
                })

        print(f"\n   ✓ Evaluated {evaluated_count} responses\n")

        # Aggregate results
        aggregated_metrics = {}
        if task_results:
            avg_trait_score = sum(r['trait_score'] for r in task_results) / len(task_results)
            avg_confidence = sum(r['confidence'] for r in task_results) / len(task_results)

            # Count by intensity thresholds
            strong_count = sum(1 for s in trait_scores if s >= 7)
            moderate_count = sum(1 for s in trait_scores if 4 <= s < 7)
            weak_count = sum(1 for s in trait_scores if s < 4)

            aggregated_metrics['avg_trait_score'] = avg_trait_score
            aggregated_metrics['avg_confidence'] = avg_confidence
            aggregated_metrics['strong_manifestation_rate'] = strong_count / len(trait_scores) if trait_scores else 0
            aggregated_metrics['moderate_manifestation_rate'] = moderate_count / len(trait_scores) if trait_scores else 0
            aggregated_metrics['weak_manifestation_rate'] = weak_count / len(trait_scores) if trait_scores else 0
            aggregated_metrics['total_evaluated'] = len(task_results)

        # Save results
        print(f"💾 Saving evaluation results...")
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        output_data = {
            "input_file": args.input,
            "task": input_data.get('task'),
            "model": input_data.get('model'),
            "evaluation_type": evaluation_type,
            "evaluator_used": "PersonalizationEvaluator",
            "judge_model": judge_model,
            "use_mock": use_mock,
            "aggregated_metrics": aggregated_metrics,
            "num_evaluated": len(task_results),
            "num_total": len(responses),
            "evaluations": evaluation_results
        }

        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"   ✓ Results saved to: {args.output}\n")
        print(f"{'='*80}")
        print(f"✅ PERSONALIZATION EVALUATION COMPLETE")
        print(f"{'='*80}")
        print(f"   Total responses: {len(task_results)}")
        print(f"   Average trait score: {aggregated_metrics.get('avg_trait_score', 0):.2f}/10")
        print(f"   Average confidence: {aggregated_metrics.get('avg_confidence', 0):.2%}")
        print(f"   Strong manifestation: {aggregated_metrics.get('strong_manifestation_rate', 0):.1%}")
        print(f"   Moderate manifestation: {aggregated_metrics.get('moderate_manifestation_rate', 0):.1%}")
        print(f"   Weak manifestation: {aggregated_metrics.get('weak_manifestation_rate', 0):.1%}")
        print(f"{'='*80}\n")
        return

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

            # Find matching task doc by question text
            task_doc = None
            if task_docs:
                for doc in task_docs:
                    doc_question = doc.get('question', '').strip()
                    if doc_question and doc_question in prompt:
                        task_doc = doc
                        break

            if not task_doc:
                if args.verbose:
                    print(f"Question {idx}: Could not match to task doc")
                evaluation_results.append({
                    **response_data,
                    "evaluation": {
                        "error": "Could not match to task document"
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
                    print(f"   Question: {doc_question[:60]}...")
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
                    print(f"   Question: {doc_question[:60]}...")
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

    # Aggregate results
    aggregated_metrics = {}
    if task_results:
        # Get all metric keys
        all_metric_keys = set()
        for result in task_results:
            all_metric_keys.update(result.keys())

        # Aggregate each metric
        for metric_key in all_metric_keys:
            values = [r[metric_key] for r in task_results if metric_key in r]
            if values:
                # Most tasks use mean aggregation
                aggregated_metrics[metric_key] = sum(values) / len(values)

    # Save results
    print(f"\n💾 Saving evaluation results...")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    output_data = {
        "input_file": args.input,
        "task": input_data.get('task'),
        "model": input_data.get('model'),
        "evaluation_type": evaluation_type,
        "evaluator_used": evaluator.name,
        "aggregated_metrics": aggregated_metrics,
        "num_evaluated": len(task_results),
        "num_total": len(responses),
        "evaluations": evaluation_results
    }

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"   ✓ Results saved to: {args.output}\n")

    # Print summary
    print(f"{'='*80}")
    print(f"✅ EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"   Evaluator: {evaluator.name}")
    print(f"   Total responses: {len(responses)}")
    print(f"   Successfully evaluated: {len(task_results)}")
    print(f"   Failed/skipped: {len(responses) - len(task_results)}")
    print(f"\n   Metrics:")
    for metric_name, metric_value in aggregated_metrics.items():
        if isinstance(metric_value, float):
            print(f"     {metric_name}: {metric_value:.4f}")
        else:
            print(f"     {metric_name}: {metric_value}")
    print(f"{'='*80}\n")
