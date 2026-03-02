"""Docker execution evaluation for evaluate-responses command."""
import json
import os
from wisent.core.utils.config_tools.constants import DEFAULT_SCORE, JSON_INDENT
from wisent.core.utils.core.hardware import eval_time_limit_s, eval_cpu_limit_s, eval_mem_limit_mb


def evaluate_docker_execution(args, input_data, responses, task_name, evaluation_results, task_results, task_config=None):
    """Handle docker_execution evaluation.

    Returns aggregated_metrics dict or None if evaluation should continue.
    """
    from wisent.core.reading.evaluators.benchmark_specific.coding.providers.livecodebench.provider import LiveCodeBenchProvider
    from wisent.core.reading.evaluators.benchmark_specific.coding.metrics.evaluator import CodingEvaluator, EvaluatorConfig, _make_schema
    from wisent.core.reading.evaluators.benchmark_specific.coding.safe_docker.recipes import RECIPE_REGISTRY
    from wisent.core.reading.evaluators.benchmark_specific.coding.output_sanitizer.python_sanitizer import PythonStandardizer
    from wisent.core.reading.evaluators.benchmark_specific.coding.output_sanitizer.cpp_sanitizer import CppStandardizer
    from wisent.core.reading.evaluators.benchmark_specific.coding.output_sanitizer.java_sanitizer import JavaStandardizer

    if task_config is None:
        task_config = {}

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
        raise ValueError(f"Unknown provider: {provider_name}")

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
    docker_config = task_config.get('docker_config', {})
    config = EvaluatorConfig(
        image=docker_config.get('image', 'coding/sandbox:polyglot-1.0'),
        time_limit_s=docker_config.get('time_limit_s', eval_time_limit_s()),
        cpu_limit_s=docker_config.get('cpu_limit_s', eval_cpu_limit_s()),
        mem_limit_mb=docker_config.get('mem_limit_mb', eval_mem_limit_mb()),
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
        "task": task_name if isinstance(input_data, list) else input_data.get('task'),
        "model": None if isinstance(input_data, list) else input_data.get('model'),
        "evaluation_type": "docker_execution",
        "evaluator_used": "CodingEvaluator",
        "aggregated_metrics": aggregated_metrics,
        "num_evaluated": len(task_results),
        "num_total": len(responses),
        "evaluations": evaluation_results
    }

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=JSON_INDENT)

    print(f"   ✓ Results saved to: {args.output}\n")
    print(f"{'='*80}")
    print(f"✅ EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"   Total problems: {len(task_results)}")
    print(f"   Passed: {int(aggregated_metrics.get('total_passed', DEFAULT_SCORE))}")
    print(f"   Failed: {len(task_results) - int(aggregated_metrics.get('total_passed', DEFAULT_SCORE))}")
    print(f"   Pass rate: {aggregated_metrics.get('pass_rate', DEFAULT_SCORE):.2%}")
    print(f"{'='*80}\n")
    return aggregated_metrics
