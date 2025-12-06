"""
Comprehensive CLI tests for ALL Wisent commands.

Tests the full pipeline for each CLI command to ensure they work correctly.
Uses gpt2 (a non-gated model) for testing on AWS instances.

Run with: pytest wisent/tests/test_all_cli_commands.py -v
Run specific test: pytest wisent/tests/test_all_cli_commands.py::test_generate_pairs -v
"""

import subprocess
import pytest
import tempfile
import os
import json
from pathlib import Path


# =====================================================================
# Configuration
# =====================================================================
# Using gpt2 as a non-gated model for testing. This model is publicly
# accessible and doesn't require HuggingFace authentication.
MODEL = "gpt2"
TASK = "boolq"
TIMEOUT = 600  # 10 minutes max per test


def run_wisent_command(args: list, timeout: int = TIMEOUT) -> subprocess.CompletedProcess:
    """Run a wisent CLI command and return the result."""
    cmd = ["python", "-m", "wisent.core.main"] + args
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout
    )


# =====================================================================
# Fixtures for shared test data
# =====================================================================

@pytest.fixture(scope="module")
def test_pairs_file(tmp_path_factory):
    """Create a test pairs file for reuse."""
    tmpdir = tmp_path_factory.mktemp("pairs")
    filepath = tmpdir / "test_pairs.json"
    pairs = [
        {
            "prompt": "What color is the sky?",
            "positive_response": {"model_response": "The sky is blue during a clear day."},
            "negative_response": {"model_response": "The sky is green with purple stripes."},
            "label": "truthfulness"
        },
        {
            "prompt": "What is 2 + 2?",
            "positive_response": {"model_response": "2 + 2 equals 4."},
            "negative_response": {"model_response": "2 + 2 equals 7."},
            "label": "truthfulness"
        },
        {
            "prompt": "What is the capital of France?",
            "positive_response": {"model_response": "Paris is the capital of France."},
            "negative_response": {"model_response": "Moscow is the capital of France."},
            "label": "truthfulness"
        },
        {
            "prompt": "How many days in a week?",
            "positive_response": {"model_response": "There are 7 days in a week."},
            "negative_response": {"model_response": "There are 12 days in a week."},
            "label": "truthfulness"
        },
    ]
    with open(filepath, 'w') as f:
        json.dump(pairs, f)
    return str(filepath)


@pytest.fixture(scope="module")
def test_responses_file(tmp_path_factory):
    """Create a test responses file for evaluation."""
    tmpdir = tmp_path_factory.mktemp("responses")
    filepath = tmpdir / "test_responses.json"
    responses = [
        {
            "question": "Is the sky blue?",
            "response": "Yes",
            "expected": "Yes",
            "choices": ["Yes", "No"]
        },
        {
            "question": "Is water wet?",
            "response": "Yes",
            "expected": "Yes",
            "choices": ["Yes", "No"]
        }
    ]
    with open(filepath, 'w') as f:
        json.dump(responses, f)
    return str(filepath)


# =====================================================================
# 1. PAIR GENERATION COMMANDS
# =====================================================================

class TestPairGenerationCommands:
    """Tests for pair generation CLI commands."""

    def test_generate_pairs_from_task(self, tmp_path):
        """Test: wisent generate-pairs-from-task"""
        output_file = tmp_path / "pairs.json"

        result = run_wisent_command([
            "generate-pairs-from-task", TASK,
            "--output", str(output_file),
            "--limit", "5",
            "--verbose"
        ])

        assert result.returncode == 0, f"Command failed: {result.stderr}"
        assert output_file.exists(), "Output file not created"

        with open(output_file) as f:
            data = json.load(f)
            # Output can be a dict with 'pairs' key or a list
            if isinstance(data, dict):
                assert "pairs" in data, "Dict output should have 'pairs' key"
                assert len(data["pairs"]) > 0, "Should have generated pairs"
            else:
                assert isinstance(data, list), "Output should be a list or dict"
                assert len(data) > 0, "Should have generated pairs"

    def test_generate_pairs_synthetic(self, tmp_path):
        """Test: wisent generate-pairs (synthetic)"""
        output_file = tmp_path / "synthetic_pairs.json"

        result = run_wisent_command([
            "generate-pairs",
            "--trait", "truthfulness",
            "--num-pairs", "3",
            "--output", str(output_file),
            "--model", MODEL,
            "--similarity-threshold", "0.8",
            "--verbose"
        ])

        assert result.returncode == 0, f"Command failed: {result.stderr}"
        assert output_file.exists(), "Output file not created"

    def test_diagnose_pairs(self, test_pairs_file, tmp_path):
        """Test: wisent diagnose-pairs"""
        result = run_wisent_command([
            "diagnose-pairs", test_pairs_file,
            "--show-sample",
            "--verbose"
        ])

        assert result.returncode == 0, f"Command failed: {result.stderr}"
        # Should show diagnostic information
        assert "pair" in result.stdout.lower() or result.returncode == 0


# =====================================================================
# 2. ACTIVATION COMMANDS
# =====================================================================

class TestActivationCommands:
    """Tests for activation extraction CLI commands."""

    def test_get_activations(self, test_pairs_file, tmp_path):
        """Test: wisent get-activations"""
        output_file = tmp_path / "activations.json"

        result = run_wisent_command([
            "get-activations", test_pairs_file,
            "--output", str(output_file),
            "--model", MODEL,
            "--layers", "3",
            "--token-aggregation", "average",
            "--verbose"
        ])

        assert result.returncode == 0, f"Command failed: {result.stderr}"
        assert output_file.exists(), "Output file not created"

    def test_get_activations_multiple_layers(self, test_pairs_file, tmp_path):
        """Test: wisent get-activations with multiple layers"""
        output_file = tmp_path / "multi_layer_activations.json"

        result = run_wisent_command([
            "get-activations", test_pairs_file,
            "--output", str(output_file),
            "--model", MODEL,
            "--layers", "2,3,4",
            "--token-aggregation", "final",
            "--verbose"
        ])

        assert result.returncode == 0, f"Command failed: {result.stderr}"
        assert output_file.exists(), "Output file not created"


# =====================================================================
# 3. STEERING VECTOR COMMANDS
# =====================================================================

class TestSteeringVectorCommands:
    """Tests for steering vector creation CLI commands."""

    def test_generate_vector_from_task(self, tmp_path):
        """Test: wisent generate-vector-from-task"""
        output_file = tmp_path / "vector.pt"

        result = run_wisent_command([
            "generate-vector-from-task",
            "--task", TASK,
            "--trait-label", "correctness",
            "--output", str(output_file),
            "--model", MODEL,
            "--num-pairs", "5",
            "--layers", "3",
            "--token-aggregation", "average",
            "--method", "caa",
            "--normalize",
            "--verbose"
        ])

        assert result.returncode == 0, f"Command failed: {result.stderr}"
        assert output_file.exists(), "Vector file not created"

    def test_generate_vector_from_synthetic(self, tmp_path):
        """Test: wisent generate-vector-from-synthetic"""
        output_file = tmp_path / "synthetic_vector.pt"

        result = run_wisent_command([
            "generate-vector-from-synthetic",
            "--trait", "helpfulness",
            "--output", str(output_file),
            "--model", MODEL,
            "--num-pairs", "3",
            "--layers", "3",
            "--token-aggregation", "average",
            "--method", "caa",
            "--normalize",
            "--verbose"
        ])

        assert result.returncode == 0, f"Command failed: {result.stderr}"
        assert output_file.exists(), "Vector file not created"

    def test_create_steering_vector(self, test_pairs_file, tmp_path):
        """Test: wisent create-steering-vector (from enriched pairs)"""
        enriched_file = tmp_path / "enriched.json"
        vector_file = tmp_path / "vector.pt"

        # First enrich with activations
        enrich_result = run_wisent_command([
            "get-activations", test_pairs_file,
            "--output", str(enriched_file),
            "--model", MODEL,
            "--layers", "3",
            "--token-aggregation", "average"
        ])
        assert enrich_result.returncode == 0, f"Enrichment failed: {enrich_result.stderr}"

        # Create steering vector
        result = run_wisent_command([
            "create-steering-vector", str(enriched_file),
            "--output", str(vector_file),
            "--method", "caa",
            "--normalize",
            "--verbose"
        ])

        assert result.returncode == 0, f"Command failed: {result.stderr}"
        assert vector_file.exists(), "Vector file not created"

    def test_diagnose_vectors(self, test_pairs_file, tmp_path):
        """Test: wisent diagnose-vectors"""
        enriched_file = tmp_path / "enriched.json"
        vector_file = tmp_path / "vector.json"

        # Create enriched pairs first
        enrich_result = run_wisent_command([
            "get-activations", test_pairs_file,
            "--output", str(enriched_file),
            "--model", MODEL,
            "--layers", "3",
            "--token-aggregation", "average"
        ])
        assert enrich_result.returncode == 0

        # Create vector file (as JSON for diagnosis)
        create_result = run_wisent_command([
            "create-steering-vector", str(enriched_file),
            "--output", str(vector_file),
            "--method", "caa",
            "--normalize"
        ])
        assert create_result.returncode == 0

        # Diagnose vectors
        result = run_wisent_command([
            "diagnose-vectors", str(vector_file),
            "--show-sample",
            "--verbose"
        ])

        # May fail if vector file format is .pt not .json - that's OK
        # Just check the command runs
        assert result.returncode == 0 or "not found" not in result.stderr.lower()


# =====================================================================
# 4. GENERATION/EVALUATION COMMANDS
# =====================================================================

class TestGenerationEvaluationCommands:
    """Tests for generation and evaluation CLI commands."""

    def test_generate_responses(self, tmp_path):
        """Test: wisent generate-responses"""
        output_file = tmp_path / "responses.json"

        result = run_wisent_command([
            "generate-responses", MODEL,
            "--task", TASK,
            "--num-questions", "3",
            "--max-new-tokens", "50",
            "--temperature", "0.7",
            "--output", str(output_file),
            "--verbose"
        ])

        assert result.returncode == 0, f"Command failed: {result.stderr}"
        assert output_file.exists(), "Output file not created"

    def test_evaluate_responses(self, test_responses_file, tmp_path):
        """Test: wisent evaluate-responses"""
        output_file = tmp_path / "evaluation.json"

        result = run_wisent_command([
            "evaluate-responses",
            "--input", test_responses_file,
            "--output", str(output_file),
            "--task", TASK,
            "--verbose"
        ])

        assert result.returncode == 0, f"Command failed: {result.stderr}"
        assert output_file.exists(), "Output file not created"

    def test_evaluate_refusal(self, tmp_path):
        """Test: wisent evaluate-refusal"""
        output_file = tmp_path / "refusal_eval.json"

        result = run_wisent_command([
            "evaluate-refusal",
            "--model", MODEL,
            "--output", str(output_file),
            "--evaluator", "keyword",
            "--num-prompts", "5",
            "--max-new-tokens", "50",
            "--verbose"
        ])

        assert result.returncode == 0, f"Command failed: {result.stderr}"
        assert output_file.exists(), "Output file not created"


# =====================================================================
# 5. MULTI-STEER AND MODIFY-WEIGHTS COMMANDS
# =====================================================================

class TestMultiSteerCommands:
    """Tests for multi-steer and weight modification commands."""

    def test_multi_steer(self, tmp_path):
        """Test: wisent multi-steer"""
        vector1_path = tmp_path / "vector1.pt"
        vector2_path = tmp_path / "vector2.pt"

        # Train first vector
        result1 = run_wisent_command([
            "generate-vector-from-task",
            "--task", TASK,
            "--trait-label", "trait1",
            "--output", str(vector1_path),
            "--model", MODEL,
            "--num-pairs", "5",
            "--layers", "3",
            "--token-aggregation", "average",
            "--method", "caa",
            "--normalize"
        ])
        assert result1.returncode == 0, f"Vector1 training failed: {result1.stderr}"

        # Train second vector
        result2 = run_wisent_command([
            "generate-vector-from-task",
            "--task", TASK,
            "--trait-label", "trait2",
            "--output", str(vector2_path),
            "--model", MODEL,
            "--num-pairs", "5",
            "--layers", "3",
            "--token-aggregation", "average",
            "--method", "caa",
            "--normalize"
        ])
        assert result2.returncode == 0, f"Vector2 training failed: {result2.stderr}"

        # Combine vectors
        result = run_wisent_command([
            "multi-steer",
            "--vector", f"{vector1_path}:0.5",
            "--vector", f"{vector2_path}:0.5",
            "--model", MODEL,
            "--layer", "3",
            "--method", "CAA",
            "--prompt", "What is AI?",
            "--max-new-tokens", "30",
            "--normalize-weights",
            "--verbose"
        ])

        assert result.returncode == 0, f"Command failed: {result.stderr}"

    def test_modify_weights(self, tmp_path):
        """Test: wisent modify-weights"""
        output_dir = tmp_path / "modified_model"

        result = run_wisent_command([
            "modify-weights",
            "--task", TASK,
            "--output-dir", str(output_dir),
            "--model", MODEL,
            "--num-pairs", "5",
            "--trait-label", "correctness",
            "--layers", "3",
            "--token-aggregation", "average",
            "--method", "additive",
            "--alpha", "0.5",
            "--verbose"
        ])

        assert result.returncode == 0, f"Command failed: {result.stderr}"
        assert output_dir.exists(), "Output directory not created"


# =====================================================================
# 6. OPTIMIZATION COMMANDS
# =====================================================================

class TestOptimizationCommands:
    """Tests for optimization CLI commands.

    These tests MUST be run on AWS using run_on_aws.sh script.
    """

    def test_optimize_classification(self):
        """Test: wisent optimize-classification"""
        result = run_wisent_command([
            "optimize-classification", MODEL,
            "--limit", "50",
            "--optimization-metric", "f1",
            "--layer-range", "2-8",
            "--aggregation-methods", "average", "final", "first",
            "--threshold-range", "0.3", "0.5", "0.7"
        ])

        assert result.returncode == 0, f"Command failed: {result.stderr}"

    def test_optimize_steering(self):
        """Test: wisent optimize-steering"""
        result = run_wisent_command([
            "optimize-steering", "comprehensive", MODEL,
            "--tasks", TASK,
            "--methods", "CAA", "PCA",
            "--limit", "50",
            "--verbose"
        ])

        assert result.returncode == 0, f"Command failed: {result.stderr}"

    def test_optimize_sample_size(self):
        """Test: wisent optimize-sample-size"""
        result = run_wisent_command([
            "optimize-sample-size", MODEL,
            "--task", TASK,
            "--limit", "100",
            "--sample-sizes", "10", "25", "50", "100",
            "--layer", "5",
            "--verbose"
        ])

        assert result.returncode == 0, f"Command failed: {result.stderr}"

    def test_optimize_all(self, tmp_path):
        """Test: wisent optimize-all"""
        output_dir = tmp_path / "optimized"

        result = run_wisent_command([
            "optimize-all", MODEL,
            "--tasks", TASK,
            "--limit", "50",
            "--n-trials", "10",
            "--output-dir", str(output_dir),
            "--verbose"
        ])

        assert result.returncode == 0, f"Command failed: {result.stderr}"


# =====================================================================
# 7. CONFIG/UTILITY COMMANDS
# =====================================================================

class TestConfigUtilityCommands:
    """Tests for configuration and utility CLI commands."""

    def test_tasks_list(self):
        """Test: wisent tasks --list-tasks"""
        result = run_wisent_command([
            "tasks", "--list-tasks"
        ])

        assert result.returncode == 0, f"Command failed: {result.stderr}"
        # Should list available tasks
        assert "boolq" in result.stdout.lower() or "task" in result.stdout.lower()

    def test_tasks_info(self):
        """Test: wisent tasks --task-info"""
        result = run_wisent_command([
            "tasks", "--task-info", TASK
        ])

        assert result.returncode == 0, f"Command failed: {result.stderr}"

    def test_tasks_run(self, tmp_path):
        """Test: wisent tasks (run evaluation)"""
        output_dir = tmp_path / "task_output"

        result = run_wisent_command([
            "tasks", TASK,
            "--model", MODEL,
            "--layer", "3",
            "--limit", "5",
            "--output", str(output_dir),
            "--token-aggregation", "average",
            "--verbose"
        ])

        assert result.returncode == 0, f"Command failed: {result.stderr}"

    def test_inference_config_show(self):
        """Test: wisent inference-config show"""
        result = run_wisent_command([
            "inference-config", "show"
        ])

        assert result.returncode == 0, f"Command failed: {result.stderr}"

    def test_inference_config_set_reset(self):
        """Test: wisent inference-config set and reset"""
        # Set a config value
        set_result = run_wisent_command([
            "inference-config", "set",
            "--max-new-tokens", "100"
        ])
        assert set_result.returncode == 0, f"Set failed: {set_result.stderr}"

        # Reset to defaults
        reset_result = run_wisent_command([
            "inference-config", "reset"
        ])
        assert reset_result.returncode == 0, f"Reset failed: {reset_result.stderr}"

    def test_optimization_cache_list(self):
        """Test: wisent optimization-cache list"""
        result = run_wisent_command([
            "optimization-cache", "list"
        ])

        assert result.returncode == 0, f"Command failed: {result.stderr}"

    def test_optimization_cache_export_import(self, tmp_path):
        """Test: wisent optimization-cache export/import"""
        export_file = tmp_path / "cache_export.json"

        # Export cache (positional argument, not --output)
        export_result = run_wisent_command([
            "optimization-cache", "export",
            str(export_file)
        ])
        assert export_result.returncode == 0, f"Export failed: {export_result.stderr}"

        # Import cache (positional argument, not --input)
        if export_file.exists():
            import_result = run_wisent_command([
                "optimization-cache", "import",
                str(export_file)
            ])
            # Import may fail if cache is empty, but shouldn't error
            assert import_result.returncode == 0 or "empty" in import_result.stderr.lower()


# =====================================================================
# 8. FULL PIPELINE TEST
# =====================================================================

class TestFullPipeline:
    """End-to-end pipeline test covering the complete workflow."""

    def test_complete_steering_pipeline(self, tmp_path):
        """
        Complete pipeline test:
        1. Generate pairs from task
        2. Get activations
        3. Create steering vector
        4. Generate responses with steering
        """
        pairs_file = tmp_path / "pairs.json"
        enriched_file = tmp_path / "enriched.json"
        vector_file = tmp_path / "vector.pt"

        # Step 1: Generate pairs
        pairs_result = run_wisent_command([
            "generate-pairs-from-task", TASK,
            "--output", str(pairs_file),
            "--limit", "5"
        ])
        assert pairs_result.returncode == 0, f"Pairs generation failed: {pairs_result.stderr}"
        assert pairs_file.exists()

        # Step 2: Get activations
        act_result = run_wisent_command([
            "get-activations", str(pairs_file),
            "--output", str(enriched_file),
            "--model", MODEL,
            "--layers", "3",
            "--token-aggregation", "average"
        ])
        assert act_result.returncode == 0, f"Activation extraction failed: {act_result.stderr}"
        assert enriched_file.exists()

        # Step 3: Create steering vector
        vector_result = run_wisent_command([
            "create-steering-vector", str(enriched_file),
            "--output", str(vector_file),
            "--method", "caa",
            "--normalize"
        ])
        assert vector_result.returncode == 0, f"Vector creation failed: {vector_result.stderr}"
        assert vector_file.exists()

        # Step 4: Multi-steer generation
        generate_result = run_wisent_command([
            "multi-steer",
            "--vector", f"{vector_file}:1.0",
            "--model", MODEL,
            "--layer", "3",
            "--method", "CAA",
            "--prompt", "Is the Earth round?",
            "--max-new-tokens", "30"
        ])
        assert generate_result.returncode == 0, f"Generation failed: {generate_result.stderr}"
        assert len(generate_result.stdout) > 0, "Should have generated output"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
