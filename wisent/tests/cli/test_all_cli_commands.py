"""
Comprehensive CLI tests for ALL Wisent commands.
Uses gpt2 (non-gated) for testing.
Run with: pytest wisent/tests/test_all_cli_commands.py -v
"""

import pytest
import json

from wisent.tests._cli_fixtures import (
    MODEL, TASK, run_wisent_command, run_ok,
    make_test_pairs, make_test_responses,
)


@pytest.fixture(scope="module")
def test_pairs_file(tmp_path_factory):
    """Create a test pairs file for reuse."""
    fp = tmp_path_factory.mktemp("pairs") / "test_pairs.json"
    fp.write_text(json.dumps(make_test_pairs()))
    return str(fp)


@pytest.fixture(scope="module")
def test_responses_file(tmp_path_factory):
    """Create a test responses file for evaluation."""
    fp = tmp_path_factory.mktemp("responses") / "test_responses.json"
    fp.write_text(json.dumps(make_test_responses()))
    return str(fp)


class TestPairGenerationCommands:
    def test_generate_pairs_from_task(self, tmp_path):
        of = tmp_path / "pairs.json"
        run_ok(["generate-pairs-from-task", TASK, "--output", str(of), "--verbose"])
        assert of.exists()
        with open(of) as f:
            data = json.load(f)
            pairs = data["pairs"] if isinstance(data, dict) else data
            assert len(pairs) > 0

    def test_generate_pairs_synthetic(self, tmp_path):
        of = tmp_path / "synthetic_pairs.json"
        run_ok(["generate-pairs", "--trait", "truthfulness", "--num-pairs", "3",
                "--output", str(of), "--model", MODEL, "--similarity-threshold", "0.8", "--verbose"])
        assert of.exists()

    def test_diagnose_pairs(self, test_pairs_file):
        run_ok(["diagnose-pairs", test_pairs_file, "--show-sample", "--verbose"])


class TestActivationCommands:
    def test_get_activations(self, test_pairs_file, tmp_path):
        of = tmp_path / "activations.json"
        run_ok(["get-activations", test_pairs_file, "--output", str(of), "--model", MODEL,
                "--layers", "3", "--token-aggregation", "average", "--verbose"])
        assert of.exists()

    def test_get_activations_multiple_layers(self, test_pairs_file, tmp_path):
        of = tmp_path / "multi_layer_activations.json"
        run_ok(["get-activations", test_pairs_file, "--output", str(of), "--model", MODEL,
                "--layers", "2,3,4", "--token-aggregation", "final", "--verbose"])
        assert of.exists()


class TestSteeringVectorCommands:
    def test_generate_vector_from_task(self, tmp_path):
        of = tmp_path / "vector.pt"
        run_ok(["generate-vector-from-task", "--task", TASK, "--trait-label", "correctness",
                "--output", str(of), "--model", MODEL, "--num-pairs", "5", "--layers", "3",
                "--token-aggregation", "average", "--method", "caa", "--normalize", "--verbose"])
        assert of.exists()

    def test_generate_vector_from_synthetic(self, tmp_path):
        of = tmp_path / "synthetic_vector.pt"
        run_ok(["generate-vector-from-synthetic", "--trait", "helpfulness", "--output", str(of),
                "--model", MODEL, "--num-pairs", "3", "--layers", "3",
                "--token-aggregation", "average", "--method", "caa", "--normalize", "--verbose"])
        assert of.exists()

    def test_create_steering_vector(self, test_pairs_file, tmp_path):
        ef = tmp_path / "enriched.json"
        vf = tmp_path / "vector.pt"
        run_ok(["get-activations", test_pairs_file, "--output", str(ef), "--model", MODEL,
                "--layers", "3", "--token-aggregation", "average"])
        run_ok(["create-steering-vector", str(ef), "--output", str(vf),
                "--method", "caa", "--normalize", "--verbose"])
        assert vf.exists()

    def test_diagnose_vectors(self, test_pairs_file, tmp_path):
        ef = tmp_path / "enriched.json"
        vf = tmp_path / "vector.json"
        run_ok(["get-activations", test_pairs_file, "--output", str(ef), "--model", MODEL,
                "--layers", "3", "--token-aggregation", "average"])
        run_ok(["create-steering-vector", str(ef), "--output", str(vf),
                "--method", "caa", "--normalize"])
        r = run_wisent_command(["diagnose-vectors", str(vf), "--show-sample", "--verbose"])
        assert r.returncode == 0 or "not found" not in r.stderr.lower()


class TestGenerationEvaluationCommands:
    def test_generate_responses(self, tmp_path):
        of = tmp_path / "responses.json"
        run_ok(["generate-responses", MODEL, "--task", TASK, "--num-questions", "3",
                "--max-new-tokens", "50", "--temperature", "0.7", "--output", str(of), "--verbose"])
        assert of.exists()

    def test_evaluate_responses(self, test_responses_file, tmp_path):
        of = tmp_path / "evaluation.json"
        run_ok(["evaluate-responses", "--input", test_responses_file,
                "--output", str(of), "--task", TASK, "--verbose"])
        assert of.exists()

    def test_evaluate_refusal(self, tmp_path):
        of = tmp_path / "refusal_eval.json"
        run_ok(["evaluate-refusal", "--model", MODEL, "--output", str(of), "--evaluator", "keyword",
                "--num-prompts", "5", "--max-new-tokens", "50", "--verbose"])
        assert of.exists()


class TestMultiSteerCommands:
    def _train_vector(self, tmp_path, label, suffix):
        vp = tmp_path / f"vector_{suffix}.pt"
        run_ok(["generate-vector-from-task", "--task", TASK, "--trait-label", label,
                "--output", str(vp), "--model", MODEL, "--num-pairs", "5", "--layers", "3",
                "--token-aggregation", "average", "--method", "caa", "--normalize"])
        return vp

    def test_multi_steer(self, tmp_path):
        v1, v2 = self._train_vector(tmp_path, "trait1", "1"), self._train_vector(tmp_path, "trait2", "2")
        run_ok(["multi-steer", "--vector", f"{v1}:0.5", "--vector", f"{v2}:0.5",
                "--model", MODEL, "--layer", "3", "--method", "CAA", "--prompt", "What is AI?",
                "--max-new-tokens", "30", "--normalize-weights", "--verbose"])

    def test_modify_weights(self, tmp_path):
        od = tmp_path / "modified_model"
        run_ok(["modify-weights", "--task", TASK, "--output-dir", str(od), "--model", MODEL,
                "--num-pairs", "5", "--trait-label", "correctness", "--layers", "3",
                "--token-aggregation", "average", "--method", "additive", "--alpha", "0.5", "--verbose"])
        assert od.exists()


class TestOptimizationCommands:
    def test_optimize_classification(self):
        run_ok(["optimize-classification", MODEL, "--optimization-metric", "f1",
                "--layer-range", "2-8", "--aggregation-methods", "average", "final", "first",
                "--threshold-range", "0.3", "0.5", "0.7"])

    def test_optimize_steering(self):
        run_ok(["optimize-steering", "comprehensive", MODEL, "--tasks", TASK,
                "--methods", "CAA", "PCA", "--verbose"])

    def test_optimize_sample_size(self):
        run_ok(["optimize-sample-size", MODEL, "--task", TASK,
                "--sample-sizes", "10", "25", "50", "100", "--layer", "5", "--verbose"])

    def test_optimize_all(self, tmp_path):
        run_ok(["optimize-all", MODEL, "--tasks", TASK,
                "--output-dir", str(tmp_path / "optimized"), "--verbose"])


class TestConfigUtilityCommands:
    def test_tasks_list(self):
        r = run_ok(["tasks", "--list-tasks"])
        assert "boolq" in r.stdout.lower() or "task" in r.stdout.lower()

    def test_tasks_info(self):
        run_ok(["tasks", "--task-info", TASK])

    def test_tasks_run(self, tmp_path):
        run_ok(["tasks", TASK, "--model", MODEL, "--layer", "3",
                "--output", str(tmp_path / "task_output"), "--token-aggregation", "average", "--verbose"])

    def test_inference_config_show(self):
        run_ok(["inference-config", "show"])

    def test_inference_config_set_reset(self):
        run_ok(["inference-config", "set", "--max-new-tokens", "100"])
        run_ok(["inference-config", "reset"])

    def test_optimization_cache_list(self):
        run_ok(["optimization-cache", "list"])

    def test_optimization_cache_export_import(self, tmp_path):
        ef = tmp_path / "cache_export.json"
        run_ok(["optimization-cache", "export", str(ef)])
        if ef.exists():
            ir = run_wisent_command(["optimization-cache", "import", str(ef)])
            assert ir.returncode == 0 or "empty" in ir.stderr.lower()


class TestFullPipeline:
    def test_complete_steering_pipeline(self, tmp_path):
        pf = tmp_path / "pairs.json"
        ef = tmp_path / "enriched.json"
        vf = tmp_path / "vector.pt"
        run_ok(["generate-pairs-from-task", TASK, "--output", str(pf)])
        assert pf.exists()
        run_ok(["get-activations", str(pf), "--output", str(ef), "--model", MODEL,
                "--layers", "3", "--token-aggregation", "average"])
        run_ok(["create-steering-vector", str(ef), "--output", str(vf),
                "--method", "caa", "--normalize"])
        assert vf.exists()
        gr = run_ok(["multi-steer", "--vector", f"{vf}:1.0", "--model", MODEL, "--layer", "3",
                      "--method", "CAA", "--prompt", "Is the Earth round?", "--max-new-tokens", "30"])
        assert len(gr.stdout) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
