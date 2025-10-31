"""
Test for multi_steer_from_trained_vectors.sh example.

This test validates practical use cases of combining pre-trained vectors
for different personas and scenarios (tech doc writer, teacher, etc).
"""

import subprocess
import pytest
import tempfile
import os


def create_named_vector(tmpdir, name, trait_label):
    """Helper to create a named test vector."""
    vector_path = os.path.join(tmpdir, f"{name}.pt")
    
    result = subprocess.run(
        [
            "python", "-m", "wisent.core.main", "generate-vector-from-task",
            "--task", "boolq",
            "--trait-label", trait_label,
            "--output", vector_path,
            "--model", "meta-llama/Llama-3.2-1B-Instruct",
            "--num-pairs", "10",
            "--layers", "3",
            "--token-aggregation", "average",
            "--method", "caa",
            "--normalize",
            "--device", "cpu"
        ],
        capture_output=True,
        text=True,
        timeout=300
    )
    
    assert result.returncode == 0, f"Failed to create {name} vector: {result.stderr}"
    return vector_path


def test_technical_documentation_persona():
    """Test combining vectors for technical documentation writer persona."""
    with tempfile.TemporaryDirectory() as tmpdir:
        formal_vector = create_named_vector(tmpdir, "formal", "formal_tone")
        technical_vector = create_named_vector(tmpdir, "technical", "technical")
        combined_path = os.path.join(tmpdir, "tech_doc.pt")
        
        result = subprocess.run(
            [
                "python", "-m", "wisent.core.main", "multi-steer",
                "--vector", f"{formal_vector}:0.5",
                "--vector", f"{technical_vector}:0.5",
                "--model", "meta-llama/Llama-3.2-1B-Instruct",
                "--layer", "3",
                "--method", "CAA",
                "--prompt", "Explain REST APIs.",
                "--max-new-tokens", "50",
                "--normalize-weights",
                "--save-combined", combined_path,
                "--device", "cpu",
                "--verbose"
            ],
            capture_output=True,
            text=True,
            timeout=180
        )
        
        assert result.returncode == 0, f"Command failed: {result.stderr}"
        assert os.path.exists(combined_path), "Combined vector not saved"


def test_friendly_teacher_persona():
    """Test combining vectors for friendly teacher persona."""
    with tempfile.TemporaryDirectory() as tmpdir:
        friendly_vector = create_named_vector(tmpdir, "friendly", "friendly")
        detailed_vector = create_named_vector(tmpdir, "detailed", "detailed")
        combined_path = os.path.join(tmpdir, "teacher.pt")
        
        result = subprocess.run(
            [
                "python", "-m", "wisent.core.main", "multi-steer",
                "--vector", f"{friendly_vector}:0.6",
                "--vector", f"{detailed_vector}:0.4",
                "--model", "meta-llama/Llama-3.2-1B-Instruct",
                "--layer", "3",
                "--method", "CAA",
                "--prompt", "How does photosynthesis work?",
                "--max-new-tokens", "50",
                "--normalize-weights",
                "--save-combined", combined_path,
                "--device", "cpu",
                "--verbose"
            ],
            capture_output=True,
            text=True,
            timeout=180
        )
        
        assert result.returncode == 0, f"Command failed: {result.stderr}"
        assert os.path.exists(combined_path), "Combined vector not saved"


def test_executive_summary_persona():
    """Test combining vectors for executive summary writer."""
    with tempfile.TemporaryDirectory() as tmpdir:
        concise_vector = create_named_vector(tmpdir, "concise", "concise")
        formal_vector = create_named_vector(tmpdir, "formal", "formal")
        combined_path = os.path.join(tmpdir, "executive.pt")
        
        result = subprocess.run(
            [
                "python", "-m", "wisent.core.main", "multi-steer",
                "--vector", f"{concise_vector}:0.6",
                "--vector", f"{formal_vector}:0.4",
                "--model", "meta-llama/Llama-3.2-1B-Instruct",
                "--layer", "3",
                "--method", "CAA",
                "--prompt", "Benefits of cloud computing.",
                "--max-new-tokens", "50",
                "--normalize-weights",
                "--save-combined", combined_path,
                "--device", "cpu",
                "--verbose"
            ],
            capture_output=True,
            text=True,
            timeout=180
        )
        
        assert result.returncode == 0, f"Command failed: {result.stderr}"
        assert os.path.exists(combined_path), "Combined vector not saved"


def test_comparing_weight_ratios():
    """Test comparing different weight ratios for the same prompt."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vector1 = create_named_vector(tmpdir, "v1", "trait1")
        vector2 = create_named_vector(tmpdir, "v2", "trait2")
        
        prompt = "Explain machine learning."
        
        # Configuration A: More weight on vector1
        result_a = subprocess.run(
            [
                "python", "-m", "wisent.core.main", "multi-steer",
                "--vector", f"{vector1}:0.7",
                "--vector", f"{vector2}:0.3",
                "--model", "meta-llama/Llama-3.2-1B-Instruct",
                "--layer", "3",
                "--method", "CAA",
                "--prompt", prompt,
                "--max-new-tokens", "50",
                "--normalize-weights",
                "--device", "cpu",
                "--verbose"
            ],
            capture_output=True,
            text=True,
            timeout=180
        )
        
        assert result_a.returncode == 0, f"Config A failed: {result_a.stderr}"
        
        # Configuration B: More weight on vector2
        result_b = subprocess.run(
            [
                "python", "-m", "wisent.core.main", "multi-steer",
                "--vector", f"{vector1}:0.3",
                "--vector", f"{vector2}:0.7",
                "--model", "meta-llama/Llama-3.2-1B-Instruct",
                "--layer", "3",
                "--method", "CAA",
                "--prompt", prompt,
                "--max-new-tokens", "50",
                "--normalize-weights",
                "--device", "cpu",
                "--verbose"
            ],
            capture_output=True,
            text=True,
            timeout=180
        )
        
        assert result_b.returncode == 0, f"Config B failed: {result_b.stderr}"
        
        # Configuration C: Balanced
        result_c = subprocess.run(
            [
                "python", "-m", "wisent.core.main", "multi-steer",
                "--vector", f"{vector1}:0.5",
                "--vector", f"{vector2}:0.5",
                "--model", "meta-llama/Llama-3.2-1B-Instruct",
                "--layer", "3",
                "--method", "CAA",
                "--prompt", prompt,
                "--max-new-tokens", "50",
                "--normalize-weights",
                "--device", "cpu",
                "--verbose"
            ],
            capture_output=True,
            text=True,
            timeout=180
        )
        
        assert result_c.returncode == 0, f"Config C failed: {result_c.stderr}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
