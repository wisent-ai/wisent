"""Shared helpers for multi-modal adapter and pipeline tests."""
from __future__ import annotations

import argparse
import logging
import sys
import traceback
from dataclasses import dataclass
from typing import Dict, Any, List

import torch
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Result of a single test."""
    name: str
    passed: bool
    message: str
    details: Dict[str, Any] | None = None


class AdapterTester:
    """Base class for adapter testing."""

    def __init__(self):
        self.results: List[TestResult] = []

    def log_result(
        self, name: str, passed: bool, message: str,
        details: Dict[str, Any] | None = None,
    ):
        result = TestResult(
            name=name, passed=passed,
            message=message, details=details,
        )
        self.results.append(result)
        status = "PASS" if passed else "FAIL"
        logger.info(f"  [{status}] {name}: {message}")

    def run_test(self, name: str, test_fn):
        """Run a test function and capture results."""
        try:
            result = test_fn()
            self.log_result(
                name, True,
                result if isinstance(result, str) else "Success",
            )
        except Exception as e:
            self.log_result(name, False, f"{type(e).__name__}: {e}")
            logger.debug(traceback.format_exc())

    def summary(self) -> tuple[int, int]:
        """Return (passed, total) counts."""
        passed = sum(1 for r in self.results if r.passed)
        return passed, len(self.results)


def create_test_audio(
    frequency: float = 440, amplitude: float = 0.5,
    duration: float = 2.0, sample_rate: int = 16000,
):
    """Create synthetic test audio waveform."""
    t = np.linspace(
        0, duration, int(sample_rate * duration),
        dtype=np.float32,
    )
    waveform = amplitude * np.sin(2 * np.pi * frequency * t)
    return torch.from_numpy(waveform), sample_rate


def create_test_video(
    num_frames: int = 16, height: int = 224,
    width: int = 224, channels: int = 3,
):
    """Create synthetic test video frames."""
    return torch.rand(num_frames, channels, height, width)


def create_test_policy_model():
    """Create a simple MLP policy network for robotics testing."""
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(12, 64), nn.ReLU(),
        nn.Linear(64, 64), nn.ReLU(),
        nn.Linear(64, 6), nn.Tanh(),
    )


def create_test_robot_state(dim: int = 6):
    """Create random robot state arrays."""
    return (
        np.random.randn(dim).astype(np.float32),
        np.random.randn(dim).astype(np.float32),
    )


def print_test_summary(results: dict):
    """Print test summary across adapter types."""
    logger.info(f"\n{'='*60}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*60}")
    total_passed = 0
    total_tests = 0
    for adapter_name, (passed, total) in results.items():
        status = "PASS" if passed == total else "FAIL"
        logger.info(
            f"  {adapter_name.upper():12} [{status}]"
            f" {passed}/{total} tests passed"
        )
        total_passed += passed
        total_tests += total
    logger.info(f"{'='*60}")
    overall = "PASS" if total_passed == total_tests else "FAIL"
    logger.info(
        f"  OVERALL      [{overall}]"
        f" {total_passed}/{total_tests} tests passed"
    )
    logger.info(f"{'='*60}")
    return total_passed == total_tests


def make_adapter_argparser(description: str):
    """Create standard argparser for adapter tests."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--text", action="store_true")
    parser.add_argument("--audio", action="store_true")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--robotics", action="store_true")
    parser.add_argument("--multimodal", action="store_true")
    return parser


@dataclass
class PipelineTestResult:
    """Result of a pipeline test."""
    adapter_name: str
    pairs_created: int
    activations_extracted: bool
    vectors_trained: bool
    generation_works: bool
    steering_has_effect: bool
    details: Dict[str, Any]

    @property
    def passed(self) -> bool:
        return all([
            self.pairs_created > 0,
            self.activations_extracted,
            self.vectors_trained,
            self.generation_works,
            self.steering_has_effect,
        ])


def print_pipeline_summary(results: list) -> bool:
    """Print pipeline test summary."""
    logger.info(f"\n{'='*60}")
    logger.info("PIPELINE TEST SUMMARY")
    logger.info(f"{'='*60}")
    all_passed = True
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        logger.info(f"  {result.adapter_name.upper():12} [{status}]")
        logger.info(f"    - Pairs: {result.pairs_created}")
        logger.info(f"    - Activations: {result.activations_extracted}")
        logger.info(f"    - Vectors: {result.vectors_trained}")
        logger.info(f"    - Generation: {result.generation_works}")
        logger.info(f"    - Effect: {result.steering_has_effect}")
        if not result.passed:
            all_passed = False
    logger.info(f"{'='*60}")
    logger.info(f"OVERALL: {'PASSED' if all_passed else 'FAILED'}")
    return all_passed
