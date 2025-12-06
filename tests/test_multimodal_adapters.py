"""
Test script for multi-modal Wisent adapters.

This script tests all adapter types with real models to verify:
1. Model loading works
2. Encoding produces valid tensors
3. Activation extraction works
4. Steering vectors can be computed
5. Generation with steering works

Usage:
    python tests/test_multimodal_adapters.py [--all] [--text] [--audio] [--video] [--robotics] [--multimodal]
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass
import tempfile
import logging

import torch
import torch.nn as nn
import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from wisent import (
    Wisent,
    TextContent,
    AudioContent,
    VideoContent,
    ImageContent,
    RobotState,
    RobotAction,
    RobotTrajectory,
    MultimodalContent,
    TextAdapter,
    AudioAdapter,
    VideoAdapter,
    RoboticsAdapter,
    MultimodalAdapter,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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

    def log_result(self, name: str, passed: bool, message: str, details: Dict[str, Any] | None = None):
        result = TestResult(name=name, passed=passed, message=message, details=details)
        self.results.append(result)
        status = "PASS" if passed else "FAIL"
        logger.info(f"  [{status}] {name}: {message}")

    def run_test(self, name: str, test_fn):
        """Run a test function and capture results."""
        try:
            result = test_fn()
            self.log_result(name, True, result if isinstance(result, str) else "Success")
        except Exception as e:
            self.log_result(name, False, f"{type(e).__name__}: {str(e)}")
            logger.debug(traceback.format_exc())

    def summary(self) -> tuple[int, int]:
        """Return (passed, total) counts."""
        passed = sum(1 for r in self.results if r.passed)
        return passed, len(self.results)


class TextAdapterTester(AdapterTester):
    """Test the TextAdapter with a small LLM."""

    MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"  # Small model for testing

    def __init__(self):
        super().__init__()
        self.adapter = None
        self.wisent = None

    def test_all(self):
        logger.info(f"\n{'='*60}")
        logger.info("Testing TextAdapter")
        logger.info(f"{'='*60}")

        self.run_test("Model Loading", self.test_model_loading)

        if self.adapter is None:
            logger.warning("Skipping remaining tests - model failed to load")
            return

        self.run_test("Tokenizer Access", self.test_tokenizer)
        self.run_test("Encoding", self.test_encoding)
        self.run_test("Intervention Points", self.test_intervention_points)
        self.run_test("Activation Extraction", self.test_activation_extraction)
        self.run_test("Wisent Integration", self.test_wisent_integration)
        self.run_test("Steering Generation", self.test_steering_generation)

    def test_model_loading(self):
        self.adapter = TextAdapter(model_name=self.MODEL_NAME)
        assert self.adapter.model is not None
        return f"Loaded {self.MODEL_NAME} on {self.adapter.device}"

    def test_tokenizer(self):
        tokenizer = self.adapter.tokenizer
        assert tokenizer is not None
        tokens = tokenizer("Hello world", return_tensors="pt")
        assert "input_ids" in tokens
        return f"Tokenizer works, vocab size: {tokenizer.vocab_size}"

    def test_encoding(self):
        content = TextContent("What is 2 + 2?")
        embedding = self.adapter.encode(content)
        assert embedding is not None
        assert embedding.dim() == 3  # [batch, seq, hidden]
        return f"Embedding shape: {tuple(embedding.shape)}"

    def test_intervention_points(self):
        points = self.adapter.get_intervention_points()
        assert len(points) > 0
        recommended = [p for p in points if p.recommended]
        return f"Found {len(points)} layers, {len(recommended)} recommended"

    def test_activation_extraction(self):
        content = TextContent("Test prompt for activation extraction")
        points = self.adapter.get_intervention_points()
        layers = [points[0].name, points[-1].name]  # First and last
        activations = self.adapter.extract_activations(content, layers)
        assert len(activations) > 0
        for layer, tensor in activations.items():
            assert tensor is not None
            assert tensor.dim() >= 2
        return f"Extracted {len(activations)} layer activations"

    def test_wisent_integration(self):
        self.wisent = Wisent(adapter=self.adapter)
        self.wisent.add_pair(
            positive="I'd be happy to help you with that question.",
            negative="I cannot help with that.",
            trait="helpfulness",
        )
        self.wisent.add_pair(
            positive="Let me explain this clearly and thoroughly.",
            negative="Whatever.",
            trait="helpfulness",
        )
        assert len(self.wisent._pairs["helpfulness"]) == 2
        return "Added 2 contrastive pairs"

    def test_steering_generation(self):
        # Train vectors
        self.wisent.train(traits=["helpfulness"])
        assert self.wisent.is_trained

        # Check vectors were created
        trait_info = self.wisent.get_trait_info("helpfulness")
        assert trait_info is not None
        assert trait_info.steering_vectors is not None

        # Generate without steering
        response_base = self.wisent.generate("What is Python?")
        assert isinstance(response_base, str)
        assert len(response_base) > 0

        # Generate with steering
        response_steered = self.wisent.generate(
            "What is Python?",
            steer={"helpfulness": 1.5}
        )
        assert isinstance(response_steered, str)
        assert len(response_steered) > 0

        return f"Generated responses (base: {len(response_base)} chars, steered: {len(response_steered)} chars)"


class AudioAdapterTester(AdapterTester):
    """Test the AudioAdapter with Whisper."""

    MODEL_NAME = "openai/whisper-tiny"  # Smallest Whisper model

    def __init__(self):
        super().__init__()
        self.adapter = None
        self.test_audio = None

    def _create_test_audio(self) -> AudioContent:
        """Create synthetic test audio."""
        # Generate a simple sine wave
        sample_rate = 16000
        duration = 2.0  # seconds
        frequency = 440  # Hz (A4 note)

        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
        waveform = 0.5 * np.sin(2 * np.pi * frequency * t)

        return AudioContent(
            waveform=torch.from_numpy(waveform),
            sample_rate=sample_rate,
        )

    def test_all(self):
        logger.info(f"\n{'='*60}")
        logger.info("Testing AudioAdapter")
        logger.info(f"{'='*60}")

        self.run_test("Model Loading", self.test_model_loading)

        if self.adapter is None:
            logger.warning("Skipping remaining tests - model failed to load")
            return

        self.run_test("Processor Access", self.test_processor)
        self.run_test("Test Audio Creation", self.test_audio_creation)
        self.run_test("Encoding", self.test_encoding)
        self.run_test("Intervention Points", self.test_intervention_points)
        self.run_test("Activation Extraction", self.test_activation_extraction)
        self.run_test("Wisent Integration", self.test_wisent_integration)

    def test_model_loading(self):
        self.adapter = AudioAdapter(model_name=self.MODEL_NAME)
        assert self.adapter.model is not None
        return f"Loaded {self.MODEL_NAME} on {self.adapter.device}"

    def test_processor(self):
        processor = self.adapter.processor
        assert processor is not None
        return "Processor loaded successfully"

    def test_audio_creation(self):
        self.test_audio = self._create_test_audio()
        tensor = self.test_audio.to_tensor()
        assert tensor is not None
        assert tensor.dim() == 1
        return f"Created test audio: {tensor.shape[0]} samples"

    def test_encoding(self):
        embedding = self.adapter.encode(self.test_audio)
        assert embedding is not None
        assert embedding.dim() >= 2
        return f"Embedding shape: {tuple(embedding.shape)}"

    def test_intervention_points(self):
        points = self.adapter.get_intervention_points()
        assert len(points) > 0
        encoder_points = [p for p in points if "encoder" in p.name]
        decoder_points = [p for p in points if "decoder" in p.name]
        return f"Found {len(encoder_points)} encoder, {len(decoder_points)} decoder layers"

    def test_activation_extraction(self):
        points = self.adapter.get_intervention_points()
        if not points:
            return "No intervention points found (model architecture)"

        layers = [points[0].name]
        activations = self.adapter.extract_activations(self.test_audio, layers)
        assert len(activations) > 0
        return f"Extracted {len(activations)} layer activations"

    def test_wisent_integration(self):
        wisent = Wisent(adapter=self.adapter)

        # Create two different test audios
        audio1 = self._create_test_audio()
        audio2 = AudioContent(
            waveform=torch.from_numpy(
                0.5 * np.sin(2 * np.pi * 880 * np.linspace(0, 2, 32000, dtype=np.float32))
            ),
            sample_rate=16000,
        )

        wisent.add_pair(positive=audio1, negative=audio2, trait="tone")
        assert len(wisent._pairs["tone"]) == 1
        return "Wisent integration successful"


class VideoAdapterTester(AdapterTester):
    """Test the VideoAdapter with VideoMAE."""

    MODEL_NAME = "MCG-NJU/videomae-base"  # VideoMAE base model

    def __init__(self):
        super().__init__()
        self.adapter = None
        self.test_video = None

    def _create_test_video(self) -> VideoContent:
        """Create synthetic test video."""
        num_frames = 16
        height, width = 224, 224
        channels = 3

        # Create random color frames (simulating video)
        frames = torch.rand(num_frames, channels, height, width)

        return VideoContent(frames=frames, fps=30.0)

    def test_all(self):
        logger.info(f"\n{'='*60}")
        logger.info("Testing VideoAdapter")
        logger.info(f"{'='*60}")

        self.run_test("Model Loading", self.test_model_loading)

        if self.adapter is None:
            logger.warning("Skipping remaining tests - model failed to load")
            return

        self.run_test("Processor Access", self.test_processor)
        self.run_test("Test Video Creation", self.test_video_creation)
        self.run_test("Encoding", self.test_encoding)
        self.run_test("Intervention Points", self.test_intervention_points)
        self.run_test("Activation Extraction", self.test_activation_extraction)
        self.run_test("Wisent Integration", self.test_wisent_integration)

    def test_model_loading(self):
        self.adapter = VideoAdapter(model_name=self.MODEL_NAME)
        assert self.adapter.model is not None
        return f"Loaded {self.MODEL_NAME} on {self.adapter.device}"

    def test_processor(self):
        processor = self.adapter.processor
        assert processor is not None
        return "Processor loaded successfully"

    def test_video_creation(self):
        self.test_video = self._create_test_video()
        tensor = self.test_video.to_tensor()
        assert tensor is not None
        assert tensor.dim() == 4  # [T, C, H, W]
        return f"Created test video: {tuple(tensor.shape)}"

    def test_encoding(self):
        embedding = self.adapter.encode(self.test_video)
        assert embedding is not None
        assert embedding.dim() >= 2
        return f"Embedding shape: {tuple(embedding.shape)}"

    def test_intervention_points(self):
        points = self.adapter.get_intervention_points()
        assert len(points) > 0
        recommended = [p for p in points if p.recommended]
        return f"Found {len(points)} layers, {len(recommended)} recommended"

    def test_activation_extraction(self):
        points = self.adapter.get_intervention_points()
        layers = [points[0].name, points[-1].name]
        activations = self.adapter.extract_activations(self.test_video, layers)
        assert len(activations) > 0
        return f"Extracted {len(activations)} layer activations"

    def test_wisent_integration(self):
        wisent = Wisent(adapter=self.adapter)

        video1 = self._create_test_video()
        video2 = VideoContent(
            frames=torch.rand(16, 3, 224, 224) * 0.5,  # Darker video
            fps=30.0,
        )

        wisent.add_pair(positive=video1, negative=video2, trait="brightness")
        assert len(wisent._pairs["brightness"]) == 1
        return "Wisent integration successful"


class RoboticsAdapterTester(AdapterTester):
    """Test the RoboticsAdapter with a simple MLP policy."""

    def __init__(self):
        super().__init__()
        self.adapter = None
        self.policy = None
        self.test_state = None

    def _create_test_policy(self) -> nn.Module:
        """Create a simple MLP policy network."""
        state_dim = 12  # e.g., joint positions + velocities
        action_dim = 6  # e.g., joint velocity commands
        hidden_dim = 64

        policy = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),  # Bound actions to [-1, 1]
        )

        return policy

    def _create_test_state(self) -> RobotState:
        """Create a test robot state."""
        return RobotState(
            joint_positions=np.random.randn(6).astype(np.float32),
            joint_velocities=np.random.randn(6).astype(np.float32),
        )

    def _create_test_trajectory(self, length: int = 10) -> RobotTrajectory:
        """Create a test trajectory."""
        states = [self._create_test_state() for _ in range(length)]
        actions = [
            RobotAction(raw_action=np.random.randn(6).astype(np.float32))
            for _ in range(length - 1)
        ]
        return RobotTrajectory(states=tuple(states), actions=tuple(actions))

    def test_all(self):
        logger.info(f"\n{'='*60}")
        logger.info("Testing RoboticsAdapter")
        logger.info(f"{'='*60}")

        self.run_test("Policy Creation", self.test_policy_creation)
        self.run_test("Adapter Loading", self.test_adapter_loading)

        if self.adapter is None:
            logger.warning("Skipping remaining tests - adapter failed to load")
            return

        self.run_test("Test State Creation", self.test_state_creation)
        self.run_test("Forward Pass", self.test_forward_pass)
        self.run_test("Intervention Points", self.test_intervention_points)
        self.run_test("Activation Extraction", self.test_activation_extraction)
        self.run_test("Trajectory Handling", self.test_trajectory_handling)
        self.run_test("Wisent Integration", self.test_wisent_integration)
        self.run_test("Steering Action", self.test_steering_action)

    def test_policy_creation(self):
        self.policy = self._create_test_policy()
        assert self.policy is not None
        param_count = sum(p.numel() for p in self.policy.parameters())
        return f"Created MLP policy with {param_count} parameters"

    def test_adapter_loading(self):
        self.adapter = RoboticsAdapter(
            model=self.policy,
            state_dim=12,
            action_dim=6,
        )
        assert self.adapter.model is not None
        return f"Adapter loaded successfully on {self.adapter.device}"

    def test_state_creation(self):
        self.test_state = self._create_test_state()
        tensor = self.test_state.to_tensor()
        assert tensor is not None
        assert tensor.dim() == 1
        return f"Created test state: {tuple(tensor.shape)}"

    def test_forward_pass(self):
        action = self.adapter._generate_unsteered(self.test_state)
        assert action is not None
        assert action.raw_action is not None
        return f"Generated action: {action.raw_action.shape}"

    def test_intervention_points(self):
        points = self.adapter.get_intervention_points()
        assert len(points) > 0
        return f"Found {len(points)} intervention points"

    def test_activation_extraction(self):
        points = self.adapter.get_intervention_points()
        layers = [points[0].name]
        activations = self.adapter.extract_activations(self.test_state, layers)
        assert len(activations) > 0
        return f"Extracted {len(activations)} layer activations"

    def test_trajectory_handling(self):
        trajectory = self._create_test_trajectory(5)
        assert trajectory.length == 5
        tensor = trajectory.to_tensor()
        assert tensor.shape[0] == 5
        return f"Trajectory: {trajectory.length} states, {len(trajectory.actions)} actions"

    def test_wisent_integration(self):
        wisent = Wisent(adapter=self.adapter)

        traj1 = self._create_test_trajectory(5)
        traj2 = self._create_test_trajectory(5)

        wisent.add_pair(positive=traj1, negative=traj2, trait="gentleness")
        assert len(wisent._pairs["gentleness"]) == 1
        return "Wisent integration successful"

    def test_steering_action(self):
        wisent = Wisent(adapter=self.adapter)

        # Add pairs
        for _ in range(3):
            wisent.add_pair(
                positive=self._create_test_state(),
                negative=self._create_test_state(),
                trait="safety",
            )

        # Train
        wisent.train(traits=["safety"])
        assert wisent.is_trained

        # Get action with steering
        action = wisent.act(self.test_state, steer={"safety": 1.0})
        assert action is not None
        return "Steering action generation successful"


class MultimodalAdapterTester(AdapterTester):
    """Test the MultimodalAdapter with a small VLM."""

    MODEL_NAME = "llava-hf/llava-1.5-7b-hf"

    def __init__(self):
        super().__init__()
        self.adapter = None
        self.test_content = None

    def _create_test_image(self) -> ImageContent:
        """Create a synthetic test image."""
        height, width = 224, 224
        channels = 3
        pixels = torch.rand(channels, height, width)
        return ImageContent(pixels=pixels)

    def _create_test_content(self) -> MultimodalContent:
        """Create multimodal content (image + text)."""
        image = self._create_test_image()
        text = TextContent("What do you see in this image?")
        return MultimodalContent(contents=(image, text))

    def test_all(self):
        logger.info(f"\n{'='*60}")
        logger.info("Testing MultimodalAdapter")
        logger.info(f"{'='*60}")

        self.run_test("Content Creation", self.test_content_creation)
        self.run_test("Model Loading", self.test_model_loading)

        if self.adapter is None:
            logger.warning("Skipping remaining tests - model failed to load")
            return

        self.run_test("Intervention Points", self.test_intervention_points)
        self.run_test("Wisent Integration", self.test_wisent_integration)

    def test_content_creation(self):
        self.test_content = self._create_test_content()
        assert self.test_content is not None

        image = self.test_content.get_image()
        text = self.test_content.get_text()

        assert image is not None
        assert text is not None
        return f"Created multimodal content (image: {image.pixels.shape}, text: '{text.text[:30]}...')"

    def test_model_loading(self):
        self.adapter = MultimodalAdapter(model_name=self.MODEL_NAME)
        assert self.adapter.model is not None
        return f"Loaded {self.MODEL_NAME} on {self.adapter.device}"

    def test_intervention_points(self):
        points = self.adapter.get_intervention_points()
        assert len(points) > 0
        vision_points = [p for p in points if "vision" in p.name]
        language_points = [p for p in points if "language" in p.name]
        return f"Found {len(vision_points)} vision, {len(language_points)} language layers"

    def test_wisent_integration(self):
        wisent = Wisent(adapter=self.adapter)

        content1 = self._create_test_content()
        content2 = self._create_test_content()

        wisent.add_pair(positive=content1, negative=content2, trait="detail")
        assert len(wisent._pairs["detail"]) == 1
        return "Wisent integration successful"


def run_all_tests(args):
    """Run all adapter tests."""
    results = {}

    if args.text or args.all:
        tester = TextAdapterTester()
        tester.test_all()
        results["text"] = tester.summary()

    if args.audio or args.all:
        tester = AudioAdapterTester()
        tester.test_all()
        results["audio"] = tester.summary()

    if args.video or args.all:
        tester = VideoAdapterTester()
        tester.test_all()
        results["video"] = tester.summary()

    if args.robotics or args.all:
        tester = RoboticsAdapterTester()
        tester.test_all()
        results["robotics"] = tester.summary()

    if args.multimodal or args.all:
        tester = MultimodalAdapterTester()
        tester.test_all()
        results["multimodal"] = tester.summary()

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*60}")

    total_passed = 0
    total_tests = 0

    for adapter_name, (passed, total) in results.items():
        status = "PASS" if passed == total else "FAIL"
        logger.info(f"  {adapter_name.upper():12} [{status}] {passed}/{total} tests passed")
        total_passed += passed
        total_tests += total

    logger.info(f"{'='*60}")
    overall_status = "PASS" if total_passed == total_tests else "FAIL"
    logger.info(f"  OVERALL      [{overall_status}] {total_passed}/{total_tests} tests passed")
    logger.info(f"{'='*60}")

    return total_passed == total_tests


def main():
    parser = argparse.ArgumentParser(description="Test multi-modal Wisent adapters")
    parser.add_argument("--all", action="store_true", help="Run all adapter tests")
    parser.add_argument("--text", action="store_true", help="Test TextAdapter")
    parser.add_argument("--audio", action="store_true", help="Test AudioAdapter")
    parser.add_argument("--video", action="store_true", help="Test VideoAdapter")
    parser.add_argument("--robotics", action="store_true", help="Test RoboticsAdapter")
    parser.add_argument("--multimodal", action="store_true", help="Test MultimodalAdapter")

    args = parser.parse_args()

    # Default to --all if nothing specified
    if not any([args.all, args.text, args.audio, args.video, args.robotics, args.multimodal]):
        args.all = True

    success = run_all_tests(args)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
