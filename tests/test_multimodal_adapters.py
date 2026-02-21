"""
Test script for multi-modal Wisent adapters.

Tests all adapter types with real models to verify:
1. Model loading works
2. Encoding produces valid tensors
3. Activation extraction works
4. Steering vectors can be computed
5. Generation with steering works

Usage:
    python tests/test_multimodal_adapters.py [--all] [--text] [--audio] [--video] [--robotics] [--multimodal]
"""
from __future__ import annotations

import sys
from pathlib import Path
import logging

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from wisent import (
    Wisent, TextContent, AudioContent,
    TextAdapter, AudioAdapter,
)
from _multimodal_helpers import (
    AdapterTester, print_test_summary, make_adapter_argparser,
)
from _multimodal_adapters_extra import (
    VideoAdapterTester, RoboticsAdapterTester, MultimodalAdapterTester,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

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

    return print_test_summary(results)


def main():
    parser = make_adapter_argparser("Test multi-modal Wisent adapters")
    args = parser.parse_args()
    if not any([args.all, args.text, args.audio, args.video,
                args.robotics, args.multimodal]):
        args.all = True
    success = run_all_tests(args)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
