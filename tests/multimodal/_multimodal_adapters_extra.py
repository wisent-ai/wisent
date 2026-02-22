"""Extra adapter testers: Video, Robotics, Multimodal."""
from __future__ import annotations
import logging
import numpy as np
import torch
import torch.nn as nn
from wisent import (
    Wisent, VideoContent, ImageContent, RobotState, RobotAction,
    RobotTrajectory, MultimodalContent, TextContent,
    VideoAdapter, RoboticsAdapter, MultimodalAdapter,)
from _multimodal_helpers import AdapterTester
logger = logging.getLogger(__name__)
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
        logger.info(f"{'='*60}")

        self.run_test("Model Loading", self.test_model_loading)

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
        logger.info(f"{'='*60}")

        self.run_test("Policy Creation", self.test_policy_creation)
        self.run_test("Adapter Loading", self.test_adapter_loading)

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
        logger.info(f"{'='*60}")

        self.run_test("Content Creation", self.test_content_creation)
        self.run_test("Model Loading", self.test_model_loading)

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

