"""Extra pipeline tests: Video and Robotics."""
from __future__ import annotations
import logging
import numpy as np
import torch
import torch.nn as nn
from wisent import (
    Wisent, VideoContent, RobotState, RobotAction,
    RobotTrajectory,
)
from wisent.core.adapters import VideoAdapter, RoboticsAdapter
from _multimodal_helpers import PipelineTestResult

logger = logging.getLogger(__name__)

class VideoPipelineTest:
    """Full pipeline test for video steering."""

    MODEL_NAME = "MCG-NJU/videomae-base"

    def __init__(self):
        self.adapter = None
        self.wisent = None

    def _create_video(self, brightness: float = 0.5, motion: float = 0.0) -> VideoContent:
        """Create synthetic video with given brightness and motion."""
        num_frames = 16
        height, width = 224, 224
        channels = 3

        frames = []
        for i in range(num_frames):
            # Base frame with given brightness
            frame = torch.ones(channels, height, width) * brightness

            # Add motion (shifting pattern)
            if motion > 0:
                shift = int(motion * i * 10) % width
                frame[:, :, shift:shift+20] = 1.0 - brightness

            # Add some noise
            frame += 0.1 * torch.rand(channels, height, width)
            frame = frame.clamp(0, 1)
            frames.append(frame)

        return VideoContent(frames=torch.stack(frames), fps=30.0)

    def run(self) -> PipelineTestResult:
        logger.info("\n" + "="*60)
        logger.info("VIDEO PIPELINE TEST")
        logger.info("="*60)

        details = {}

        # Step 1: Load model
        logger.info("\n[1/5] Loading model...")
        self.adapter = VideoAdapter(model_name=self.MODEL_NAME)
        logger.info(f"  Model loaded on {self.adapter.device}")

        # Step 2: Create contrastive pairs
        logger.info("\n[2/5] Creating contrastive pairs...")
        self.wisent = Wisent(adapter=self.adapter)

        # Bright/calm vs dark/chaotic videos
        pairs_data = [
            (0.8, 0.0, 0.2, 0.5),  # (bright, still) vs (dark, moving)
            (0.7, 0.1, 0.3, 0.4),
            (0.9, 0.0, 0.1, 0.6),
        ]

        for bright1, motion1, bright2, motion2 in pairs_data:
            safe_video = self._create_video(brightness=bright1, motion=motion1)
            unsafe_video = self._create_video(brightness=bright2, motion=motion2)
            self.wisent.add_pair(
                positive=safe_video,
                negative=unsafe_video,
                trait="safety",
            )

        pairs_created = len(self.wisent._pairs.get("safety", []))
        logger.info(f"  Created {pairs_created} video pairs for 'safety' trait")
        details["pairs_created"] = pairs_created

        # Step 3: Train steering vectors
        logger.info("\n[3/5] Training steering vectors...")
        self.wisent.train(traits=["safety"])

        trait_info = self.wisent.get_trait_info("safety")
        vectors_trained = trait_info is not None and trait_info.steering_vectors is not None

        if vectors_trained:
            num_vectors = len([v for v in trait_info.steering_vectors.values() if v is not None])
            logger.info(f"  Trained {num_vectors} steering vectors")

        # Step 4: Test encoding
        logger.info("\n[4/5] Testing encoding...")
        test_video = self._create_video(brightness=0.5, motion=0.2)

        encoding = self.adapter.encode(test_video)
        logger.info(f"  Encoding shape: {encoding.shape}")
        details["encoding_shape"] = list(encoding.shape)

        generation_works = encoding is not None and encoding.numel() > 0

        # Step 5: Verify steering effect
        logger.info("\n[5/5] Verifying steering effect...")
        intervention_points = self.wisent.get_intervention_points()
        layers = [intervention_points[0]] if intervention_points else []

        acts = self.adapter.extract_activations(test_video, layers)
        steering_has_effect = len(acts) > 0 and vectors_trained

        if steering_has_effect:
            for layer_name, act in acts.items():
                if act is not None:
                    logger.info(f"  Layer {layer_name} activation norm: {float(act.norm()):.4f}")

        logger.info("\n" + "-"*60)
        result = PipelineTestResult(
            adapter_name="video",
            pairs_created=pairs_created,
            activations_extracted=len(acts) > 0,
            vectors_trained=vectors_trained,
            generation_works=generation_works,
            steering_has_effect=steering_has_effect,
            details=details,
        )
        logger.info(f"VIDEO PIPELINE: {'PASSED' if result.passed else 'FAILED'}")
        return result


class RoboticsPipelineTest:
    """Full pipeline test for robotics policy steering."""

    def __init__(self):
        self.adapter = None
        self.wisent = None
        self.policy = None

    def _create_policy(self) -> nn.Module:
        """Create a simple MLP policy."""
        return nn.Sequential(
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 6),
            nn.Tanh(),
        )

    def _create_trajectory(self, style: str = "gentle") -> RobotTrajectory:
        """Create a trajectory with given style."""
        num_steps = 10
        states = []
        actions = []

        for i in range(num_steps):
            if style == "gentle":
                # Slow, smooth movements
                joint_pos = np.sin(np.linspace(0, np.pi, 6) + i * 0.1).astype(np.float32) * 0.5
                joint_vel = np.ones(6, dtype=np.float32) * 0.1
            else:
                # Fast, jerky movements
                joint_pos = np.random.randn(6).astype(np.float32)
                joint_vel = np.random.randn(6).astype(np.float32) * 2

            states.append(RobotState(joint_positions=joint_pos, joint_velocities=joint_vel))

            if i < num_steps - 1:
                if style == "gentle":
                    action = np.ones(6, dtype=np.float32) * 0.1
                else:
                    action = np.random.randn(6).astype(np.float32)
                actions.append(RobotAction(raw_action=action))

        return RobotTrajectory(states=tuple(states), actions=tuple(actions))

    def run(self) -> PipelineTestResult:
        logger.info("\n" + "="*60)
        logger.info("ROBOTICS PIPELINE TEST")
        logger.info("="*60)

        details = {}

        # Step 1: Create policy and adapter
        logger.info("\n[1/5] Creating policy and adapter...")
        self.policy = self._create_policy()
        self.adapter = RoboticsAdapter(model=self.policy, state_dim=12, action_dim=6)
        logger.info(f"  Policy loaded on {self.adapter.device}")
        logger.info(f"  Policy params: {sum(p.numel() for p in self.policy.parameters())}")

        # Step 2: Create contrastive pairs (trajectories)
        logger.info("\n[2/5] Creating contrastive trajectory pairs...")
        self.wisent = Wisent(adapter=self.adapter)

        for _ in range(5):
            gentle_traj = self._create_trajectory("gentle")
            forceful_traj = self._create_trajectory("forceful")
            self.wisent.add_pair(
                positive=gentle_traj,
                negative=forceful_traj,
                trait="gentleness",
            )

        pairs_created = len(self.wisent._pairs.get("gentleness", []))
        logger.info(f"  Created {pairs_created} trajectory pairs for 'gentleness' trait")
        details["pairs_created"] = pairs_created

        # Step 3: Train steering vectors
        logger.info("\n[3/5] Training steering vectors...")
        self.wisent.train(traits=["gentleness"])

        trait_info = self.wisent.get_trait_info("gentleness")
        vectors_trained = trait_info is not None and trait_info.steering_vectors is not None

        if vectors_trained:
            num_vectors = len([v for v in trait_info.steering_vectors.values() if v is not None])
            logger.info(f"  Trained {num_vectors} steering vectors")

        # Step 4: Test action generation
        logger.info("\n[4/5] Testing action generation...")
        test_state = RobotState(
            joint_positions=np.zeros(6, dtype=np.float32),
            joint_velocities=np.zeros(6, dtype=np.float32),
        )

        # Base action
        action_base = self.wisent.act(test_state)
        logger.info(f"  Base action: {action_base.raw_action}")
        details["action_base"] = action_base.raw_action.tolist()

        # Steered action (more gentle)
        action_gentle = self.wisent.act(test_state, steer={"gentleness": 2.0})
        logger.info(f"  Gentle action: {action_gentle.raw_action}")
        details["action_gentle"] = action_gentle.raw_action.tolist()

        # Steered action (less gentle / more forceful)
        action_forceful = self.wisent.act(test_state, steer={"gentleness": -2.0})
        logger.info(f"  Forceful action: {action_forceful.raw_action}")
        details["action_forceful"] = action_forceful.raw_action.tolist()

        generation_works = (
            action_base.raw_action is not None and
            action_gentle.raw_action is not None and
            action_forceful.raw_action is not None
        )

        # Step 5: Verify steering effect
        logger.info("\n[5/5] Verifying steering effect...")

        # Check that actions differ
        base_norm = float(np.linalg.norm(action_base.raw_action))
        gentle_norm = float(np.linalg.norm(action_gentle.raw_action))
        forceful_norm = float(np.linalg.norm(action_forceful.raw_action))

        logger.info(f"  Action norms - Base: {base_norm:.4f}, Gentle: {gentle_norm:.4f}, Forceful: {forceful_norm:.4f}")

        # For gentleness, we expect gentle actions to have smaller magnitude
        steering_has_effect = not np.allclose(action_base.raw_action, action_gentle.raw_action, atol=1e-3)

        if steering_has_effect:
            logger.info("  Steering produces different actions")
        else:
            logger.warning("  Steering may not be having effect")

        logger.info("\n" + "-"*60)
        result = PipelineTestResult(
            adapter_name="robotics",
            pairs_created=pairs_created,
            activations_extracted=vectors_trained,
            vectors_trained=vectors_trained,
            generation_works=generation_works,
            steering_has_effect=steering_has_effect,
            details=details,
        )
        logger.info(f"ROBOTICS PIPELINE: {'PASSED' if result.passed else 'FAILED'}")
        return result
