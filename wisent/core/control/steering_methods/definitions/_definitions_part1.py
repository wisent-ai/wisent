"""Steering method definitions: CAA, OSTRZE, TECZA."""

from wisent.core.utils.config_tools.constants import (
    STEERING_STRENGTH_RANGE_WIDE,
    STEERING_STRENGTH_RANGE_NARROW,
)
from wisent.core.control.steering_methods.registry.registry import (
    SteeringMethodDefinition,
    SteeringMethodParameter,
    SteeringMethodType,
)


CAA_DEFINITION = SteeringMethodDefinition(
    name="caa",
    method_type=SteeringMethodType.CAA,
    description="Contrastive Activation Addition - computes mean(positive) - mean(negative) steering vectors",
    method_class_path="wisent.core.control.steering_methods.methods.caa.CAAMethod",
    parameters=[
        SteeringMethodParameter(
            name="normalize",
            type=bool,
            default=True, required=False,
            help="L2-normalize the steering vector",
            action="store_true",
            cli_flag="--caa-normalize",
        ),
    ],
    optimization_config={
        "strength_search_range": STEERING_STRENGTH_RANGE_WIDE,
    },
    strength_range=STEERING_STRENGTH_RANGE_WIDE,
)


OSTRZE_DEFINITION = SteeringMethodDefinition(
    name="ostrze",
    method_type=SteeringMethodType.OSTRZE,
    description="Classifier-based steering using logistic regression decision boundary. Works better than CAA when geometry is orthogonal (each pair has unique direction rather than shared direction).",
    method_class_path="wisent.core.control.steering_methods.methods.ostrze.OstrzeMethod",
    parameters=[
        SteeringMethodParameter(
            name="normalize",
            type=bool,
            default=True, required=False,
            help="L2-normalize the steering vector",
            action="store_true",
            cli_flag="--ostrze-normalize",
        ),
        SteeringMethodParameter(
            name="C",
            type=float,
            help="Regularization strength (inverse). Smaller values = stronger regularization.",
            cli_flag="--ostrze-C",
        ),
    ],
    optimization_config={
        "strength_search_range": STEERING_STRENGTH_RANGE_WIDE,
    },
    strength_range=STEERING_STRENGTH_RANGE_WIDE,
)


TECZA_DEFINITION = SteeringMethodDefinition(
    name="tecza",
    method_type=SteeringMethodType.TECZA,
    description="TECZA - Projected Representations for Independent Steering Manifolds. Gradient-optimized multi-directional steering.",
    method_class_path="wisent.core.control.steering_methods.methods.tecza.TECZAMethod",
    parameters=[
        SteeringMethodParameter(
            name="num_directions",
            type=int,
            help="Number of directions to discover per layer",
            cli_flag="--tecza-num-directions",
        ),
        SteeringMethodParameter(
            name="optimization_steps",
            type=int,
            help="Number of gradient descent steps for direction optimization",
            cli_flag="--tecza-optimization-steps",
        ),
        SteeringMethodParameter(
            name="learning_rate",
            type=float,
            help="Learning rate for direction optimization",
            cli_flag="--tecza-learning-rate",
        ),
        SteeringMethodParameter(
            name="retain_weight",
            type=float,
            help="Weight for retain loss (preserving behavior on harmless examples)",
            cli_flag="--tecza-retain-weight",
        ),
        SteeringMethodParameter(
            name="independence_weight",
            type=float,
            help="Weight for representational independence loss between directions",
            cli_flag="--tecza-independence-weight",
        ),
        SteeringMethodParameter(
            name="normalize",
            type=bool,
            default=True, required=False,
            help="L2-normalize the final directions",
            action="store_true",
            cli_flag="--tecza-normalize",
        ),
        SteeringMethodParameter(
            name="use_caa_init",
            type=bool,
            default=True, required=False,
            help="Initialize first direction using CAA (difference-in-means)",
            action="store_true",
            cli_flag="--tecza-use-caa-init",
        ),
        SteeringMethodParameter(
            name="cone_constraint",
            type=bool,
            default=True, required=False,
            help="Constrain directions to form a polyhedral cone",
            action="store_true",
            cli_flag="--tecza-cone-constraint",
        ),
        SteeringMethodParameter(
            name="min_cosine_similarity",
            type=float,
            help="Minimum cosine similarity between directions",
            cli_flag="--tecza-min-cosine-similarity",
        ),
        SteeringMethodParameter(
            name="max_cosine_similarity",
            type=float,
            help="Maximum cosine similarity between directions (avoid redundancy)",
            cli_flag="--tecza-max-cosine-similarity",
        ),
    ],
    optimization_config={
        "strength_search_range": STEERING_STRENGTH_RANGE_NARROW,
        "num_directions_range": (1, 7),
    },
    strength_range=STEERING_STRENGTH_RANGE_NARROW,
)
