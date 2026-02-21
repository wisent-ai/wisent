"""Steering method definitions: CAA, OSTRZE, TECZA."""

from wisent.core.steering_methods.registry import (
    SteeringMethodDefinition,
    SteeringMethodParameter,
    SteeringMethodType,
)


CAA_DEFINITION = SteeringMethodDefinition(
    name="caa",
    method_type=SteeringMethodType.CAA,
    description="Contrastive Activation Addition - computes mean(positive) - mean(negative) steering vectors",
    method_class_path="wisent.core.steering_methods.methods.caa.CAAMethod",
    parameters=[
        SteeringMethodParameter(
            name="normalize",
            type=bool,
            default=True,
            help="L2-normalize the steering vector",
            action="store_true",
            cli_flag="--caa-normalize",
        ),
    ],
    optimization_config={
        "strength_search_range": (0.1, 5.0),
        "default_strength": 1.0,
    },
    default_strength=1.0,
    strength_range=(0.1, 5.0),
)


OSTRZE_DEFINITION = SteeringMethodDefinition(
    name="ostrze",
    method_type=SteeringMethodType.OSTRZE,
    description="Classifier-based steering using logistic regression decision boundary. Works better than CAA when geometry is orthogonal (each pair has unique direction rather than shared direction).",
    method_class_path="wisent.core.steering_methods.methods.ostrze.OstrzeMethod",
    parameters=[
        SteeringMethodParameter(
            name="normalize",
            type=bool,
            default=True,
            help="L2-normalize the steering vector",
            action="store_true",
            cli_flag="--ostrze-normalize",
        ),
        SteeringMethodParameter(
            name="C",
            type=float,
            default=1.0,
            help="Regularization strength (inverse). Smaller values = stronger regularization.",
            cli_flag="--ostrze-C",
        ),
    ],
    optimization_config={
        "strength_search_range": (0.1, 5.0),
        "default_strength": 1.0,
    },
    default_strength=1.0,
    strength_range=(0.1, 5.0),
)


TECZA_DEFINITION = SteeringMethodDefinition(
    name="tecza",
    method_type=SteeringMethodType.TECZA,
    description="TECZA - Projected Representations for Independent Steering Manifolds. Gradient-optimized multi-directional steering.",
    method_class_path="wisent.core.steering_methods.methods.tecza.TECZAMethod",
    parameters=[
        SteeringMethodParameter(
            name="num_directions",
            type=int,
            default=3,
            help="Number of directions to discover per layer",
            cli_flag="--tecza-num-directions",
        ),
        SteeringMethodParameter(
            name="optimization_steps",
            type=int,
            default=100,
            help="Number of gradient descent steps for direction optimization",
            cli_flag="--tecza-optimization-steps",
        ),
        SteeringMethodParameter(
            name="learning_rate",
            type=float,
            default=0.01,
            help="Learning rate for direction optimization",
            cli_flag="--tecza-learning-rate",
        ),
        SteeringMethodParameter(
            name="retain_weight",
            type=float,
            default=0.1,
            help="Weight for retain loss (preserving behavior on harmless examples)",
            cli_flag="--tecza-retain-weight",
        ),
        SteeringMethodParameter(
            name="independence_weight",
            type=float,
            default=0.05,
            help="Weight for representational independence loss between directions",
            cli_flag="--tecza-independence-weight",
        ),
        SteeringMethodParameter(
            name="normalize",
            type=bool,
            default=True,
            help="L2-normalize the final directions",
            action="store_true",
            cli_flag="--tecza-normalize",
        ),
        SteeringMethodParameter(
            name="use_caa_init",
            type=bool,
            default=True,
            help="Initialize first direction using CAA (difference-in-means)",
            action="store_true",
            cli_flag="--tecza-use-caa-init",
        ),
        SteeringMethodParameter(
            name="cone_constraint",
            type=bool,
            default=True,
            help="Constrain directions to form a polyhedral cone",
            action="store_true",
            cli_flag="--tecza-cone-constraint",
        ),
        SteeringMethodParameter(
            name="min_cosine_similarity",
            type=float,
            default=0.3,
            help="Minimum cosine similarity between directions",
            cli_flag="--tecza-min-cosine-similarity",
        ),
        SteeringMethodParameter(
            name="max_cosine_similarity",
            type=float,
            default=0.95,
            help="Maximum cosine similarity between directions (avoid redundancy)",
            cli_flag="--tecza-max-cosine-similarity",
        ),
    ],
    optimization_config={
        "strength_search_range": (0.1, 3.0),
        "default_strength": 1.0,
        "num_directions_range": (1, 7),
    },
    default_strength=1.0,
    strength_range=(0.1, 3.0),
)
