"""Steering method definitions: MLP, NURT, SZLAK, WICHER."""

from wisent.core.steering_methods.registry import (
    SteeringMethodDefinition,
    SteeringMethodParameter,
    SteeringMethodType,
)


MLP_DEFINITION = SteeringMethodDefinition(
    name="mlp",
    method_type=SteeringMethodType.MLP,
    description="MLP-based steering using adversarial gradient direction from trained classifier. Captures non-linear decision boundaries.",
    method_class_path="wisent.core.steering_methods.methods.mlp.MLPMethod",
    parameters=[
        SteeringMethodParameter(
            name="hidden_dim",
            type=int,
            default=256,
            help="Hidden dimension for MLP layers",
            cli_flag="--mlp-hidden-dim",
        ),
        SteeringMethodParameter(
            name="num_layers",
            type=int,
            default=2,
            help="Number of hidden layers in MLP",
            cli_flag="--mlp-num-layers",
        ),
        SteeringMethodParameter(
            name="dropout",
            type=float,
            default=0.1,
            help="Dropout rate for regularization",
            cli_flag="--mlp-dropout",
        ),
        SteeringMethodParameter(
            name="epochs",
            type=int,
            default=100,
            help="Training epochs for MLP classifier",
            cli_flag="--mlp-epochs",
        ),
        SteeringMethodParameter(
            name="learning_rate",
            type=float,
            default=0.001,
            help="Learning rate for MLP training",
            cli_flag="--mlp-learning-rate",
        ),
        SteeringMethodParameter(
            name="weight_decay",
            type=float,
            default=0.01,
            help="Weight decay for regularization",
            cli_flag="--mlp-weight-decay",
        ),
        SteeringMethodParameter(
            name="normalize",
            type=bool,
            default=True,
            help="L2-normalize the steering vector",
            action="store_true",
            cli_flag="--mlp-normalize",
        ),
    ],
    optimization_config={
        "strength_search_range": (0.1, 5.0),
        "default_strength": 1.0,
    },
    default_strength=1.0,
    strength_range=(0.1, 5.0),
)


NURT_DEFINITION = SteeringMethodDefinition(
    name="nurt",
    method_type=SteeringMethodType.NURT,
    description="Concept Flow - Flow matching in SVD-derived concept subspace. Learns on-manifold transport between contrastive distributions.",
    method_class_path="wisent.core.steering_methods.methods.nurt.NurtMethod",
    parameters=[
        SteeringMethodParameter(
            name="num_dims",
            type=int,
            default=0,
            help="Number of concept subspace dims (0 = auto from variance)",
            cli_flag="--nurt-num-dims",
        ),
        SteeringMethodParameter(
            name="variance_threshold",
            type=float,
            default=0.80,
            help="Cumulative variance threshold for auto dim selection",
            cli_flag="--nurt-variance-threshold",
        ),
        SteeringMethodParameter(
            name="training_epochs",
            type=int,
            default=300,
            help="Training epochs for flow matching",
            cli_flag="--nurt-training-epochs",
        ),
        SteeringMethodParameter(
            name="lr",
            type=float,
            default=0.001,
            help="Learning rate for AdamW optimizer",
            cli_flag="--nurt-lr",
        ),
        SteeringMethodParameter(
            name="num_integration_steps",
            type=int,
            default=4,
            help="Number of Euler integration steps at inference",
            cli_flag="--nurt-num-integration-steps",
        ),
        SteeringMethodParameter(
            name="t_max",
            type=float,
            default=1.0,
            help="Integration endpoint (controls max steering strength)",
            cli_flag="--nurt-t-max",
        ),
        SteeringMethodParameter(
            name="flow_hidden_dim",
            type=int,
            default=0,
            help="Velocity network hidden dim (0 = auto from concept_dim)",
            cli_flag="--nurt-hidden-dim",
        ),
    ],
    optimization_config={
        "strength_search_range": (0.1, 3.0),
        "default_strength": 1.0,
    },
    default_strength=1.0,
    strength_range=(0.1, 3.0),
)


# =============================================================================

SZLAK_DEFINITION = SteeringMethodDefinition(
    name="szlak",
    method_type=SteeringMethodType.SZLAK,
    description="Geodesic Optimal Transport - manifold-aware transport via k-NN geodesic distances and Sinkhorn OT.",
    method_class_path="wisent.core.steering_methods.methods.szlak.SzlakMethod",
    parameters=[
        SteeringMethodParameter(
            name="k_neighbors",
            type=int,
            default=10,
            help="Number of nearest neighbors for k-NN graph construction",
            cli_flag="--szlak-k-neighbors",
        ),
        SteeringMethodParameter(
            name="sinkhorn_reg",
            type=float,
            default=0.1,
            help="Entropic regularization for Sinkhorn solver",
            cli_flag="--szlak-sinkhorn-reg",
        ),
        SteeringMethodParameter(
            name="sinkhorn_max_iter",
            type=int,
            default=100,
            help="Maximum iterations for Sinkhorn convergence",
            cli_flag="--szlak-sinkhorn-max-iter",
        ),
        SteeringMethodParameter(
            name="inference_k",
            type=int,
            default=5,
            help="Number of nearest source points for inference interpolation",
            cli_flag="--szlak-inference-k",
        ),
        SteeringMethodParameter(
            name="cost_mode",
            type=str,
            default="geodesic",
            help="Cost computation mode: geodesic (k-NN shortest path) or attention_affinity (EOT dot-product)",
            cli_flag="--szlak-cost-mode",
            choices=["geodesic", "attention_affinity"],
        ),
    ],
    optimization_config={
        "strength_search_range": (0.1, 3.0),
        "default_strength": 1.0,
    },
    default_strength=1.0,
    strength_range=(0.1, 3.0),
)


WICHER_DEFINITION = SteeringMethodDefinition(
    name="wicher",
    method_type=SteeringMethodType.WICHER,
    description="WICHER — subspace-projected Broyden steering via low-rank SVD concept basis with adaptive regularization.",
    method_class_path="wisent.core.steering_methods.methods.wicher.WicherMethod",
    parameters=[
        SteeringMethodParameter(
            name="concept_dim",
            type=int,
            default=0,
            help="Concept subspace dimensionality (0 = auto from variance)",
            cli_flag="--wicher-concept-dim",
        ),
        SteeringMethodParameter(
            name="variance_threshold",
            type=float,
            default=0.80,
            help="Cumulative variance threshold for auto dim selection",
            cli_flag="--wicher-variance-threshold",
        ),
        SteeringMethodParameter(
            name="num_steps",
            type=int,
            default=3,
            help="Number of Broyden iterations per forward pass",
            cli_flag="--wicher-num-steps",
        ),
        SteeringMethodParameter(
            name="alpha",
            type=float,
            default=5e-3,
            help="Base Tikhonov regularisation",
            cli_flag="--wicher-alpha",
        ),
        SteeringMethodParameter(
            name="eta",
            type=float,
            default=0.5,
            help="Step-size multiplier per Broyden iteration",
            cli_flag="--wicher-eta",
        ),
        SteeringMethodParameter(
            name="beta",
            type=float,
            default=0.0,
            help="EMA momentum coefficient (0 = disabled)",
            cli_flag="--wicher-beta",
        ),
        SteeringMethodParameter(
            name="alpha_decay",
            type=float,
            default=1.0,
            help="Per-step decay factor for alpha",
            cli_flag="--wicher-alpha-decay",
        ),
    ],
    optimization_config={
        "strength_search_range": (0.1, 3.0),
        "default_strength": 1.0,
    },
    default_strength=1.0,
    strength_range=(0.1, 3.0),
)
