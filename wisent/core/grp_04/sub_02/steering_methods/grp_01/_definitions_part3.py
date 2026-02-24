"""Steering method definitions: MLP, NURT, SZLAK, WICHER."""

from wisent.core.steering_methods.registry import (
    SteeringMethodDefinition,
    SteeringMethodParameter,
    SteeringMethodType,
)
from wisent.core.constants import (
    DEFAULT_BASE_STRENGTH,
    DEFAULT_VARIANCE_THRESHOLD,
    BROYDEN_DEFAULT_NUM_STEPS, BROYDEN_DEFAULT_ALPHA, BROYDEN_DEFAULT_ETA,
    BROYDEN_DEFAULT_BETA, BROYDEN_DEFAULT_ALPHA_DECAY,
    MLP_HIDDEN_DIM, MLP_NUM_LAYERS, MLP_OPTIMIZATION_STEPS,
    MLP_DROPOUT, MLP_LEARNING_RATE, MLP_WEIGHT_DECAY,
    NURT_TRAINING_EPOCHS, NURT_NUM_INTEGRATION_STEPS, NURT_T_MAX,
    NURT_LR, SZLAK_SINKHORN_REG, SZLAK_INFERENCE_K,
    STEERING_STRENGTH_RANGE_WIDE, STEERING_STRENGTH_RANGE_NARROW,
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
            default=MLP_HIDDEN_DIM,
            help="Hidden dimension for MLP layers",
            cli_flag="--mlp-hidden-dim",
        ),
        SteeringMethodParameter(
            name="num_layers",
            type=int,
            default=MLP_NUM_LAYERS,
            help="Number of hidden layers in MLP",
            cli_flag="--mlp-num-layers",
        ),
        SteeringMethodParameter(
            name="dropout",
            type=float,
            default=MLP_DROPOUT,
            help="Dropout rate for regularization",
            cli_flag="--mlp-dropout",
        ),
        SteeringMethodParameter(
            name="epochs",
            type=int,
            default=MLP_OPTIMIZATION_STEPS,
            help="Training epochs for MLP classifier",
            cli_flag="--mlp-epochs",
        ),
        SteeringMethodParameter(
            name="learning_rate",
            type=float,
            default=MLP_LEARNING_RATE,
            help="Learning rate for MLP training",
            cli_flag="--mlp-learning-rate",
        ),
        SteeringMethodParameter(
            name="weight_decay",
            type=float,
            default=MLP_WEIGHT_DECAY,
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
        "strength_search_range": STEERING_STRENGTH_RANGE_WIDE,
        "default_strength": DEFAULT_BASE_STRENGTH,
    },
    default_strength=DEFAULT_BASE_STRENGTH,
    strength_range=STEERING_STRENGTH_RANGE_WIDE,
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
            default=DEFAULT_VARIANCE_THRESHOLD,
            help="Cumulative variance threshold for auto dim selection",
            cli_flag="--nurt-variance-threshold",
        ),
        SteeringMethodParameter(
            name="training_epochs",
            type=int,
            default=NURT_TRAINING_EPOCHS,
            help="Training epochs for flow matching",
            cli_flag="--nurt-training-epochs",
        ),
        SteeringMethodParameter(
            name="lr",
            type=float,
            default=NURT_LR,
            help="Learning rate for AdamW optimizer",
            cli_flag="--nurt-lr",
        ),
        SteeringMethodParameter(
            name="num_integration_steps",
            type=int,
            default=NURT_NUM_INTEGRATION_STEPS,
            help="Number of Euler integration steps at inference",
            cli_flag="--nurt-num-integration-steps",
        ),
        SteeringMethodParameter(
            name="t_max",
            type=float,
            default=NURT_T_MAX,
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
        "strength_search_range": STEERING_STRENGTH_RANGE_NARROW,
        "default_strength": DEFAULT_BASE_STRENGTH,
    },
    default_strength=DEFAULT_BASE_STRENGTH,
    strength_range=STEERING_STRENGTH_RANGE_NARROW,
)


# =============================================================================

SZLAK_DEFINITION = SteeringMethodDefinition(
    name="szlak",
    method_type=SteeringMethodType.SZLAK,
    description="Attention-Transport steering via EOT cost inversion with one-sided Sinkhorn.",
    method_class_path="wisent.core.steering_methods.methods.szlak.SzlakMethod",
    parameters=[
        SteeringMethodParameter(
            name="sinkhorn_reg",
            type=float,
            default=SZLAK_SINKHORN_REG,
            help="Entropic regularization for one-sided EOT solver",
            cli_flag="--szlak-sinkhorn-reg",
        ),
        SteeringMethodParameter(
            name="inference_k",
            type=int,
            default=SZLAK_INFERENCE_K,
            help="Number of nearest source points for inference interpolation",
            cli_flag="--szlak-inference-k",
        ),
    ],
    optimization_config={
        "strength_search_range": STEERING_STRENGTH_RANGE_NARROW,
        "default_strength": DEFAULT_BASE_STRENGTH,
    },
    default_strength=DEFAULT_BASE_STRENGTH,
    strength_range=STEERING_STRENGTH_RANGE_NARROW,
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
            default=DEFAULT_VARIANCE_THRESHOLD,
            help="Cumulative variance threshold for auto dim selection",
            cli_flag="--wicher-variance-threshold",
        ),
        SteeringMethodParameter(
            name="num_steps",
            type=int,
            default=BROYDEN_DEFAULT_NUM_STEPS,
            help="Number of Broyden iterations per forward pass",
            cli_flag="--wicher-num-steps",
        ),
        SteeringMethodParameter(
            name="alpha",
            type=float,
            default=BROYDEN_DEFAULT_ALPHA,
            help="Base Tikhonov regularisation",
            cli_flag="--wicher-alpha",
        ),
        SteeringMethodParameter(
            name="eta",
            type=float,
            default=BROYDEN_DEFAULT_ETA,
            help="Step-size multiplier per Broyden iteration",
            cli_flag="--wicher-eta",
        ),
        SteeringMethodParameter(
            name="beta",
            type=float,
            default=BROYDEN_DEFAULT_BETA,
            help="EMA momentum coefficient (0 = disabled)",
            cli_flag="--wicher-beta",
        ),
        SteeringMethodParameter(
            name="alpha_decay",
            type=float,
            default=BROYDEN_DEFAULT_ALPHA_DECAY,
            help="Per-step decay factor for alpha",
            cli_flag="--wicher-alpha-decay",
        ),
    ],
    optimization_config={
        "strength_search_range": STEERING_STRENGTH_RANGE_NARROW,
        "default_strength": DEFAULT_BASE_STRENGTH,
    },
    default_strength=DEFAULT_BASE_STRENGTH,
    strength_range=STEERING_STRENGTH_RANGE_NARROW,
)
