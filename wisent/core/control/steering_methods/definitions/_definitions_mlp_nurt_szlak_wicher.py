"""Steering method definitions: MLP, NURT, SZLAK, WICHER."""

from wisent.core.control.steering_methods.registry.registry import (
    SteeringMethodDefinition,
    SteeringMethodParameter,
    SteeringMethodType,
)


MLP_DEFINITION = SteeringMethodDefinition(
    name="mlp",
    method_type=SteeringMethodType.MLP,
    description="MLP-based steering using adversarial gradient direction from trained classifier. Captures non-linear decision boundaries.",
    method_class_path="wisent.core.control.steering_methods.methods.mlp.MLPMethod",
    parameters=[
        SteeringMethodParameter(
            name="hidden_dim",
            type=int,
            help="Hidden dimension for MLP layers",
            cli_flag="--mlp-hidden-dim",
        ),
        SteeringMethodParameter(
            name="num_layers",
            type=int,
            help="Number of hidden layers in MLP",
            cli_flag="--mlp-num-layers",
        ),
        SteeringMethodParameter(
            name="dropout",
            type=float,
            help="Dropout rate for regularization",
            cli_flag="--mlp-dropout",
        ),
        SteeringMethodParameter(
            name="epochs",
            type=int,
            help="Training epochs for MLP classifier",
            cli_flag="--mlp-epochs",
        ),
        SteeringMethodParameter(
            name="learning_rate",
            type=float,
            help="Learning rate for MLP training",
            cli_flag="--mlp-learning-rate",
        ),
        SteeringMethodParameter(
            name="weight_decay",
            type=float,
            help="Weight decay for regularization",
            cli_flag="--mlp-weight-decay",
        ),
        SteeringMethodParameter(
            name="early_stop_tol",
            type=float,
            help="Early stopping tolerance for training convergence",
            cli_flag="--mlp-early-stop-tol",
        ),
        SteeringMethodParameter(
            name="normalize",
            type=bool,
            default=True, required=False,
            help="L2-normalize the steering vector",
            action="store_true",
            cli_flag="--mlp-normalize",
        ),
        SteeringMethodParameter(
            name="mlp_input_divisor",
            type=int,
            help="Divisor to reduce hidden_dim for input projection",
            cli_flag="--mlp-input-divisor",
        ),
        SteeringMethodParameter(
            name="gating_hidden_dim_divisor",
            type=int,
            help="Divisor for gating layer hidden dimension",
            cli_flag="--mlp-gating-hidden-dim-divisor",
        ),
        SteeringMethodParameter(
            name="mlp_early_stopping_patience",
            type=int,
            help="Early stopping patience (epochs without improvement)",
            cli_flag="--mlp-early-stopping-patience",
        ),
    ],
)


NURT_DEFINITION = SteeringMethodDefinition(
    name="nurt",
    method_type=SteeringMethodType.NURT,
    description="Concept Flow - Flow matching in SVD-derived concept subspace. Learns on-manifold transport between contrastive distributions.",
    method_class_path="wisent.core.control.steering_methods.methods.nurt.NurtMethod",
    parameters=[
        SteeringMethodParameter(
            name="num_dims",
            type=int,
            default=0, required=False,
            help="Number of concept subspace dims (0 = auto from variance)",
            cli_flag="--nurt-num-dims",
        ),
        SteeringMethodParameter(
            name="variance_threshold",
            type=float,
            help="Cumulative variance threshold for auto dim selection",
            cli_flag="--nurt-variance-threshold",
        ),
        SteeringMethodParameter(
            name="training_epochs",
            type=int,
            help="Training epochs for flow matching",
            cli_flag="--nurt-training-epochs",
        ),
        SteeringMethodParameter(
            name="lr",
            type=float,
            help="Learning rate for AdamW optimizer",
            cli_flag="--nurt-lr",
        ),
        SteeringMethodParameter(
            name="num_integration_steps",
            type=int,
            help="Number of Euler integration steps at inference",
            cli_flag="--nurt-num-integration-steps",
        ),
        SteeringMethodParameter(
            name="t_max",
            type=float,
            help="Integration endpoint (controls max steering strength)",
            cli_flag="--nurt-t-max",
        ),
        SteeringMethodParameter(
            name="flow_hidden_dim",
            type=int,
            default=0, required=False,
            help="Velocity network hidden dim (0 = auto from concept_dim)",
            cli_flag="--nurt-hidden-dim",
        ),
        SteeringMethodParameter(name="max_concept_dim", type=int,
            help="Maximum concept subspace dimensionality",
            cli_flag="--nurt-max-concept-dim"),
        SteeringMethodParameter(name="lr_min", type=float,
            help="Minimum learning rate for cosine annealing scheduler",
            cli_flag="--nurt-lr-min"),
        SteeringMethodParameter(name="weight_decay", type=float,
            help="Weight decay for AdamW optimizer",
            cli_flag="--nurt-weight-decay"),
        SteeringMethodParameter(name="max_grad_norm", type=float,
            help="Maximum gradient norm for clipping",
            cli_flag="--nurt-max-grad-norm"),
    ],
)


# =============================================================================

SZLAK_DEFINITION = SteeringMethodDefinition(
    name="szlak",
    method_type=SteeringMethodType.SZLAK,
    description="Attention-Transport steering via EOT cost inversion with one-sided Sinkhorn.",
    method_class_path="wisent.core.control.steering_methods.methods.szlak.SzlakMethod",
    parameters=[
        SteeringMethodParameter(
            name="sinkhorn_reg",
            type=float,
            help="Entropic regularization for one-sided EOT solver",
            cli_flag="--szlak-sinkhorn-reg",
        ),
        SteeringMethodParameter(
            name="inference_k",
            type=int,
            help="Number of nearest source points for inference interpolation",
            cli_flag="--szlak-inference-k",
        ),
        SteeringMethodParameter(name="max_iter", type=int,
            help="Maximum Sinkhorn iterations",
            cli_flag="--szlak-max-iter"),
    ],
)


WICHER_DEFINITION = SteeringMethodDefinition(
    name="wicher",
    method_type=SteeringMethodType.WICHER,
    description="WICHER — subspace-projected Broyden steering via low-rank SVD concept basis with adaptive regularization.",
    method_class_path="wisent.core.control.steering_methods.methods.wicher.WicherMethod",
    parameters=[
        SteeringMethodParameter(
            name="concept_dim",
            type=int,
            default=0, required=False,
            help="Concept subspace dimensionality (0 = auto from variance)",
            cli_flag="--wicher-concept-dim",
        ),
        SteeringMethodParameter(
            name="variance_threshold",
            type=float,
            help="Cumulative variance threshold for auto dim selection",
            cli_flag="--wicher-variance-threshold",
        ),
        SteeringMethodParameter(
            name="num_steps",
            type=int,
            help="Number of Broyden iterations per forward pass",
            cli_flag="--wicher-num-steps",
        ),
        SteeringMethodParameter(
            name="alpha",
            type=float,
            help="Base Tikhonov regularisation",
            cli_flag="--wicher-alpha",
        ),
        SteeringMethodParameter(
            name="eta",
            type=float,
            help="Step-size multiplier per Broyden iteration",
            cli_flag="--wicher-eta",
        ),
        SteeringMethodParameter(
            name="beta",
            type=float,
            help="EMA momentum coefficient (0 = disabled)",
            cli_flag="--wicher-beta",
        ),
        SteeringMethodParameter(
            name="alpha_decay",
            type=float,
            help="Per-step decay factor for alpha",
            cli_flag="--wicher-alpha-decay",
        ),
    ],
)
