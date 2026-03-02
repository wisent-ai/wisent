"""Validated default parameters for each steering method.

Empty until we have empirically validated configs from actual
training runs. Every method requires explicit parameters from
the user or a saved optimizer config (model+task overrides in
~/.wisent/configs/). Attempting to instantiate a method without
providing its required parameters will raise a ValueError.

Auto-computed parameters (sensor_layer, steering_layers, etc.)
and boolean flags (normalize, use_caa_init, etc.) are handled
by inline defaults on SteeringMethodParameter, not here.
"""

VALIDATED_METHOD_DEFAULTS: dict[str, dict] = {}
