"""Verify three-layer config merge for all steering methods.

Exercises: validated defaults (empty), inline defaults, user overrides,
missing-param error, and method instantiation behavior.
"""
import sys

SEP = "=" * len("ALL CHECKS PASSED")


def run_checks():
    """Run all config merge verification checks."""
    from wisent.core.control.steering_methods.configs.validated_defaults import (
        VALIDATED_METHOD_DEFAULTS,
    )
    from wisent.core.control.steering_methods.configs import (
        load_validated_method_defaults,
    )
    from wisent.core.control.steering_methods.registry.registry import (
        SteeringMethodParameter,
        SteeringMethodRegistry,
        SteeringMethodDefinition,
        SteeringMethodType,
        get_steering_method,
        _UNSET,
    )
    from wisent.core.control.steering_methods.registry.registry_instantiation import (
        build_merged_params,
    )

    # Validated defaults should be empty (no arbitrary defaults)
    assert not VALIDATED_METHOD_DEFAULTS, (
        f"Expected empty validated defaults, got {len(VALIDATED_METHOD_DEFAULTS)} entries"
    )
    print("PASS: validated defaults is empty (no arbitrary defaults)")

    # load returns empty dict for any method
    assert load_validated_method_defaults("grom") == {}
    assert load_validated_method_defaults("nonexistent") == {}
    print("PASS: load_validated_method_defaults returns empty for all methods")

    # SteeringMethodParameter required + has_default
    p = SteeringMethodParameter(name="x", type=int, help="x")
    assert p.required is True and not p.has_default
    p2 = SteeringMethodParameter(name="y", type=bool, help="y", default=True, required=False)
    assert not p2.required and p2.has_default and p2.default is True
    print("PASS: SteeringMethodParameter.required and has_default")

    # get_default_params only returns params with has_default (booleans, auto-computed)
    grom_def = SteeringMethodRegistry.get("grom")
    inline = grom_def.get_default_params()
    assert "normalize" in inline, "boolean flags should have inline defaults"
    assert "num_directions" not in inline, "tunable params should NOT have inline defaults"
    print(f"PASS: get_default_params has {len(inline)} inline defaults (booleans/auto only)")

    # get_required_param_names identifies all tunable params as required
    req = grom_def.get_required_param_names()
    assert "num_directions" in req
    assert "learning_rate" in req
    assert "behavior_weight" in req
    assert "normalize" not in req, "boolean with default should not be required"
    print(f"PASS: get_required_param_names: {req}")

    # Without validated defaults, building params for grom with no user kwargs fails
    try:
        build_merged_params(grom_def, None, None, {})
        print("FAIL: grom should fail without explicit params")
        return False
    except ValueError as e:
        assert "num_directions" in str(e)
        print(f"PASS: grom fails without explicit params: {e}")

    # CAA should still work (only has normalize which is required=False with default)
    caa_def = SteeringMethodRegistry.get("caa")
    caa_params = build_merged_params(caa_def, None, None, {})
    assert "normalize" in caa_params
    print(f"PASS: caa works with no params (only boolean defaults): {caa_params}")

    # Ostrze needs C (required=True, no default) so it should fail
    ostrze_def = SteeringMethodRegistry.get("ostrze")
    try:
        build_merged_params(ostrze_def, None, None, {})
        print("FAIL: ostrze should fail without C param")
        return False
    except ValueError as e:
        assert "C" in str(e)
        print(f"PASS: ostrze fails without explicit C: {e}")

    # Providing all required params explicitly works
    # Derive test values from param definitions: type(required) casts True to valid value
    test_kwargs = {
        p.name: p.type(p.required)
        for p in grom_def.parameters
        if p.required and not p.has_default
    }
    grom_params = build_merged_params(grom_def, None, None, test_kwargs)
    assert "num_directions" in grom_params
    assert "normalize" in grom_params  # from inline default
    print(f"PASS: grom works with all explicit params ({len(grom_params)} total)")

    # CAA instantiation works
    caa = get_steering_method("caa")
    print(f"PASS: caa instantiates -> {type(caa).__name__}")

    # Every method with required params fails without them
    methods_that_should_fail = [
        "ostrze", "tecza", "tetno", "grom",
        "mlp", "nurt", "szlak", "wicher", "przelom",
    ]
    for name in methods_that_should_fail:
        try:
            get_steering_method(name)
            print(f"  FAIL: {name} should have failed without params")
            return False
        except ValueError as e:
            print(f"  PASS: {name} fails correctly -> {e}")

    return True


if __name__ == "__main__":
    print(SEP)
    ok = run_checks()
    print(SEP)
    if ok:
        print("ALL CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED")
    sys.exit(not ok)
