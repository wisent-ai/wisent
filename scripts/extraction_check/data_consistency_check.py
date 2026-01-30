#!/usr/bin/env python3
"""Check data consistency for extracted activations."""


def check_data_consistency(cur, model_id, model_name, expected_hidden_dim):
    """
    Check data consistency for a model's extracted activations.

    Verifies:
    - hiddenDim in RawActivation matches expected model dimension
    - neuronCount in Activation matches expected model dimension
    - No NULL hiddenStates in RawActivation
    - No NULL activationData in Activation

    Returns:
        True if all checks pass, False otherwise
    """
    print(f"\n{'='*70}")
    print(f"DATA CONSISTENCY CHECK: {model_name}")
    print(f"{'='*70}")

    issues = []

    # Check hidden dimension in RawActivation
    cur.execute('''
        SELECT DISTINCT "hiddenDim"
        FROM "RawActivation"
        WHERE "modelId" = %s AND "hiddenDim" IS NOT NULL
        LIMIT 10
    ''', (model_id,))
    raw_dims = [row[0] for row in cur.fetchall()]

    if raw_dims:
        print(f"RawActivation hiddenDim values: {raw_dims}")
        if expected_hidden_dim and expected_hidden_dim not in raw_dims:
            issues.append(f"RawActivation: expected hiddenDim {expected_hidden_dim}, found {raw_dims}")

    # Check neuron count in Activation
    cur.execute('''
        SELECT DISTINCT "neuronCount"
        FROM "Activation"
        WHERE "modelId" = %s AND "neuronCount" IS NOT NULL
        LIMIT 10
    ''', (model_id,))
    activation_dims = [row[0] for row in cur.fetchall()]

    if activation_dims:
        print(f"Activation neuronCount values: {activation_dims}")
        if expected_hidden_dim and expected_hidden_dim not in activation_dims:
            issues.append(f"Activation: expected neuronCount {expected_hidden_dim}, found {activation_dims}")

    # Check for any NULL hiddenStates in RawActivation
    cur.execute('''
        SELECT COUNT(*)
        FROM "RawActivation"
        WHERE "modelId" = %s AND "hiddenStates" IS NULL
    ''', (model_id,))
    null_raw = cur.fetchone()[0]
    if null_raw > 0:
        issues.append(f"RawActivation: {null_raw} NULL hiddenStates")

    # Check for any NULL activationData in Activation
    cur.execute('''
        SELECT COUNT(*)
        FROM "Activation"
        WHERE "modelId" = %s AND "activationData" IS NULL
    ''', (model_id,))
    null_act = cur.fetchone()[0]
    if null_act > 0:
        issues.append(f"Activation: {null_act} NULL activationData")

    if issues:
        print(f"\nIssues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print(f"\nNo data consistency issues found")

    return len(issues) == 0
