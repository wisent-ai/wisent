"""
Apply steering vectors to model and evaluate steered output.

This script:
1. Loads trained steering vectors from previous step
2. Loads model and applies steering vectors to it
3. Generates outputs with and without steering
4. Evaluates how steering changes model behavior
"""
import os
import json
from pathlib import Path
from wisent.core.models.wisent_model import WisentModel
from wisent.core.activations.core.atoms import LayerActivations
from wisent.cli.data_loaders.data_loader_rotator import DataLoaderRotator


def get_config():
    """Read configuration from environment variables with defaults."""
    return {
        'benchmark': os.getenv('WISENT_BENCHMARK', 'gsm8k'),
        'model': os.getenv('WISENT_MODEL', 'meta-llama/Llama-3.2-1B-Instruct'),
        'training_limit': int(os.getenv('WISENT_TRAINING_LIMIT', '10')),
        'testing_limit': int(os.getenv('WISENT_TESTING_LIMIT', '2')),
        'steering_strength': float(os.getenv('WISENT_STEERING_STRENGTH', '1.0')),
        'device': os.getenv('WISENT_DEVICE', 'cpu'),
        'save_dir': os.getenv('WISENT_SAVE_DIR', './steering_output'),
    }


def load_steering_vectors(save_dir):
    """Load steering vectors from training output."""
    steering_path = Path(save_dir) / "steering_vectors.json"

    with open(steering_path, 'r') as f:
        data = json.load(f)

    # Convert back to LayerActivations format
    layers_dict = {}
    for layer_name, vector_list in data.items():
        layers_dict[layer_name] = vector_list

    return LayerActivations(layers_dict)


def generate_steered_outputs(model, test_pairs, steering_vectors, strength):
    """Generate outputs with and without steering."""
    print('  Generating unsteered outputs...')
    unsteered_outputs = []
    for pair in test_pairs:
        output = model.generate(
            prompt=pair.prompt,
            max_new_tokens=50,
            temperature=0.7
        )
        unsteered_outputs.append(output)

    print('  Applying steering vectors to model...')
    # Apply steering to model layers
    model.apply_steering(steering_vectors, strength=strength)

    print('  Generating steered outputs...')
    steered_outputs = []
    for pair in test_pairs:
        output = model.generate(
            prompt=pair.prompt,
            max_new_tokens=50,
            temperature=0.7
        )
        steered_outputs.append(output)

    # Remove steering
    model.remove_steering()

    return unsteered_outputs, steered_outputs


def main():
    config = get_config()

    print('=' * 80)
    print('STEP 2: Apply Steering to Model')
    print('=' * 80)
    print(f'Benchmark: {config["benchmark"]}')
    print(f'Model: {config["model"]}')
    print(f'Steering strength: {config["steering_strength"]}')
    print(f'Device: {config["device"]}')
    print('=' * 80)

    # Load steering vectors
    print('\n[1/4] Loading trained steering vectors...')
    steering_vectors = load_steering_vectors(config['save_dir'])
    layer_names = list(steering_vectors.keys())
    print(f'✓ Loaded steering vectors for layers: {layer_names}')

    # Load test data
    print('\n[2/4] Loading test data from benchmark...')
    data_loader_rot = DataLoaderRotator()
    data_loader_rot.use("task_interface")
    data = data_loader_rot.load(
        task=config['benchmark'],
        training_limit=config['training_limit'],
        testing_limit=config['testing_limit']
    )

    test_set = data['test_qa_pairs']
    print(f'✓ Loaded {len(test_set.pairs)} test pairs')

    # Load model
    print('\n[3/4] Loading model...')
    model = WisentModel(model_name=config['model'], device=config['device'])
    print(f'✓ Loaded {config["model"]}')

    # Generate steered outputs
    print('\n[4/4] Generating outputs with and without steering...')
    print(f'  Steering strength: {config["steering_strength"]}')

    unsteered_outputs, steered_outputs = generate_steered_outputs(
        model,
        test_set.pairs[:3],  # Use first 3 test pairs
        steering_vectors,
        strength=config['steering_strength']
    )

    # Display results
    print('\n' + '=' * 80)
    print('STEERING RESULTS')
    print('=' * 80)

    for i, (pair, unsteered, steered) in enumerate(zip(test_set.pairs[:3], unsteered_outputs, steered_outputs)):
        print(f'\n--- Example {i+1} ---')
        print(f'Prompt: {pair.prompt[:100]}...')
        print(f'\nUnsteered Output: {unsteered[:150]}...')
        print(f'\nSteered Output: {steered[:150]}...')
        print(f'\nExpected (Positive): {pair.positive_response.model_response[:100]}...')

    print('\n' + '=' * 80)
    print('STEERING APPLICATION COMPLETE')
    print('=' * 80)


if __name__ == '__main__':
    main()
