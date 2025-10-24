"""
Train classifier on contrastive pairs using wisent infrastructure.

This script:
1. Loads contrastive pairs from TaskInterface
2. Trains steering vectors using WisentSteeringTrainer with CAA method
3. Saves trained steering vectors for later use
"""
import os
from wisent.core.trainers.steering_trainer import WisentSteeringTrainer
from wisent.core.models.wisent_model import WisentModel
from wisent.core.data_loaders.rotator import DataLoaderRotator
from wisent.core.steering_methods.rotator import SteeringMethodRotator


def get_config():
    """Read configuration from environment variables with defaults."""
    return {
        'benchmark': os.getenv('WISENT_BENCHMARK', 'gsm8k'),
        'model': os.getenv('WISENT_MODEL', 'meta-llama/Llama-3.2-1B-Instruct'),
        'training_limit': int(os.getenv('WISENT_TRAINING_LIMIT', '10')),
        'testing_limit': int(os.getenv('WISENT_TESTING_LIMIT', '2')),
        'layers_spec': os.getenv('WISENT_LAYERS_SPEC', '8'),
        'device': os.getenv('WISENT_DEVICE', 'cpu'),
        'save_dir': os.getenv('WISENT_SAVE_DIR', './steering_output'),
    }


def main():
    config = get_config()

    print('=' * 80)
    print('STEP 1: Train Classifier on Contrastive Pairs')
    print('=' * 80)
    print(f'Benchmark: {config["benchmark"]}')
    print(f'Model: {config["model"]}')
    print(f'Layers: {config["layers_spec"]}')
    print(f'Device: {config["device"]}')
    print('=' * 80)

    # Load data from TaskInterface
    print('\n[1/4] Loading data from benchmark...')
    data_loader_rot = DataLoaderRotator()
    data_loader_rot.use("task_interface")
    data = data_loader_rot.load(
        task=config['benchmark'],
        training_limit=config['training_limit'],
        testing_limit=config['testing_limit']
    )

    train_set = data['train_qa_pairs']
    test_set = data['test_qa_pairs']

    print(f'✓ Loaded {len(train_set.pairs)} training, {len(test_set.pairs)} test pairs')

    # Load model
    print('\n[2/4] Loading model...')
    model = WisentModel(model_name=config['model'], device=config['device'])
    print(f'✓ Loaded {config["model"]}')

    # Get steering method (CAA)
    print('\n[3/4] Initializing CAA steering method...')
    steering_rot = SteeringMethodRotator()
    steering_rot.use("caa")
    caa_method = steering_rot._method
    print('✓ CAA method initialized')

    # Train steering vectors (classifier)
    print('\n[4/4] Training classifier (steering vectors)...')
    trainer = WisentSteeringTrainer(
        model=model,
        pair_set=train_set,
        steering_method=caa_method,
        store_device=config['device']
    )

    training_result = trainer.run(
        layers_spec=config['layers_spec'],
        aggregation="continuation_token",
        return_full_sequence=False,
        normalize_layers=True,
        save_dir=config['save_dir']
    )

    print('✓ Training complete')
    print(f'  Layers trained: {list(training_result.metadata.get("layers", {}).keys())}')
    print(f'  Steering vectors saved to: {config["save_dir"]}')

    print('\n' + '=' * 80)
    print('CLASSIFIER TRAINING COMPLETE')
    print('=' * 80)


if __name__ == '__main__':
    main()
