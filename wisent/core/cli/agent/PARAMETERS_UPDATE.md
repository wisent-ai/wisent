# Agent Parameters Update Plan

## New Parameters Added to Parser (agent_parser.py)

### Synthetic Pair Generation
- `--num-pairs` - Override time-budget-based calculation
- `--similarity-threshold` - Deduplication threshold
- `--max-workers` - Parallel workers

### Activation Collection
- `--token-aggregation` - average/final/first/max/min (default: average)
- `--prompt-strategy` - chat_template/direct_completion/etc (default: chat_template)
- `--normalize-layers` - Flag to normalize activations
- `--return-full-sequence` - Flag to return full sequence

### Classifier Training
- `--classifier-epochs` - Training epochs (default: 50) ✓ DONE
- `--classifier-lr` - Learning rate (default: 1e-3) ✓ DONE
- `--classifier-batch-size` - Batch size (default: adaptive) ✓ DONE
- `--classifier-type` - logistic/mlp (default: logistic)

## Files to Update

### 1. step1_generate_pairs.py ✓ DONE
- Added num_pairs, similarity_threshold, max_workers parameters
- Updated logic to use these parameters

### 2. step2_train_classifier.py - NEEDS UPDATE
Add parameters:
- token_aggregation
- prompt_strategy
- normalize_layers
- return_full_sequence
- classifier_type

### 3. step3_evaluate_response.py - NEEDS UPDATE
Add parameters:
- token_aggregation
- prompt_strategy
- normalize_layers
- return_full_sequence

### 4. step4_apply_steering.py - NEEDS UPDATE
Add parameters:
- token_aggregation
- prompt_strategy
- normalize_layers
- return_full_sequence
- steering_method (currently hardcoded to CAA)

### 5. main.py - NEEDS COMPLETE UPDATE
Pass all parameters through to step functions using getattr()

## Mapping CLI names to Python enums

### ActivationAggregationStrategy
- "average" -> ActivationAggregationStrategy.MEAN_POOLING
- "final" -> ActivationAggregationStrategy.LAST_TOKEN
- "first" -> ActivationAggregationStrategy.FIRST_TOKEN
- "max" -> ActivationAggregationStrategy.MAX_POOLING
- "min" -> ActivationAggregationStrategy.MIN_POOLING

### PromptConstructionStrategy
- "chat_template" -> PromptConstructionStrategy.CHAT_TEMPLATE
- "direct_completion" -> PromptConstructionStrategy.DIRECT_COMPLETION
- "instruction_following" -> PromptConstructionStrategy.INSTRUCTION_FOLLOWING
- "multiple_choice" -> PromptConstructionStrategy.MULTIPLE_CHOICE
- "role_playing" -> PromptConstructionStrategy.ROLE_PLAYING

### Classifier Type
- "logistic" -> LogisticClassifier
- "mlp" -> MLPClassifier

### Steering Method (currently only CAA supported)
- "CAA" -> CAAMethod
- Others not yet implemented in agent
