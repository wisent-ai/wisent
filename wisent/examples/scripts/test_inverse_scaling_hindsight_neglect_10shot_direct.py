from lm_eval.tasks import TaskManager
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors.inverse_scaling import InverseScalingExtractor

# Load task
task_manager = TaskManager()
task_dict = task_manager.load_task_or_group("inverse_scaling_hindsight_neglect_10shot")
task_obj = list(task_dict.values())[0]

# Create extractor
extractor = InverseScalingExtractor()

# Extract pairs
pairs = extractor.extract_contrastive_pairs(task_obj, limit=2)

print(f"Extracted {len(pairs)} pairs")
for i, pair in enumerate(pairs):
    print(f"\nPair {i+1}:")
    print(f"Prompt: {pair.prompt[:200]}...")
    print(f"Positive: {pair.positive_response.model_response}")
    print(f"Negative: {pair.negative_response.model_response}")
