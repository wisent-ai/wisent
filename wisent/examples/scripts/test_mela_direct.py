from lm_eval.tasks import TaskManager
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors.mela import MelaExtractor

task_manager = TaskManager()
# Load a specific subtask instead of the group
task_dict = task_manager.load_task_or_group("mela_en")
task_obj = list(task_dict.values())[0]

print(f"Testing mela_en")
print(f"Task type: {type(task_obj)}")

extractor = MelaExtractor()
pairs = extractor.extract_contrastive_pairs(task_obj, limit=2)

print(f"Extracted {len(pairs)} pairs")
for i, pair in enumerate(pairs):
    print(f"\nPair {i+1}:")
    print(f"Prompt: {pair.prompt[:200]}...")
    print(f"Positive: {pair.positive_response.model_response}")
    print(f"Negative: {pair.negative_response.model_response}")
