"""
Direct test of evalita_sp extractor.
"""
from lm_eval import tasks
from lm_eval.tasks import TaskManager
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors.evalita_sp import EvalitaSpExtractor

# Load task
task_manager = TaskManager()
task_dict = task_manager.load_task_or_group("evalita-sp_sum_task_fp-small_p1")
task_obj = list(task_dict.values())[0]

# Create extractor
extractor = EvalitaSpExtractor()

# Get one doc and extract pair
docs = list(task_obj.test_docs())
if docs:
    doc = docs[0]
    print("Testing document:")
    print(f"  source (first 100 chars): {doc['source'][:100]}...")
    print(f"  target (first 100 chars): {doc['target'][:100]}...")

    pair = extractor._extract_pair_from_doc(doc)

    if pair:
        print("\n✓ Pair extracted successfully!")
        print(f"\nPrompt (first 200 chars):\n{pair.prompt[:200]}...")
        print(f"\nPositive response (first 100 chars):\n{pair.positive_response.model_response[:100]}...")
        print(f"\nNegative response (first 100 chars):\n{pair.negative_response.model_response[:100]}...")
    else:
        print("\n✗ Failed to extract pair")
else:
    print("No docs found")
