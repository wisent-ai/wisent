from wisent_guard.core.contrastive_pairs.lm_eval_pairs.lm_extractor_registry import get_extractor
from lm_eval import tasks

task_dict = tasks.get_task_dict(["cb"])
boolq_task = task_dict["cb"]
extractor = get_extractor("cb")
preferred_doc = "test"
limit = 300
pairs = extractor.extract_contrastive_pairs(boolq_task, limit=limit, preferred_doc=preferred_doc)
print(len(pairs))


