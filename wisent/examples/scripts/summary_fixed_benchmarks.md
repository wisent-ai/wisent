# Fixed Benchmarks Summary

## Successfully Fixed

1. **ArabCulture** ✓
   - Issue: Task name mismatch (ArabCulture → arab_culture)
   - Issue: Schema not recognized (first_statement + options + answer_key)
   - Fix: Added name mappings and Format 1 handler for ArabCulture schema
   - Status: **Working** - Creates 2 contrastive pairs, all evaluations correct

2. **acp_bench** ✓
   - Issue: Subtasks not registered (14 bool/mcq variants)
   - Issue: yes/no format not recognized
   - Fix: Added Format 3 handler for context+question+answer (yes/no) schema
   - Fix: Registered all 14 subtasks (7 bool, 7 mcq)
   - Status: **Working** - Creates 2 contrastive pairs, all evaluations correct

## Requires Different Approach

3. **Tag**
   - Issue: Does not exist in lm-eval harness under any searched name
   - Searched: tag, TAG, teca, taq, tga, and all variations
   - Status: **Cannot find** - May be misnamed in original test list

4. **acp_bench_hard**
   - Issue: Uses generation format, not multiple choice
   - Schema: context + question + answer (where answer is dict with neg/pos effect lists)
   - Current evaluator: log_likelihoods (unsuitable for generation)
   - Required evaluator: exact_match or structured output comparison
   - Status: **Requires generation evaluator** - Beyond current extractor scope

## Files Modified

- `/wisent/core/contrastive_pairs/lm_eval_pairs/lm_task_extractors/arabculture.py`
  - Added Format 1 handler for first_statement + options schema
  - Added all name aliases

- `/wisent/core/contrastive_pairs/lm_eval_pairs/lm_task_extractors/acp_bench.py`
  - Added Format 3 handler for yes/no questions
  - Added 14 subtask names (bool/mcq variants)
  - Added 8 _gen subtask names (for acp_bench_hard, but schema incompatible)

- `/wisent/core/contrastive_pairs/lm_eval_pairs/lm_extractor_manifest.py`
  - Added ArabCulture aliases (arabculture, ArabCulture, arab_culture)
  - Added acp_bench subtasks (bool, mcq, gen variants)
  - Added acp_bench_hard and acp_bench_hard_with_pddl

- `/wisent/core/data_loaders/loaders/lm_loader.py`
  - Added task name mappings for ArabCulture → arab_culture

## Test Results

### ArabCulture
```json
{
  "task_name": "ArabCulture",
  "num_pairs": 2,
  "all_correct": true
}
```

### acp_bench
```json
{
  "task_name": "acp_bench",
  "num_pairs": 2,
  "all_correct": true
}
```
