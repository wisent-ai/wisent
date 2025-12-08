# Example Contrastive Pairs

Pre-built contrastive pair datasets for common steering use cases.

## humanization_human_vs_ai.json

**Purpose:** Train steering vectors for humanization (making AI text appear more human-like)

**Source:** [aadityaubhat/GPT-wiki-intro](https://huggingface.co/datasets/aadityaubhat/GPT-wiki-intro) dataset
- **Positive examples:** Real human-written Wikipedia introductions
- **Negative examples:** GPT-3 generated versions of the same topics

**Usage:**
```bash
# Use with optimize-weights
wisent optimize-weights \
    --steering-vectors ./path/to/vectors.json \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --evaluator custom \
    --custom-evaluator wisent.core.evaluators.custom.examples.gptzero \
    --custom-evaluator-kwargs '{"api_key": "YOUR_GPTZERO_KEY"}' \
    --output-dir ./humanized_model

# Or generate steering vectors from these pairs
wisent get-activations \
    --pairs-file humanization_human_vs_ai.json \
    --output humanization_with_activations.json \
    --model meta-llama/Llama-3.2-1B-Instruct

wisent create-steering-vector \
    --enriched-pairs-file humanization_with_activations.json \
    --output humanization_vectors.json
```

**Structure:**
```json
{
  "pairs": [
    {
      "prompt": "Write an introduction about: [Topic]",
      "positive_response": {
        "model_response": "[Human-written Wikipedia text]",
        "metadata": {"source": "wikipedia", "is_human": true}
      },
      "negative_response": {
        "model_response": "[GPT-3 generated text]",
        "metadata": {"source": "gpt-3", "is_human": false}
      }
    }
  ],
  "metadata": {
    "source": "aadityaubhat/GPT-wiki-intro",
    "positive_label": "human_written",
    "negative_label": "ai_generated"
  }
}
```
