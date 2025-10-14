# Example of collecting activations from a model using WisentGuard on gsm8k dataset
# We are going to create contrastive pairs and collect activations from specific layers and plot them.

# First we need to load the model

from wisent_guard.core.models.wisent_model import WisentModel

llm = WisentModel(
    model_name="/home/gg/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6",
    device="cuda",
)

# A faw examples from gsm8k dataset. We are going to use load dataset from datasets library
from datasets import load_dataset

dataset = load_dataset("openai/gsm8k", "main", split="train[:10%]")  # Load a small subset for demonstration

# Prepare examples
examples = []
for item in dataset:
    question = item["question"].strip()
    answer = item["answer"].strip().split("####")[-1].strip()  

    # we need to handle
    # numbers like 1,080

    answer = answer.replace(",", "")
    try:
        answer = int(answer)
    except ValueError:
        try:
            answer = float(answer)
        except ValueError:
            continue  # Skip if we can't parse the answer as a number
    
    negative_answer = answer + 1  # Simple incorrect answer for demonstration
    examples.append({
        "prompt": question,
        "positive_response": f"{answer}",
        "negative_response": f"{negative_answer}",
    })

print(f"Loaded {len(examples)} examples from gsm8k dataset.")

print("Example:", examples[0])

_CUT = 100

examples = examples[:_CUT]

# Creating contrastive pairs
from wisent_guard.core.contrastive_pairs.core.pair import ContrastivePair
from wisent_guard.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
pairs = [
    ContrastivePair(
        prompt=ex["prompt"],
        positive_response=PositiveResponse(
            model_response=ex["positive_response"],
            layers_activations=None,
            label="correct",
        ),
        negative_response=NegativeResponse(
            model_response=ex["negative_response"],
            layers_activations=None,
            label="incorrect",
        ),
        label="math_problem",
        trait_description="math",
    )
    for ex in examples
]

# Now we can collect activations from specific layers
from wisent_guard.core.activations.activations_collector import ActivationCollector

collector = ActivationCollector(
    model=llm
)
layers_to_collect = ["1", "3", "5", "8", "10", "16"]  # Example layer names
aggregation_strategy = "continuation_token"  # Example aggregation strategy
pairs_with_activations = [collector.collect_for_pair(pair, layers=layers_to_collect, aggregation=aggregation_strategy) for pair in pairs]

def plot_activations(pairs: list[ContrastivePair], layer_name):
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    import numpy as np

    print(f"Plotting activations for layer {layer_name}")
    print(f"Number of pairs: {len(pairs)}")
    positive_acts = []
    negative_acts = []
    for pair in pairs:
        pos_act = pair.positive_response.layers_activations[layer_name]
        neg_act = pair.negative_response.layers_activations[layer_name]
        if pos_act is not None:
            positive_acts.append(pos_act.cpu().float().numpy())
        if neg_act is not None:
            negative_acts.append(neg_act.cpu().float().numpy())

    if not positive_acts or not negative_acts:
        print(f"No activations found for layer {layer_name}")
        return

    positive_acts = np.vstack(positive_acts)
    negative_acts = np.vstack(negative_acts)

    pca = PCA(n_components=2)
    all_acts = np.vstack([positive_acts, negative_acts])
    pca.fit(all_acts)
    pos_pca = pca.transform(positive_acts)
    neg_pca = pca.transform(negative_acts)

    plt.figure(figsize=(8, 6))
    plt.scatter(pos_pca[:, 0], pos_pca[:, 1], color='blue', label='Positive Responses')
    plt.scatter(neg_pca[:, 0], neg_pca[:, 1], color='red', label='Negative Responses')
    plt.title(f'Activations at Layer {layer_name}')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_activations(pairs_with_activations, layer_name="10")