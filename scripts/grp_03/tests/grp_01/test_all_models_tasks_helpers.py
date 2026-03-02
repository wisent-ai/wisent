"""
Helper functions and data for test_all_models_tasks.py.

Contains task pair loading, activation extraction functions,
and metric computation utilities.
"""
import torch
import numpy as np
import random
from datasets import load_dataset
from wisent.core.utils.config_tools.constants import NORM_EPS, DEFAULT_RANDOM_SEED

random_tokens = ["I", "Well", "The", "Sure", "Let", "That", "It", "This", "My", "To"]


def load_task_pairs(task_name, n_samples=30):
    """Load contrastive pairs for a given task."""
    random.seed(DEFAULT_RANDOM_SEED)

    if task_name == "truthfulqa_gen":
        ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
        samples = random.sample(list(ds), min(n_samples, len(ds)))
        return [{"question": s["question"], "positive": s["best_answer"],
                 "negative": random.choice(s["incorrect_answers"])} for s in samples if s["incorrect_answers"]]

    elif task_name == "happy":
        prompts = [
            "How are you feeling today?", "Tell me about your day.",
            "What's on your mind?", "How would you describe your mood?",
            "What are your thoughts right now?",
        ]
        happy_responses = [
            "I'm feeling wonderful and full of joy!",
            "Everything is going great, I'm so happy!",
            "I feel fantastic and optimistic about everything!",
            "Life is beautiful and I'm grateful for it!",
            "I'm in such a positive mood today!",
        ]
        sad_responses = [
            "I'm feeling terrible and miserable.",
            "Everything is going wrong, I'm so sad.",
            "I feel awful and pessimistic about everything.",
            "Life is hard and I'm struggling.",
            "I'm in such a negative mood today.",
        ]
        return [{"question": prompts[i % len(prompts)],
                 "positive": happy_responses[i % len(prompts)],
                 "negative": sad_responses[i % len(prompts)]} for i in range(n_samples)]

    elif task_name == "left_wing":
        prompts = [
            "What do you think about economic policy?",
            "How should we address inequality?",
            "What's your view on healthcare?",
            "What about immigration policy?",
            "How should we handle climate change?",
        ]
        left_responses = [
            "We need more government intervention and wealth redistribution.",
            "Progressive taxation and social programs are essential.",
            "Universal healthcare is a fundamental right.",
            "We should welcome immigrants and provide paths to citizenship.",
            "Aggressive government action is needed to combat climate change.",
        ]
        right_responses = [
            "Free markets and limited government are the answer.",
            "Lower taxes and individual responsibility drive prosperity.",
            "Private healthcare and competition improve outcomes.",
            "We need strict border control and merit-based immigration.",
            "Market solutions and innovation will address environmental issues.",
        ]
        return [{"question": prompts[i % len(prompts)],
                 "positive": left_responses[i % len(prompts)],
                 "negative": right_responses[i % len(prompts)]} for i in range(n_samples)]

    elif task_name == "livecodebench":
        prompts = [
            "Write a function to add two numbers.",
            "Write a function to check if a number is even.",
            "Write a function to reverse a string.",
            "Write a function to find the maximum in a list.",
            "Write a function to check if a string is a palindrome.",
        ]
        correct = [
            "def add(a, b): return a + b",
            "def is_even(n): return n % 2 == 0",
            "def reverse(s): return s[::-1]",
            "def find_max(lst): return max(lst)",
            "def is_palindrome(s): return s == s[::-1]",
        ]
        incorrect = [
            "def add(a, b): return a - b",
            "def is_even(n): return n % 2 == 1",
            "def reverse(s): return s",
            "def find_max(lst): return min(lst)",
            "def is_palindrome(s): return s != s[::-1]",
        ]
        return [{"question": prompts[i % len(prompts)],
                 "positive": correct[i % len(prompts)],
                 "negative": incorrect[i % len(prompts)]} for i in range(n_samples)]

    else:
        raise ValueError(f"Unknown task: {task_name}")


def generate_nonsense_pairs(n_samples=30):
    """Generate nonsense contrastive pairs as baseline."""
    random.seed(123)
    nonsense_words = ["flurp", "blargh", "zorp", "quux", "xyzzy", "plugh", "wibble", "fnord",
                      "gronk", "splat", "blip", "narf", "zoink", "floop", "glorp", "snarf"]
    pairs = []
    for i in range(n_samples):
        ctx = " ".join(random.choices(nonsense_words, k=random.randint(3, 10)))
        pos = " ".join(random.choices(nonsense_words, k=random.randint(3, 10)))
        neg = " ".join(random.choices(nonsense_words, k=random.randint(3, 10)))
        pairs.append({"question": ctx, "positive": pos, "negative": neg})
    return pairs


def compute_cosine(diffs):
    cosines = []
    for i in range(len(diffs)):
        for j in range(i+1, len(diffs)):
            cos = np.dot(diffs[i], diffs[j]) / (np.linalg.norm(diffs[i]) * np.linalg.norm(diffs[j]) + NORM_EPS)
            cosines.append(cos)
    return np.mean(cosines) if cosines else 0


def compute_mean_direction(diffs):
    mean_dir = np.mean(diffs, axis=0)
    norm = np.linalg.norm(mean_dir)
    return mean_dir / norm if norm > NORM_EPS else mean_dir


def get_last_token_act(model, tokenizer, text, layer, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length).to(device)
    with torch.no_grad():
        outputs = model(inputs.input_ids, output_hidden_states=True)
    return outputs.hidden_states[layer][0, -1, :].cpu().float()


def get_mean_answer_tokens_act(model, tokenizer, text, answer, layer, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length).to(device)
    answer_tokens = tokenizer(answer, add_special_tokens=False)["input_ids"]
    num_answer_tokens = len(answer_tokens)
    with torch.no_grad():
        outputs = model(inputs.input_ids, output_hidden_states=True)
    hidden = outputs.hidden_states[layer][0]
    answer_hidden = hidden[-num_answer_tokens-1:-1, :]
    return answer_hidden.mean(dim=0).cpu().float()


def get_first_answer_token_act(model, tokenizer, text, answer, layer, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length).to(device)
    answer_tokens = tokenizer(answer, add_special_tokens=False)["input_ids"]
    num_answer_tokens = len(answer_tokens)
    with torch.no_grad():
        outputs = model(inputs.input_ids, output_hidden_states=True)
    hidden = outputs.hidden_states[layer][0]
    first_answer_idx = hidden.shape[0] - num_answer_tokens - 1
    return hidden[first_answer_idx, :].cpu().float()


def get_max_norm_answer_act(model, tokenizer, text, answer, layer, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length).to(device)
    answer_tokens = tokenizer(answer, add_special_tokens=False)["input_ids"]
    num_answer_tokens = len(answer_tokens)
    with torch.no_grad():
        outputs = model(inputs.input_ids, output_hidden_states=True)
    hidden = outputs.hidden_states[layer][0]
    answer_hidden = hidden[-num_answer_tokens-1:-1, :]
    norms = torch.norm(answer_hidden, dim=1)
    max_idx = torch.argmax(norms)
    return answer_hidden[max_idx, :].cpu().float()


def get_weighted_mean_answer_act(model, tokenizer, text, answer, layer, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length).to(device)
    answer_tokens = tokenizer(answer, add_special_tokens=False)["input_ids"]
    num_answer_tokens = len(answer_tokens)
    with torch.no_grad():
        outputs = model(inputs.input_ids, output_hidden_states=True)
    hidden = outputs.hidden_states[layer][0]
    answer_hidden = hidden[-num_answer_tokens-1:-1, :]
    weights = torch.exp(-torch.arange(answer_hidden.shape[0], dtype=torch.float32) * 0.5)
    weights = weights / weights.sum()
    weighted_mean = (answer_hidden * weights.unsqueeze(1).to(answer_hidden.device)).sum(dim=0)
    return weighted_mean.cpu().float()


def extract_diffs_for_strategy(model, tokenizer, pairs, strategy, layer, device):
    """Extract diffs for a given strategy, layer, and set of pairs."""
    diffs = []
    for pair in pairs:
        q = pair["question"][:500]
        pos, neg = pair["positive"], pair["negative"]
        random_token = random_tokens[hash(q) % len(random_tokens)]

        if strategy.startswith("chat_"):
            pos_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": q}, {"role": "assistant", "content": pos}],
                tokenize=False, add_generation_prompt=False
            )
            neg_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": q}, {"role": "assistant", "content": neg}],
                tokenize=False, add_generation_prompt=False
            )
            if strategy == "chat_mean":
                pos_act = get_mean_answer_tokens_act(model, tokenizer, pos_text, pos, layer, device)
                neg_act = get_mean_answer_tokens_act(model, tokenizer, neg_text, neg, layer, device)
            elif strategy == "chat_first":
                pos_act = get_first_answer_token_act(model, tokenizer, pos_text, pos, layer, device)
                neg_act = get_first_answer_token_act(model, tokenizer, neg_text, neg, layer, device)
            elif strategy == "chat_last":
                pos_act = get_last_token_act(model, tokenizer, pos_text, layer, device)
                neg_act = get_last_token_act(model, tokenizer, neg_text, layer, device)
            elif strategy == "chat_max_norm":
                pos_act = get_max_norm_answer_act(model, tokenizer, pos_text, pos, layer, device)
                neg_act = get_max_norm_answer_act(model, tokenizer, neg_text, neg, layer, device)
            elif strategy == "chat_weighted":
                pos_act = get_weighted_mean_answer_act(model, tokenizer, pos_text, pos, layer, device)
                neg_act = get_weighted_mean_answer_act(model, tokenizer, neg_text, neg, layer, device)
        elif strategy == "role_play":
            pos_instr = f"Behave like a person that would answer {q} with {pos}"
            neg_instr = f"Behave like a person that would answer {q} with {neg}"
            pos_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": pos_instr}, {"role": "assistant", "content": random_token}],
                tokenize=False, add_generation_prompt=False
            )
            neg_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": neg_instr}, {"role": "assistant", "content": random_token}],
                tokenize=False, add_generation_prompt=False
            )
            pos_act = get_last_token_act(model, tokenizer, pos_text, layer, device)
            neg_act = get_last_token_act(model, tokenizer, neg_text, layer, device)
        elif strategy == "mc_balanced":
            pos_goes_in_b = hash(q) % 2 == 0
            if pos_goes_in_b:
                mc_text = f"Which is correct?\nA. {neg[:200]}\nB. {pos[:200]}\nAnswer:"
                pos_ans, neg_ans = "B", "A"
            else:
                mc_text = f"Which is correct?\nA. {pos[:200]}\nB. {neg[:200]}\nAnswer:"
                pos_ans, neg_ans = "A", "B"
            pos_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": mc_text}, {"role": "assistant", "content": pos_ans}],
                tokenize=False, add_generation_prompt=False
            )
            neg_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": mc_text}, {"role": "assistant", "content": neg_ans}],
                tokenize=False, add_generation_prompt=False
            )
            pos_act = get_last_token_act(model, tokenizer, pos_text, layer, device)
            neg_act = get_last_token_act(model, tokenizer, neg_text, layer, device)

        diffs.append((pos_act - neg_act).numpy())
    return diffs
