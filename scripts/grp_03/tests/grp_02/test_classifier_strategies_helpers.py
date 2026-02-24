"""
Helper functions for test_classifier_strategies.py.

Contains task pair loading, text building for training strategies,
and activation extraction for both training and inference.
"""
import torch
import numpy as np
import random
from datasets import load_dataset
from wisent.core.constants import TOKENIZER_MAX_LENGTH_CLUSTER, DEFAULT_RANDOM_SEED

RANDOM_TOKENS = ["I", "Well", "The", "Sure", "Let", "That", "It", "This", "My", "To"]


def load_task_pairs(task_name, n_samples=100):
    random.seed(DEFAULT_RANDOM_SEED)

    if task_name == "truthfulqa":
        ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
        samples = random.sample(list(ds), min(n_samples, len(ds)))
        pairs = []
        for s in samples:
            if s["incorrect_answers"]:
                pairs.append({
                    "question": s["question"],
                    "positive": s["best_answer"],
                    "negative": random.choice(s["incorrect_answers"])
                })
        return pairs

    elif task_name == "happy":
        prompts = ["How are you feeling today?", "Tell me about your day.", "What's on your mind?",
                   "How would you describe your mood?", "What are your thoughts right now?"]
        happy = ["I'm feeling wonderful and full of joy!", "Everything is going great, I'm so happy!",
                 "I feel fantastic and optimistic about everything!", "Life is beautiful and I'm grateful for it!",
                 "I'm in such a positive mood today!"]
        sad = ["I'm feeling terrible and miserable.", "Everything is going wrong, I'm so sad.",
               "I feel awful and pessimistic about everything.", "Life is hard and I'm struggling.",
               "I'm in such a negative mood today."]
        return [{"question": prompts[i % 5], "positive": happy[i % 5], "negative": sad[i % 5]} for i in range(n_samples)]

    elif task_name == "left_wing":
        prompts = ["What do you think about economic policy?", "How should we address inequality?",
                   "What's your view on healthcare?", "What about immigration policy?", "How should we handle climate change?"]
        left = ["We need more government intervention and wealth redistribution.",
                "Progressive taxation and social programs are essential.", "Universal healthcare is a fundamental right.",
                "We should welcome immigrants and provide paths to citizenship.",
                "Aggressive government action is needed to combat climate change."]
        right = ["Free markets and limited government are the answer.",
                 "Lower taxes and individual responsibility drive prosperity.",
                 "Private healthcare and competition improve outcomes.",
                 "We need strict border control and merit-based immigration.",
                 "Market solutions and innovation will address environmental issues."]
        return [{"question": prompts[i % 5], "positive": left[i % 5], "negative": right[i % 5]} for i in range(n_samples)]

    elif task_name == "livecodebench":
        prompts = ["Write a function to add two numbers.", "Write a function to check if a number is even.",
                   "Write a function to reverse a string.", "Write a function to find the maximum in a list.",
                   "Write a function to check if a string is a palindrome."]
        correct = ["def add(a, b): return a + b", "def is_even(n): return n % 2 == 0",
                   "def reverse(s): return s[::-1]", "def find_max(lst): return max(lst)",
                   "def is_palindrome(s): return s == s[::-1]"]
        incorrect = ["def add(a, b): return a - b", "def is_even(n): return n % 2 == 1",
                     "def reverse(s): return s", "def find_max(lst): return min(lst)",
                     "def is_palindrome(s): return s != s[::-1]"]
        return [{"question": prompts[i % 5], "positive": correct[i % 5], "negative": incorrect[i % 5]} for i in range(n_samples)]

    raise ValueError(f"Unknown task: {task_name}")


def build_train_text(tokenizer, question, answer, train_strategy, other_answer=None):
    """Build text for TRAINING based on train_strategy."""
    if train_strategy.startswith("chat_"):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}, {"role": "assistant", "content": answer}],
            tokenize=False, add_generation_prompt=False
        )
        return text, answer

    elif train_strategy == "role_play":
        random_token = RANDOM_TOKENS[hash(question) % len(RANDOM_TOKENS)]
        instruction = f"Behave like a person that would answer {question} with {answer}"
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}, {"role": "assistant", "content": random_token}],
            tokenize=False, add_generation_prompt=False
        )
        return text, random_token

    elif train_strategy == "mc_balanced":
        pos_goes_in_b = hash(question) % 2 == 0
        other = other_answer or "Alternative"
        if pos_goes_in_b:
            mc_text = f"Which is correct?\nA. {other[:200]}\nB. {answer[:200]}\nAnswer:"
            choice = "B"
        else:
            mc_text = f"Which is correct?\nA. {answer[:200]}\nB. {other[:200]}\nAnswer:"
            choice = "A"
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": mc_text}, {"role": "assistant", "content": choice}],
            tokenize=False, add_generation_prompt=False
        )
        return text, choice

    raise ValueError(f"Unknown train_strategy: {train_strategy}")


def get_train_activation(model, tokenizer, text, answer, layer, train_strategy, device):
    """Extract activation for TRAINING based on train_strategy."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=TOKENIZER_MAX_LENGTH_CLUSTER).to(device)
    with torch.no_grad():
        outputs = model(inputs.input_ids, output_hidden_states=True)
    hidden = outputs.hidden_states[layer][0]
    seq_len = hidden.shape[0]

    answer_tokens = tokenizer(answer, add_special_tokens=False)["input_ids"]
    num_ans = len(answer_tokens)

    if train_strategy == "chat_last":
        return hidden[-1].cpu().float().numpy()
    elif train_strategy == "chat_first":
        idx = max(0, seq_len - num_ans - 1)
        return hidden[idx].cpu().float().numpy()
    elif train_strategy == "chat_mean":
        if num_ans > 0 and seq_len > num_ans:
            return hidden[-num_ans-1:-1].mean(dim=0).cpu().float().numpy()
        return hidden[-1].cpu().float().numpy()
    elif train_strategy == "chat_max_norm":
        if num_ans > 0 and seq_len > num_ans:
            ans_hidden = hidden[-num_ans-1:-1]
            max_idx = torch.argmax(torch.norm(ans_hidden, dim=1))
            return ans_hidden[max_idx].cpu().float().numpy()
        return hidden[-1].cpu().float().numpy()
    elif train_strategy == "chat_weighted":
        if num_ans > 0 and seq_len > num_ans:
            ans_hidden = hidden[-num_ans-1:-1]
            w = torch.exp(-torch.arange(ans_hidden.shape[0], dtype=torch.float32, device=ans_hidden.device) * 0.5)
            w = w / w.sum()
            return (ans_hidden * w.unsqueeze(1)).sum(dim=0).cpu().float().numpy()
        return hidden[-1].cpu().float().numpy()
    elif train_strategy in ("role_play", "mc_balanced"):
        return hidden[-1].cpu().float().numpy()

    raise ValueError(f"Unknown train_strategy: {train_strategy}")


def get_inference_score(clf, model, tokenizer, text, layer, inference_strategy, device):
    """Get classifier score using inference_strategy."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=TOKENIZER_MAX_LENGTH_CLUSTER).to(device)
    with torch.no_grad():
        outputs = model(inputs.input_ids, output_hidden_states=True)
    hidden = outputs.hidden_states[layer][0].cpu().float().numpy()
    seq_len = hidden.shape[0]

    if inference_strategy == "last_token":
        return clf.predict_proba([hidden[-1]])[0, 1]
    elif inference_strategy == "first_token":
        return clf.predict_proba([hidden[0]])[0, 1]
    elif inference_strategy in ("all_mean", "all_max", "all_min"):
        all_scores = []
        for t in range(seq_len):
            score = clf.predict_proba([hidden[t]])[0, 1]
            all_scores.append(score)
        if inference_strategy == "all_mean":
            return np.mean(all_scores)
        elif inference_strategy == "all_max":
            return np.max(all_scores)
        elif inference_strategy == "all_min":
            return np.min(all_scores)

    raise ValueError(f"Unknown inference_strategy: {inference_strategy}")
