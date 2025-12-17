#!/usr/bin/env python3
"""
Test prompt strategies and extraction methods across multiple models.

Strategies (prompt formatting):
- chat_mean: Direct Q+A chat format, mean of answer tokens
- chat_first: Direct Q+A chat format, first answer token
- chat_last: Direct Q+A chat format, last token
- chat_gen_point: Direct Q+A chat format, last token before answer (generation point)
- chat_max_norm: Direct Q+A chat format, token with max norm in answer
- chat_weighted: Direct Q+A chat format, position-weighted mean (earlier tokens weighted more)
- role_play: "Behave like person who answers Q with A" format, last token
- mc_balanced: Multiple choice format, last token
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from datasets import load_dataset
import random
import gc

MODELS = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-2-7b-chat-hf",
    "Qwen/Qwen3-8B",
    "openai/gpt-oss-20b",
]

TASKS = ["truthfulqa_gen", "happy", "left_wing", "livecodebench"]

def load_task_pairs(task_name, n_samples=30):
    """Load contrastive pairs for a given task."""
    random.seed(42)
    
    if task_name == "truthfulqa_gen":
        ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
        samples = random.sample(list(ds), min(n_samples, len(ds)))
        return [{"question": s["question"], "positive": s["best_answer"], 
                 "negative": random.choice(s["incorrect_answers"])} for s in samples if s["incorrect_answers"]]
    
    elif task_name == "happy":
        # Sentiment: happy vs sad responses
        prompts = [
            "How are you feeling today?",
            "Tell me about your day.",
            "What's on your mind?",
            "How would you describe your mood?",
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
        pairs = []
        for i in range(n_samples):
            idx = i % len(prompts)
            pairs.append({
                "question": prompts[idx],
                "positive": happy_responses[idx],
                "negative": sad_responses[idx],
            })
        return pairs
    
    elif task_name == "left_wing":
        # Political: left vs right wing responses
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
        pairs = []
        for i in range(n_samples):
            idx = i % len(prompts)
            pairs.append({
                "question": prompts[idx],
                "positive": left_responses[idx],
                "negative": right_responses[idx],
            })
        return pairs
    
    elif task_name == "livecodebench":
        # Code: correct vs incorrect solutions
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
        pairs = []
        for i in range(n_samples):
            idx = i % len(prompts)
            pairs.append({
                "question": prompts[idx],
                "positive": correct[idx],
                "negative": incorrect[idx],
            })
        return pairs
    
    else:
        raise ValueError(f"Unknown task: {task_name}")

random_tokens = ["I", "Well", "The", "Sure", "Let", "That", "It", "This", "My", "To"]

def generate_nonsense_pairs(n_samples=30):
    """Generate nonsense contrastive pairs as baseline."""
    random.seed(123)
    nonsense_words = ["flurp", "blargh", "zorp", "quux", "xyzzy", "plugh", "wibble", "fnord", 
                      "gronk", "splat", "blip", "narf", "zoink", "floop", "glorp", "snarf"]
    pairs = []
    for i in range(n_samples):
        ctx_len = random.randint(3, 10)
        pos_len = random.randint(3, 10)
        neg_len = random.randint(3, 10)
        ctx = " ".join(random.choices(nonsense_words, k=ctx_len))
        pos = " ".join(random.choices(nonsense_words, k=pos_len))
        neg = " ".join(random.choices(nonsense_words, k=neg_len))
        pairs.append({"question": ctx, "positive": pos, "negative": neg})
    return pairs

nonsense_pairs = generate_nonsense_pairs(30)

def compute_cosine(diffs):
    cosines = []
    for i in range(len(diffs)):
        for j in range(i+1, len(diffs)):
            cos = np.dot(diffs[i], diffs[j]) / (np.linalg.norm(diffs[i]) * np.linalg.norm(diffs[j]) + 1e-8)
            cosines.append(cos)
    return np.mean(cosines) if cosines else 0

def compute_mean_direction(diffs):
    mean_dir = np.mean(diffs, axis=0)
    norm = np.linalg.norm(mean_dir)
    return mean_dir / norm if norm > 1e-8 else mean_dir

for model_name in MODELS:
    print(f"\n{'#'*70}")
    print(f"# Model: {model_name}")
    print(f"{'#'*70}")
    
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="mps")
    num_layers = model.config.num_hidden_layers
    print(f"Model has {num_layers} layers")
    
    # Pick layers based on model depth
    if num_layers <= 16:
        test_layers = [4, 6, 8, 10, 12, 14]
    elif num_layers <= 32:
        test_layers = [8, 12, 16, 20, 24, 28]
    else:
        test_layers = [10, 20, 30, 40, 50, 60]
    test_layers = [l for l in test_layers if l < num_layers]
    
    def get_last_token_act(text, layer):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to("mps")
        with torch.no_grad():
            outputs = model(inputs.input_ids, output_hidden_states=True)
        return outputs.hidden_states[layer][0, -1, :].cpu().float()
    
    def get_mean_answer_tokens_act(text, answer, layer):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to("mps")
        answer_tokens = tokenizer(answer, add_special_tokens=False)["input_ids"]
        num_answer_tokens = len(answer_tokens)
        with torch.no_grad():
            outputs = model(inputs.input_ids, output_hidden_states=True)
        hidden = outputs.hidden_states[layer][0]
        answer_hidden = hidden[-num_answer_tokens-1:-1, :]
        return answer_hidden.mean(dim=0).cpu().float()
    
    def get_first_answer_token_act(text, answer, layer):
        """Extract activation at the first token of the answer."""
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to("mps")
        answer_tokens = tokenizer(answer, add_special_tokens=False)["input_ids"]
        num_answer_tokens = len(answer_tokens)
        with torch.no_grad():
            outputs = model(inputs.input_ids, output_hidden_states=True)
        hidden = outputs.hidden_states[layer][0]
        first_answer_idx = hidden.shape[0] - num_answer_tokens - 1
        return hidden[first_answer_idx, :].cpu().float()
    
    def get_generation_point_act(text, answer, layer):
        """Extract activation at the last token before the answer starts (decision point)."""
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to("mps")
        answer_tokens = tokenizer(answer, add_special_tokens=False)["input_ids"]
        num_answer_tokens = len(answer_tokens)
        with torch.no_grad():
            outputs = model(inputs.input_ids, output_hidden_states=True)
        hidden = outputs.hidden_states[layer][0]
        gen_point_idx = max(0, hidden.shape[0] - num_answer_tokens - 2)
        return hidden[gen_point_idx, :].cpu().float()
    
    def get_max_norm_answer_act(text, answer, layer):
        """Extract activation at the token with maximum norm in the answer region."""
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to("mps")
        answer_tokens = tokenizer(answer, add_special_tokens=False)["input_ids"]
        num_answer_tokens = len(answer_tokens)
        with torch.no_grad():
            outputs = model(inputs.input_ids, output_hidden_states=True)
        hidden = outputs.hidden_states[layer][0]
        answer_hidden = hidden[-num_answer_tokens-1:-1, :]
        norms = torch.norm(answer_hidden, dim=1)
        max_idx = torch.argmax(norms)
        return answer_hidden[max_idx, :].cpu().float()
    
    def get_weighted_mean_answer_act(text, answer, layer):
        """Extract position-weighted mean of answer tokens (earlier tokens weighted more)."""
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to("mps")
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

    # Test all strategies
    strategies = [
        "chat_mean",
        "chat_first", 
        "chat_last",
        "chat_gen_point",
        "chat_max_norm",
        "chat_weighted",
        "role_play", 
        "mc_balanced"
    ]
    
    for task_name in TASKS:
        print(f"\n  === Task: {task_name} ===")
        pairs = load_task_pairs(task_name, n_samples=30)
        print(f"  Loaded {len(pairs)} pairs")
        
        for strategy in strategies:
            print(f"\n    Strategy: {strategy}")
        
            for layer in test_layers:
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
                            pos_act = get_mean_answer_tokens_act(pos_text, pos, layer)
                            neg_act = get_mean_answer_tokens_act(neg_text, neg, layer)
                        elif strategy == "chat_first":
                            pos_act = get_first_answer_token_act(pos_text, pos, layer)
                            neg_act = get_first_answer_token_act(neg_text, neg, layer)
                        elif strategy == "chat_last":
                            pos_act = get_last_token_act(pos_text, layer)
                            neg_act = get_last_token_act(neg_text, layer)
                        elif strategy == "chat_gen_point":
                            pos_act = get_generation_point_act(pos_text, pos, layer)
                            neg_act = get_generation_point_act(neg_text, neg, layer)
                        elif strategy == "chat_max_norm":
                            pos_act = get_max_norm_answer_act(pos_text, pos, layer)
                            neg_act = get_max_norm_answer_act(neg_text, neg, layer)
                        elif strategy == "chat_weighted":
                            pos_act = get_weighted_mean_answer_act(pos_text, pos, layer)
                            neg_act = get_weighted_mean_answer_act(neg_text, neg, layer)
                        
                    elif strategy == "role_play":
                        pos_instruction = f"Behave like a person that would answer {q} with {pos}"
                        neg_instruction = f"Behave like a person that would answer {q} with {neg}"
                        pos_text = tokenizer.apply_chat_template(
                            [{"role": "user", "content": pos_instruction}, {"role": "assistant", "content": random_token}],
                            tokenize=False, add_generation_prompt=False
                        )
                        neg_text = tokenizer.apply_chat_template(
                            [{"role": "user", "content": neg_instruction}, {"role": "assistant", "content": random_token}],
                            tokenize=False, add_generation_prompt=False
                        )
                        pos_act = get_last_token_act(pos_text, layer)
                        neg_act = get_last_token_act(neg_text, layer)
                        
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
                        pos_act = get_last_token_act(pos_text, layer)
                        neg_act = get_last_token_act(neg_text, layer)
                    
                    diffs.append((pos_act - neg_act).numpy())
                
                # Compute nonsense diffs for this strategy/layer
                nonsense_diffs = []
                for npair in nonsense_pairs:
                    nq = npair["question"]
                    npos, nneg = npair["positive"], npair["negative"]
                    nrandom_token = random_tokens[hash(nq) % len(random_tokens)]
                    
                    if strategy.startswith("chat_"):
                        npos_text = tokenizer.apply_chat_template(
                            [{"role": "user", "content": nq}, {"role": "assistant", "content": npos}],
                            tokenize=False, add_generation_prompt=False
                        )
                        nneg_text = tokenizer.apply_chat_template(
                            [{"role": "user", "content": nq}, {"role": "assistant", "content": nneg}],
                            tokenize=False, add_generation_prompt=False
                        )
                        if strategy == "chat_mean":
                            npos_act = get_mean_answer_tokens_act(npos_text, npos, layer)
                            nneg_act = get_mean_answer_tokens_act(nneg_text, nneg, layer)
                        elif strategy == "chat_first":
                            npos_act = get_first_answer_token_act(npos_text, npos, layer)
                            nneg_act = get_first_answer_token_act(nneg_text, nneg, layer)
                        elif strategy == "chat_last":
                            npos_act = get_last_token_act(npos_text, layer)
                            nneg_act = get_last_token_act(nneg_text, layer)
                        elif strategy == "chat_gen_point":
                            npos_act = get_generation_point_act(npos_text, npos, layer)
                            nneg_act = get_generation_point_act(nneg_text, nneg, layer)
                        elif strategy == "chat_max_norm":
                            npos_act = get_max_norm_answer_act(npos_text, npos, layer)
                            nneg_act = get_max_norm_answer_act(nneg_text, nneg, layer)
                        elif strategy == "chat_weighted":
                            npos_act = get_weighted_mean_answer_act(npos_text, npos, layer)
                            nneg_act = get_weighted_mean_answer_act(nneg_text, nneg, layer)
                    elif strategy == "role_play":
                        npos_instr = f"Behave like a person that would answer {nq} with {npos}"
                        nneg_instr = f"Behave like a person that would answer {nq} with {nneg}"
                        npos_text = tokenizer.apply_chat_template(
                            [{"role": "user", "content": npos_instr}, {"role": "assistant", "content": nrandom_token}],
                            tokenize=False, add_generation_prompt=False
                        )
                        nneg_text = tokenizer.apply_chat_template(
                            [{"role": "user", "content": nneg_instr}, {"role": "assistant", "content": nrandom_token}],
                            tokenize=False, add_generation_prompt=False
                        )
                        npos_act = get_last_token_act(npos_text, layer)
                        nneg_act = get_last_token_act(nneg_text, layer)
                    elif strategy == "mc_balanced":
                        npos_goes_in_b = hash(nq) % 2 == 0
                        if npos_goes_in_b:
                            nmc_text = f"Which is correct?\nA. {nneg[:200]}\nB. {npos[:200]}\nAnswer:"
                            npos_ans, nneg_ans = "B", "A"
                        else:
                            nmc_text = f"Which is correct?\nA. {npos[:200]}\nB. {nneg[:200]}\nAnswer:"
                            npos_ans, nneg_ans = "A", "B"
                        npos_text = tokenizer.apply_chat_template(
                            [{"role": "user", "content": nmc_text}, {"role": "assistant", "content": npos_ans}],
                            tokenize=False, add_generation_prompt=False
                        )
                        nneg_text = tokenizer.apply_chat_template(
                            [{"role": "user", "content": nmc_text}, {"role": "assistant", "content": nneg_ans}],
                            tokenize=False, add_generation_prompt=False
                        )
                        npos_act = get_last_token_act(npos_text, layer)
                        nneg_act = get_last_token_act(nneg_text, layer)
                    nonsense_diffs.append((npos_act - nneg_act).numpy())
                
                diffs = np.stack(diffs)
                nonsense_diffs = np.stack(nonsense_diffs)
                
                # Compute metrics
                M_cos = compute_cosine(diffs)
                N_cos = compute_cosine(nonsense_diffs)
                M_dir = compute_mean_direction(diffs)
                N_dir = compute_mean_direction(nonsense_diffs)
                M_vs_N = np.dot(M_dir, N_dir)
                
                print(f"      Layer {layer:2d}: M_cos={M_cos:.3f}, N_cos={N_cos:.3f}, M_vs_N={M_vs_N:.3f}")
    
    # Cleanup
    del model
    del tokenizer
    gc.collect()
    torch.mps.empty_cache()

print("\nDone")
