"""
Activation extraction strategies for cluster_all_benchmarks.py.

Contains functions for extracting activations using different strategies:
chat_mean, chat_first, chat_last, chat_max_norm, chat_weighted,
role_play, mc_balanced.
"""
import torch
import numpy as np
from typing import List, Tuple, Optional
from wisent.core.constants import NORM_EPS

RANDOM_TOKENS = ["I", "Well", "The", "Sure", "Let", "That", "It", "This", "My", "To"]

def get_last_token_act(model, tokenizer, text: str, layer: int, device: str) -> torch.Tensor:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length).to(device)
    with torch.no_grad():
        outputs = model(inputs.input_ids, output_hidden_states=True)
    return outputs.hidden_states[layer][0, -1, :].cpu().float()


def get_mean_answer_tokens_act(model, tokenizer, text: str, answer: str, layer: int, device: str) -> torch.Tensor:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length).to(device)
    answer_tokens = tokenizer(answer, add_special_tokens=False)["input_ids"]
    num_answer_tokens = len(answer_tokens)
    with torch.no_grad():
        outputs = model(inputs.input_ids, output_hidden_states=True)
    hidden = outputs.hidden_states[layer][0]
    if num_answer_tokens > 0 and num_answer_tokens < hidden.shape[0]:
        answer_hidden = hidden[-num_answer_tokens-1:-1, :]
        return answer_hidden.mean(dim=0).cpu().float()
    return hidden[-1].cpu().float()


def get_first_answer_token_act(model, tokenizer, text: str, answer: str, layer: int, device: str) -> torch.Tensor:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length).to(device)
    answer_tokens = tokenizer(answer, add_special_tokens=False)["input_ids"]
    num_answer_tokens = len(answer_tokens)
    with torch.no_grad():
        outputs = model(inputs.input_ids, output_hidden_states=True)
    hidden = outputs.hidden_states[layer][0]
    if num_answer_tokens > 0 and num_answer_tokens < hidden.shape[0]:
        first_answer_idx = hidden.shape[0] - num_answer_tokens - 1
        return hidden[first_answer_idx, :].cpu().float()
    return hidden[-1].cpu().float()


def get_generation_point_act(model, tokenizer, text: str, answer: str, layer: int, device: str) -> torch.Tensor:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length).to(device)
    answer_tokens = tokenizer(answer, add_special_tokens=False)["input_ids"]
    num_answer_tokens = len(answer_tokens)
    with torch.no_grad():
        outputs = model(inputs.input_ids, output_hidden_states=True)
    hidden = outputs.hidden_states[layer][0]
    gen_point_idx = max(0, hidden.shape[0] - num_answer_tokens - 2)
    return hidden[gen_point_idx, :].cpu().float()


def get_max_norm_answer_act(model, tokenizer, text: str, answer: str, layer: int, device: str) -> torch.Tensor:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length).to(device)
    answer_tokens = tokenizer(answer, add_special_tokens=False)["input_ids"]
    num_answer_tokens = len(answer_tokens)
    with torch.no_grad():
        outputs = model(inputs.input_ids, output_hidden_states=True)
    hidden = outputs.hidden_states[layer][0]
    if num_answer_tokens > 0 and num_answer_tokens < hidden.shape[0]:
        answer_hidden = hidden[-num_answer_tokens-1:-1, :]
        norms = torch.norm(answer_hidden, dim=1)
        max_idx = torch.argmax(norms)
        return answer_hidden[max_idx, :].cpu().float()
    return hidden[-1].cpu().float()


def get_weighted_mean_answer_act(model, tokenizer, text: str, answer: str, layer: int, device: str) -> torch.Tensor:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length).to(device)
    answer_tokens = tokenizer(answer, add_special_tokens=False)["input_ids"]
    num_answer_tokens = len(answer_tokens)
    with torch.no_grad():
        outputs = model(inputs.input_ids, output_hidden_states=True)
    hidden = outputs.hidden_states[layer][0]
    if num_answer_tokens > 0 and num_answer_tokens < hidden.shape[0]:
        answer_hidden = hidden[-num_answer_tokens-1:-1, :]
        weights = torch.exp(-torch.arange(answer_hidden.shape[0], dtype=torch.float32) * 0.5)
        weights = weights / weights.sum()
        weighted_mean = (answer_hidden * weights.unsqueeze(1).to(answer_hidden.device)).sum(dim=0)
        return weighted_mean.cpu().float()
    return hidden[-1].cpu().float()


def get_activation(model, tokenizer, prompt: str, response: str, layer: int, device: str, strategy: str) -> torch.Tensor:
    """Get activation using specified strategy."""
    random_token = RANDOM_TOKENS[hash(prompt) % len(RANDOM_TOKENS)]
    
    if strategy.startswith("chat_"):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt[:500]}, {"role": "assistant", "content": response}],
            tokenize=False, add_generation_prompt=False
        )
        
        if strategy == "chat_mean":
            return get_mean_answer_tokens_act(model, tokenizer, text, response, layer, device)
        elif strategy == "chat_first":
            return get_first_answer_token_act(model, tokenizer, text, response, layer, device)
        elif strategy == "chat_last":
            return get_last_token_act(model, tokenizer, text, layer, device)
        elif strategy == "chat_max_norm":
            return get_max_norm_answer_act(model, tokenizer, text, response, layer, device)
        elif strategy == "chat_weighted":
            return get_weighted_mean_answer_act(model, tokenizer, text, response, layer, device)
    
    elif strategy == "role_play":
        instruction = f"Behave like a person that would answer {prompt[:300]} with {response[:200]}"
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}, {"role": "assistant", "content": random_token}],
            tokenize=False, add_generation_prompt=False
        )
        return get_last_token_act(model, tokenizer, text, layer, device)
    
    elif strategy == "mc_balanced":
        # This needs both positive and negative - handled separately
        raise ValueError("mc_balanced requires special handling")
    
    # Fallback
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt[:500]}, {"role": "assistant", "content": response}],
        tokenize=False, add_generation_prompt=False
    )
    return get_last_token_act(model, tokenizer, text, layer, device)


def get_mc_balanced_activations(model, tokenizer, prompt: str, pos_response: str, neg_response: str, layer: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get activations for mc_balanced strategy (needs both responses)."""
    pos_goes_in_b = hash(prompt) % 2 == 0
    if pos_goes_in_b:
        mc_text = f"Which is correct?\nA. {neg_response[:200]}\nB. {pos_response[:200]}\nAnswer:"
        pos_ans, neg_ans = "B", "A"
    else:
        mc_text = f"Which is correct?\nA. {pos_response[:200]}\nB. {neg_response[:200]}\nAnswer:"
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
    return pos_act, neg_act


def load_benchmark_pairs(benchmark_name: str, loader: LMEvalDataLoader, limit: int = 60) -> List:
    task_name_lower = benchmark_name.lower()
    is_hf = task_name_lower in {k.lower() for k in HF_EXTRACTORS.keys()}
    
    if is_hf:
        pairs = lm_build_contrastive_pairs(task_name=benchmark_name, lm_eval_task=None, limit=limit)
    else:
        task_obj = loader.load_lm_eval_task(benchmark_name)
        if isinstance(task_obj, dict):
            pairs = []
            for subname, subtask in list(task_obj.items())[:3]:
                try:
                    sub_pairs = lm_build_contrastive_pairs(task_name=subname, lm_eval_task=subtask, limit=limit//3)
                    pairs.extend(sub_pairs)
                except:
                    pass
        else:
            pairs = lm_build_contrastive_pairs(task_name=benchmark_name, lm_eval_task=task_obj, limit=limit)
    return pairs


def compute_directions_for_strategy(
    model, tokenizer, pairs: List, layer: int, device: str, strategy: str, max_pairs: int = 50
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Compute steering direction using specified strategy."""
    pos_acts, neg_acts = [], []
    
    for pair in pairs[:max_pairs]:
        try:
            prompt = pair.prompt
            pos_response = pair.positive_response.model_response
            neg_response = pair.negative_response.model_response
            
            if strategy == "mc_balanced":
                pos_act, neg_act = get_mc_balanced_activations(
                    model, tokenizer, prompt, pos_response, neg_response, layer, device
                )
            else:
                pos_act = get_activation(model, tokenizer, prompt, pos_response, layer, device, strategy)
                neg_act = get_activation(model, tokenizer, prompt, neg_response, layer, device, strategy)
            
            pos_acts.append(pos_act)
            neg_acts.append(neg_act)
        except Exception as e:
            continue
    
    if len(pos_acts) < 10:
        return None, None, None
    
    pos_tensor = torch.stack(pos_acts)
    neg_tensor = torch.stack(neg_acts)
    direction = pos_tensor.mean(dim=0) - neg_tensor.mean(dim=0)
    norm = torch.norm(direction)
    if norm > NORM_EPS:
        direction = direction / norm
    return direction, pos_tensor, neg_tensor

def load_benchmark_pairs(benchmark_name: str, loader: LMEvalDataLoader, limit: int = 60) -> List:
    task_name_lower = benchmark_name.lower()
    is_hf = task_name_lower in {k.lower() for k in HF_EXTRACTORS.keys()}
    
    if is_hf:
        pairs = lm_build_contrastive_pairs(task_name=benchmark_name, lm_eval_task=None, limit=limit)
    else:
        task_obj = loader.load_lm_eval_task(benchmark_name)
        if isinstance(task_obj, dict):
            pairs = []
            for subname, subtask in list(task_obj.items())[:3]:
                try:
                    sub_pairs = lm_build_contrastive_pairs(task_name=subname, lm_eval_task=subtask, limit=limit//3)
                    pairs.extend(sub_pairs)
                except:
                    pass
        else:
            pairs = lm_build_contrastive_pairs(task_name=benchmark_name, lm_eval_task=task_obj, limit=limit)
    return pairs

