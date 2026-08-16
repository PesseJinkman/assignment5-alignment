from collections.abc import Callable
from typing import Literal
import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizer, PreTrainedModel
from cs336_alignment.checkpoint import get_model_and_tokenizer


MODEL_ID = "allenai/OLMo-2-0425-1B"

def tokenize_prompt_and_output(
    prompt_strs: list[str], 
    output_strs: list[str], 
    tokenizer: PreTrainedTokenizer
) -> dict[str, torch.Tensor]:

    if len(prompt_strs) != len(output_strs):
        raise ValueError("Prompt and output batch sizes differ.")

    prompt_ids = tokenizer(prompt_strs, add_special_tokens=False)["input_ids"]
    output_ids = tokenizer(output_strs, add_special_tokens=False)["input_ids"]

    combined_lengths = [
        len(prompt) + len(output)
        for prompt, output in zip(prompt_ids, output_ids)
    ]
    max_length = max(combined_lengths)

    input_ids = []
    labels = []
    response_masks = []

    for prompt, output in zip(prompt_ids, output_ids):
        combined = prompt + output
        padding_length = max_length - len(combined)

        padded = combined + [tokenizer.pad_token_id] * padding_length

        mask = [0] * len(prompt) + [1] * len(output) + [0] * padding_length

        input_ids.append(padded[:-1])
        labels.append(padded[1:])
        response_masks.append(mask[1:])

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "response_mask": torch.tensor(response_masks, dtype=torch.bool),
    }

def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:

    logits = model(input_ids=input_ids).logits
    all_log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = torch.gather(all_log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    if not return_token_entropy:
        return {
            "log_probs": token_log_probs
        }

    token_entropy = -torch.sum(all_log_probs.exp() * all_log_probs, dim=-1)

    return {
        "log_probs": token_log_probs,
        "token_entropy": token_entropy
    }

def compute_rollout_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
) -> tuple[torch.Tensor, dict[str, float]]:


    format_rewards = 0.0
    answer_rewards = 0.0
    rewards = 0.0
    rewards_list = []
    for response, ground_truth in zip(rollout_responses, repeated_ground_truths):
        all_rewards = reward_fn(response, ground_truth)
        format_rewards += all_rewards['format_reward']
        answer_rewards += all_rewards['answer_reward']
        rewards += all_rewards['reward']
        rewards_list.append(all_rewards['reward'])

    n = len(rollout_responses)

    metadata = {
        'mean_format_rewards': format_rewards/n,
        'mean_answer_rewards': answer_rewards/n,
        'mean_total_rewards': rewards/n
    }

    return torch.Tensor(rewards_list), metadata

def compute_group_normalized_rewards(
    raw_rewards: torch.Tensor,
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
):

    grouped_rewards = raw_rewards.reshape(-1, group_size)

    if baseline=="mean" and advantage_normalizer=="std":
        grouped_mean = torch.mean(grouped_rewards, dim=-1, keepdim=True)
        grouped_std = torch.std(grouped_rewards, dim=-1, keepdim=True) + advantage_eps
        advantages = (grouped_rewards-grouped_mean)/grouped_std
        advantages = advantages.reshape(-1)
        metadata = {
            'max_reward': torch.max(raw_rewards).item(),
            'min_reward': torch.min(raw_rewards).item(),
            'max_mean_reward': torch.max(grouped_mean).item(),
            'min_mean_reward': torch.min(grouped_mean).item(), 
        }  

        return advantages, metadata

    raise NotImplementedError