from collections.abc import Callable
from typing import Literal
import torch
import torch.nn.functional as F
from torch.optim import Optimizer
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

def compute_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    response_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

    if importance_reweighting_method=="none":
        per_token_policy_gradient_loss = raw_rewards_or_advantages*policy_log_probs
        metadata = {
            'clip-fraction': 1.0
        }

        return -per_token_policy_gradient_loss, metadata # return -ve loss for gradient ascent in pytorch

    raise NotImplementedError

def aggregate_loss_across_microbatch(
    per_token_policy_gradient_loss: torch.Tensor,
    mask: torch.Tensor,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> torch.Tensor:

    if loss_normalization == "sequence":
        masked_loss = per_token_policy_gradient_loss*mask
        loss_per_sequence = masked_loss.sum(dim=-1) / mask.sum(dim=-1).clamp_min(1)
        batch_loss = torch.mean(loss_per_sequence)
        return batch_loss

    raise NotImplementedError

def grpo_train_step(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    optimizer: Optimizer,
    gradient_accumulation_steps: int,
    max_grad_norm: float | None,
    reward_fn: Callable[[str, str], dict[str, float]],
    repeated_prompts: list[str],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    # Reward normalization
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
    # Importance reweighting and clipping
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    # Loss normalization
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:

    tokenized_prompt_and_output = tokenize_prompt_and_output(repeated_prompts, rollout_responses, tokenizer)    
    raw_rewards, rewards_metadata = compute_rollout_rewards(reward_fn, rollout_responses, repeated_ground_truths)
    advantages, advantage_metadata = compute_group_normalized_rewards(raw_rewards, group_size, baseline, advantage_eps, advantage_normalizer)

    microbatch_size = len(repeated_prompts) // gradient_accumulation_steps

    optimizer.zero_grad()

    loss = 0.0
    for i in range(0, len(repeated_prompts), microbatch_size):
        microbatch_inputs = tokenized_prompt_and_output["input_ids"][i:i+microbatch_size]
        microbatch_labels = tokenized_prompt_and_output["labels"][i:i+microbatch_size]
        microbatch_mask = tokenized_prompt_and_output["response_mask"][i:i+microbatch_size]
        microbatch_advantages = advantages[i : i + microbatch_size].unsqueeze(-1)
        microbatch_old_log_probs = (old_log_probs[i : i + microbatch_size] if old_log_probs is not None else None)
        policy_log_probs = get_response_log_probs(model, microbatch_inputs, microbatch_labels)
        per_token_policy_gradient_loss, loss_metadata = compute_policy_gradient_loss(microbatch_advantages, policy_log_probs["log_probs"], importance_reweighting_method, microbatch_old_log_probs, cliprange, microbatch_mask)
        microbatch_loss = aggregate_loss_across_microbatch(per_token_policy_gradient_loss, microbatch_mask, loss_normalization, normalization_constant)

        if loss_normalization == 'sequence':
            microbatch_loss = microbatch_loss * (microbatch_inputs.shape[0]/len(repeated_prompts))

        microbatch_loss.backward()
        loss += microbatch_loss

    total_norm = 0.0
    if max_grad_norm is not None:
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
    
    optimizer.step()
    optimizer.zero_grad()

    metadata = {
        "rewards_metadata": rewards_metadata,
        "advantage_metadata": advantage_metadata,
        "loss_metadata": loss_metadata,
        "total_norm": total_norm
    }
    
    return loss, metadata

