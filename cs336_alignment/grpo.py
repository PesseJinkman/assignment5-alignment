from collections.abc import Callable
from typing import Literal
import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from transformers import PreTrainedTokenizer, PreTrainedModel, AutoModelForCausalLM, AutoTokenizer
from cs336_alignment.checkpoint import get_model_and_tokenizer
from cs336_alignment.vllm_utils import VLLMServer
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn, question_only_reward_fn
import json

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
) -> tuple[torch.Tensor, dict[str, float | torch.Tensor]]:

    if len(rollout_responses) != len(repeated_ground_truths):
        raise ValueError("Rollout responses and ground truths must have the same length.")
    if not rollout_responses:
        raise ValueError("Cannot compute rollout rewards for an empty batch.")

    rewards_list = []
    format_rewards_list = []
    answer_rewards_list = []
    for response, ground_truth in zip(rollout_responses, repeated_ground_truths):
        all_rewards = reward_fn(response, ground_truth)
        rewards_list.append(all_rewards['reward'])
        format_rewards_list.append(all_rewards['format_reward'])
        answer_rewards_list.append(all_rewards['answer_reward'])

    raw_rewards = torch.tensor(rewards_list, dtype=torch.float32)
    format_rewards = torch.tensor(format_rewards_list, dtype=torch.float32)
    answer_rewards = torch.tensor(answer_rewards_list, dtype=torch.float32)

    metadata = {
        "mean_total_rewards": raw_rewards.mean().item(),
        "std_total_rewards": raw_rewards.std(unbiased=False).item(),
        "min_total_rewards": raw_rewards.min().item(),
        "max_total_rewards": raw_rewards.max().item(),
        "mean_format_rewards": format_rewards.mean().item(),
        "mean_answer_rewards": answer_rewards.mean().item(),
        "nonzero_reward_fraction": (raw_rewards != 0).float().mean().item(),
        "num_rollouts": float(raw_rewards.numel()),
        "raw_rewards": raw_rewards.detach(),
        "format_rewards": format_rewards.detach(),
        "answer_rewards": answer_rewards.detach(),
    }

    return raw_rewards, metadata

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
            "advantage_mean": advantages.mean().item(),
            "advantage_std": advantages.std(unbiased=False).item(),
            "advantage_min": advantages.min().item(),
            "advantage_max": advantages.max().item(),
            "group_reward_mean_mean": grouped_mean.mean().item(),
            "group_reward_mean_min": grouped_mean.min().item(),
            "group_reward_mean_max": grouped_mean.max().item(),
            "group_reward_std_mean": grouped_std.mean().item(),
            "zero_variance_group_fraction": (
                grouped_std <= advantage_eps
            ).float().mean().item(),
            "advantages": advantages.detach(),
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
) -> tuple[torch.Tensor, dict[str, float | torch.Tensor]]:

    if importance_reweighting_method=="none":
        per_token_policy_gradient_loss = raw_rewards_or_advantages*policy_log_probs
        metadata = {
            "clip_fraction": 0.0,
            "importance_ratio_mean": 1.0,
            "policy_log_prob_mean": policy_log_probs.detach().mean().item(),
            "policy_log_prob_std": policy_log_probs.detach().std(unbiased=False).item(),
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
) -> tuple[torch.Tensor, dict[str, object]]:

    tokenized_prompt_and_output = tokenize_prompt_and_output(repeated_prompts, rollout_responses, tokenizer)    
    raw_rewards, rewards_metadata = compute_rollout_rewards(reward_fn, rollout_responses, repeated_ground_truths)
    advantages, advantage_metadata = compute_group_normalized_rewards(raw_rewards, group_size, baseline, advantage_eps, advantage_normalizer)

    microbatch_size = len(repeated_prompts) // gradient_accumulation_steps
    device = next(model.parameters()).device

    optimizer.zero_grad(set_to_none=True)

    loss = 0.0
    response_entropy_sum = 0.0
    response_log_prob_sum = 0.0
    response_token_count = 0
    response_lengths = []
    policy_metadata_sums: dict[str, float] = {}
    processed_examples = 0
    for i in range(0, len(repeated_prompts), microbatch_size):
        microbatch_inputs = tokenized_prompt_and_output["input_ids"][i:i+microbatch_size].to(device)
        microbatch_labels = tokenized_prompt_and_output["labels"][i:i+microbatch_size].to(device)
        microbatch_mask = tokenized_prompt_and_output["response_mask"][i:i+microbatch_size].to(device)
        microbatch_advantages = advantages[i : i + microbatch_size].unsqueeze(-1).to(device)
        microbatch_old_log_probs = (
            old_log_probs[i : i + microbatch_size].to(device)
            if old_log_probs is not None
            else None
        )

        policy_outputs = get_response_log_probs(
            model,
            microbatch_inputs,
            microbatch_labels,
            return_token_entropy=True,
        )
        per_token_policy_gradient_loss, microbatch_policy_metadata = compute_policy_gradient_loss(microbatch_advantages, policy_outputs["log_probs"], importance_reweighting_method, microbatch_old_log_probs, cliprange, microbatch_mask)
        microbatch_loss = aggregate_loss_across_microbatch(per_token_policy_gradient_loss, microbatch_mask, loss_normalization, normalization_constant)

        if loss_normalization == 'sequence':
            microbatch_loss = microbatch_loss * (microbatch_inputs.shape[0]/len(repeated_prompts))

        microbatch_loss.backward()
        loss += microbatch_loss

        detached_mask = microbatch_mask.detach()
        response_entropy_sum += (
            policy_outputs["token_entropy"].detach() * detached_mask
        ).sum().item()
        response_log_prob_sum += (
            policy_outputs["log_probs"].detach() * detached_mask
        ).sum().item()
        response_token_count += detached_mask.sum().item()
        response_lengths.extend(detached_mask.sum(dim=-1).cpu().tolist())
        microbatch_examples = microbatch_inputs.shape[0]
        processed_examples += microbatch_examples
        for key, value in microbatch_policy_metadata.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().float().mean().item()
            policy_metadata_sums[key] = (
                policy_metadata_sums.get(key, 0.0)
                + float(value) * microbatch_examples
            )

    if max_grad_norm is not None:
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
    else:
        parameter_grad_norms = [
            parameter.grad.detach().norm(2)
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        total_norm = (
            torch.stack(parameter_grad_norms).norm(2)
            if parameter_grad_norms
            else torch.tensor(0.0, device=device)
        )
    
    if not torch.isfinite(loss.detach()):
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError("GRPO loss became non-finite before the optimizer step.")
    if not torch.isfinite(total_norm.detach()):
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError("GRPO gradient norm became non-finite before the optimizer step.")

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    mean_response_log_prob = response_log_prob_sum / max(response_token_count, 1)
    loss_metadata = {
        key: value / max(processed_examples, 1)
        for key, value in policy_metadata_sums.items()
    }
    train_metadata = {
        "mean_token_entropy": response_entropy_sum / max(response_token_count, 1),
        "mean_response_log_prob": mean_response_log_prob,
        "mean_response_nll": -mean_response_log_prob,
        "response_perplexity": float(
            torch.exp(torch.tensor(min(-mean_response_log_prob, 80.0))).item()
        ),
        "num_response_tokens": float(response_token_count),
        "mean_response_length": float(sum(response_lengths) / max(len(response_lengths), 1)),
        "min_response_length": float(min(response_lengths, default=0)),
        "max_response_length": float(max(response_lengths, default=0)),
    }

    metadata = {
        "rewards_metadata": rewards_metadata,
        "advantage_metadata": advantage_metadata,
        "loss_metadata": loss_metadata,
        "train_metadata": train_metadata,
        "total_norm": total_norm,
    }
    
    return loss, metadata

TRAIN_DATA_PATH = "data/gsm8k/train.jsonl"
TEST_DATA_PATH = "data/gsm8k/test.jsonl"
PROMPT_PATH = "cs336_alignment/prompts/r1_zero.prompt"
MODEL_ID = "allenai/OLMo-2-0425-1B"


def main() -> None:
    import random
    import time
    from pathlib import Path

    import wandb

    config = {
        "model_id": MODEL_ID,
        "train_data_path": TRAIN_DATA_PATH,
        "val_data_path": TEST_DATA_PATH,
        "prompt_path": PROMPT_PATH,
        "reward_function": "auto",
        "n_train_examples": 6400,
        "n_val_examples": 1024,
        "num_rollout_steps": 200,
        "learning_rate": 1e-5,
        "rollout_batch_size": 256,
        "train_batch_size": 256,
        "group_size": 8,
        "gradient_accumulation_steps": 32,
        "sampling_temperature": 1.0,
        "sampling_max_tokens": 512,
        "generation_batch_size": 256,
        "max_grad_norm": 1.0,
        "eval_every": 10,
        "log_rollouts_every": 40,
        "logged_rollouts": 16,
        "save_every": 0,
        "output_dir": "experiments/grpo",
        "seed": 0,
        "wandb_project": "cs336-a5-grpo",
        "wandb_run_name": None,
        "wandb_mode": "online",
    }

    if config["rollout_batch_size"] != config["train_batch_size"]:
        raise ValueError("Standard on-policy GRPO requires equal rollout and train batch sizes.")
    if config["rollout_batch_size"] % config["group_size"] != 0:
        raise ValueError("rollout_batch_size must be divisible by group_size.")
    if config["train_batch_size"] % config["gradient_accumulation_steps"] != 0:
        raise ValueError("train_batch_size must be divisible by gradient_accumulation_steps.")
    if config["eval_every"] <= 0 or config["log_rollouts_every"] <= 0:
        raise ValueError("eval_every and log_rollouts_every must be positive.")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("This script requires two CUDA GPUs: policy on GPU 0 and vLLM on GPU 1.")

    rng = random.Random(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])

    with open(config["train_data_path"], "r", encoding="utf-8") as stream:
        train_data = [json.loads(line) for line in stream if line.strip()]
    with open(config["val_data_path"], "r", encoding="utf-8") as stream:
        val_data = [json.loads(line) for line in stream if line.strip()]
    with open(config["prompt_path"], "r", encoding="utf-8") as stream:
        prompt_template = stream.read()

    if config["n_train_examples"] > len(train_data):
        raise ValueError(
            f"Requested {config['n_train_examples']} training examples, but only {len(train_data)} are available."
        )
    if config["n_val_examples"] > len(val_data):
        raise ValueError(
            f"Requested {config['n_val_examples']} validation examples, but only {len(val_data)} are available."
        )

    rng.shuffle(train_data)
    train_data = train_data[: config["n_train_examples"]]
    val_data = val_data[: config["n_val_examples"]]

    reward_name = config["reward_function"]
    if reward_name == "auto":
        reward_name = "question_only" if "question_only" in Path(config["prompt_path"]).stem else "r1_zero"
    reward_fn = question_only_reward_fn if reward_name == "question_only" else r1_zero_reward_fn
    config["resolved_reward_function"] = reward_name

    def format_example(example: dict) -> tuple[str, str]:
        if "####" not in example["answer"]:
            raise ValueError("Expected GSM8K answers to contain a '####' ground-truth delimiter.")
        prompt = prompt_template.format(question=example["question"])
        ground_truth = example["answer"].rsplit("####", 1)[1].strip()
        return prompt, ground_truth

    train_examples = [format_example(example) for example in train_data]
    val_examples = [format_example(example) for example in val_data]
    prompts_per_rollout = config["rollout_batch_size"] // config["group_size"]
    required_train_prompts = config["num_rollout_steps"] * prompts_per_rollout
    if required_train_prompts > len(train_examples):
        raise ValueError(
            "The requested run needs "
            f"{required_train_prompts} training prompts, but n_train_examples={len(train_examples)}."
        )

    policy_device = "cuda:0"
    policy, tokenizer = get_model_and_tokenizer(config["model_id"], policy_device)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("The tokenizer must define either a pad token or an EOS token.")
        tokenizer.pad_token = tokenizer.eos_token
    policy.train()
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=config["learning_rate"],
        betas=(0.9, 0.95),
        weight_decay=0.0,
    )

    sampling_params = {
        "temperature": config["sampling_temperature"],
        "top_p": 1.0,
        "max_tokens": config["sampling_max_tokens"],
        "n": config["group_size"],
        "seed": config["seed"],
        "stop": ["</answer>"],
        "include_stop_str_in_output": True,
    }
    validation_sampling_params = dict(sampling_params, n=1)

    run_name = config["wandb_run_name"] or f"grpo-seed-{config['seed']}"
    output_dir = Path(config["output_dir"]) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = wandb.init(
        project=config["wandb_project"],
        name=run_name,
        mode=config["wandb_mode"],
        config=config,
        save_code=True,
    )
    wandb.define_metric("rollout_step")
    wandb.define_metric("train/*", step_metric="rollout_step")
    wandb.define_metric("val/*", step_metric="rollout_step")
    wandb.define_metric("time/*", step_metric="rollout_step")
    wandb.define_metric("throughput/*", step_metric="rollout_step")
    wandb.define_metric("gpu/*", step_metric="rollout_step")
    wandb.define_metric("train/loss", step_metric="rollout_step", summary="min")
    wandb.define_metric("train/reward", step_metric="rollout_step", summary="max")
    wandb.define_metric("val/reward", step_metric="rollout_step", summary="max")

    vllm_server = VLLMServer(model_id=config["model_id"], gpu=1, seed=config["seed"])

    def add_scalar_metrics(
        destination: dict,
        prefix: str,
        source: dict,
    ) -> None:
        for key, value in source.items():
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    continue
                value = value.detach().float().item()
            if isinstance(value, (int, float)):
                destination[f"{prefix}/{key}"] = float(value)

    def evaluate(step: int) -> None:
        evaluation_started = time.perf_counter()
        val_prompts = [prompt for prompt, _ in val_examples]
        val_ground_truths = [ground_truth for _, ground_truth in val_examples]
        completions = vllm_server.generate_completions(
            prompts=val_prompts,
            sampling_params=validation_sampling_params,
            batch_size=config["generation_batch_size"],
        )
        if len(completions) != len(val_examples):
            raise RuntimeError(
                f"Expected {len(val_examples)} validation completions, received {len(completions)}."
            )

        total_rewards = []
        format_rewards = []
        answer_rewards = []
        response_lengths = []
        finish_reason_counts: dict[str, int] = {}
        for completion, ground_truth in zip(completions, val_ground_truths):
            rewards = reward_fn(completion.text, ground_truth)
            total_rewards.append(rewards["reward"])
            format_rewards.append(rewards["format_reward"])
            answer_rewards.append(rewards["answer_reward"])
            response_lengths.append(len(completion.token_ids))
            finish_reason = completion.finish_reason or "unknown"
            finish_reason_counts[finish_reason] = finish_reason_counts.get(finish_reason, 0) + 1

        count = len(completions)
        total_reward_tensor = torch.tensor(total_rewards, dtype=torch.float32)
        response_length_tensor = torch.tensor(response_lengths, dtype=torch.float32)
        val_metrics = {
            "val/reward": total_reward_tensor.mean().item(),
            "val/reward_std": total_reward_tensor.std(unbiased=False).item(),
            "val/reward_min": total_reward_tensor.min().item(),
            "val/reward_max": total_reward_tensor.max().item(),
            "val/nonzero_reward_fraction": (total_reward_tensor != 0).float().mean().item(),
            "val/format_reward": sum(format_rewards) / count,
            "val/answer_reward": sum(answer_rewards) / count,
            "val/average_response_length": response_length_tensor.mean().item(),
            "val/response_length_std": response_length_tensor.std(unbiased=False).item(),
            "val/response_length_min": response_length_tensor.min().item(),
            "val/response_length_max": response_length_tensor.max().item(),
            "val/evaluation_seconds": time.perf_counter() - evaluation_started,
            "val/reward_histogram": wandb.Histogram(total_rewards),
            "val/response_length_histogram": wandb.Histogram(response_lengths),
            "rollout_step": step,
        }
        for finish_reason, finish_count in finish_reason_counts.items():
            metric_name = finish_reason.replace(" ", "_").replace("/", "_")
            val_metrics[f"val/finish_reason_{metric_name}_fraction"] = finish_count / count
        wandb_run.log(val_metrics)
        print(
            f"step={step} val_reward={val_metrics['val/reward']:.4f} "
            f"val_format_reward={val_metrics['val/format_reward']:.4f} "
            f"val_response_length={val_metrics['val/average_response_length']:.1f}",
            flush=True,
        )

    try:
        vllm_server.start()
        vllm_server.init_weight_sync(policy_device)
        vllm_server.sync_policy_weights(policy)
        evaluate(step=0)

        for step in range(1, config["num_rollout_steps"] + 1):
            step_started = time.perf_counter()
            torch.cuda.reset_peak_memory_stats(0)
            start = (step - 1) * prompts_per_rollout
            batch_examples = train_examples[start : start + prompts_per_rollout]
            batch_prompts = [prompt for prompt, _ in batch_examples]
            batch_ground_truths = [ground_truth for _, ground_truth in batch_examples]

            rollout_started = time.perf_counter()
            completions = vllm_server.generate_completions(
                prompts=batch_prompts,
                sampling_params=sampling_params,
                batch_size=config["generation_batch_size"],
            )
            rollout_seconds = time.perf_counter() - rollout_started
            if len(completions) != config["rollout_batch_size"]:
                raise RuntimeError(
                    f"Expected {config['rollout_batch_size']} rollouts, received {len(completions)}."
                )

            repeated_prompts = [
                prompt
                for prompt in batch_prompts
                for _ in range(config["group_size"])
            ]
            repeated_ground_truths = [
                ground_truth
                for ground_truth in batch_ground_truths
                for _ in range(config["group_size"])
            ]
            rollout_responses = [completion.text for completion in completions]

            training_started = time.perf_counter()
            loss, metadata = grpo_train_step(
                model=policy,
                tokenizer=tokenizer,
                optimizer=optimizer,
                gradient_accumulation_steps=config["gradient_accumulation_steps"],
                max_grad_norm=config["max_grad_norm"],
                reward_fn=reward_fn,
                repeated_prompts=repeated_prompts,
                rollout_responses=rollout_responses,
                repeated_ground_truths=repeated_ground_truths,
                group_size=config["group_size"],
            )
            training_seconds = time.perf_counter() - training_started

            grad_norm = metadata["total_norm"]
            if isinstance(grad_norm, torch.Tensor):
                grad_norm = grad_norm.item()
            loss_value = loss.detach().float().item()
            if not torch.isfinite(torch.tensor(loss_value)):
                raise FloatingPointError(f"Non-finite training loss at rollout step {step}: {loss_value}")

            sync_started = time.perf_counter()
            vllm_server.sync_policy_weights(policy)
            sync_seconds = time.perf_counter() - sync_started
            step_seconds = time.perf_counter() - step_started

            train_metadata = metadata["train_metadata"]
            rewards_metadata = metadata["rewards_metadata"]
            advantage_metadata = metadata["advantage_metadata"]
            loss_metadata = metadata["loss_metadata"]
            response_lengths = [len(completion.token_ids) for completion in completions]
            finish_reason_counts: dict[str, int] = {}
            for completion in completions:
                finish_reason = completion.finish_reason or "unknown"
                finish_reason_counts[finish_reason] = finish_reason_counts.get(finish_reason, 0) + 1

            train_metrics = {
                "train/loss": loss_value,
                "train/gradient_norm": float(grad_norm),
                "train/gradient_was_clipped": float(
                    config["max_grad_norm"] is not None
                    and grad_norm > config["max_grad_norm"]
                ),
                "train/learning_rate": optimizer.param_groups[0]["lr"],
                "train/token_entropy": train_metadata["mean_token_entropy"],
                "train/reward": rewards_metadata["mean_total_rewards"],
                "train/reward_std": rewards_metadata["std_total_rewards"],
                "train/format_reward": rewards_metadata["mean_format_rewards"],
                "train/answer_reward": rewards_metadata["mean_answer_rewards"],
                "train/nonzero_reward_fraction": rewards_metadata["nonzero_reward_fraction"],
                "train/average_response_length": sum(response_lengths) / len(response_lengths),
                "train/response_length_min": min(response_lengths),
                "train/response_length_max": max(response_lengths),
                "time/rollout_seconds": rollout_seconds,
                "time/training_seconds": training_seconds,
                "time/weight_sync_seconds": sync_seconds,
                "time/step_seconds": step_seconds,
                "throughput/rollouts_per_second": len(completions) / max(step_seconds, 1e-12),
                "throughput/response_tokens_per_second": train_metadata["num_response_tokens"] / max(step_seconds, 1e-12),
                "gpu/peak_allocated_gib": torch.cuda.max_memory_allocated(0) / (1024 ** 3),
                "gpu/peak_reserved_gib": torch.cuda.max_memory_reserved(0) / (1024 ** 3),
                "train/reward_histogram": wandb.Histogram(rewards_metadata["raw_rewards"].cpu().numpy()),
                "train/advantage_histogram": wandb.Histogram(advantage_metadata["advantages"].cpu().numpy()),
                "train/response_length_histogram": wandb.Histogram(response_lengths),
                "rollout_step": step,
            }
            add_scalar_metrics(train_metrics, "train/rewards", rewards_metadata)
            add_scalar_metrics(train_metrics, "train/advantages", advantage_metadata)
            add_scalar_metrics(train_metrics, "train/policy", loss_metadata)
            add_scalar_metrics(train_metrics, "train/tokens", train_metadata)
            for finish_reason, finish_count in finish_reason_counts.items():
                metric_name = finish_reason.replace(" ", "_").replace("/", "_")
                train_metrics[f"train/finish_reason_{metric_name}_fraction"] = (
                    finish_count / len(completions)
                )
            wandb_run.log(train_metrics)
            print(
                f"step={step} loss={train_metrics['train/loss']:.6f} "
                f"reward={train_metrics['train/reward']:.4f} "
                f"format_reward={train_metrics['train/format_reward']:.4f} "
                f"grad_norm={train_metrics['train/gradient_norm']:.4f} "
                f"entropy={train_metrics['train/token_entropy']:.4f} "
                f"step_seconds={step_seconds:.1f}",
                flush=True,
            )

            if step % config["log_rollouts_every"] == 0:
                table = wandb.Table(
                    columns=["prompt", "response", "ground_truth", "reward"]
                )
                for prompt, response, ground_truth in list(
                    zip(repeated_prompts, rollout_responses, repeated_ground_truths)
                )[: config["logged_rollouts"]]:
                    table.add_data(
                        prompt,
                        response,
                        ground_truth,
                        reward_fn(response, ground_truth)["reward"],
                    )
                wandb_run.log({"train/rollouts": table, "rollout_step": step})

            if step % config["eval_every"] == 0:
                evaluate(step)

            if config["save_every"] > 0 and step % config["save_every"] == 0:
                checkpoint_dir = output_dir / f"step-{step}"
                policy.save_pretrained(checkpoint_dir)
                tokenizer.save_pretrained(checkpoint_dir)

        policy.save_pretrained(output_dir / "final")
        tokenizer.save_pretrained(output_dir / "final")
        wandb_run.summary["final_checkpoint"] = str(output_dir / "final")
        wandb_run.summary["completed_rollout_steps"] = config["num_rollout_steps"]
    finally:
        vllm_server.stop()
        wandb_run.finish()


if __name__ == "__main__":
    main()
