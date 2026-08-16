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