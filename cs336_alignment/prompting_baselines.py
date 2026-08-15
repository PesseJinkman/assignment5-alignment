from cs336_alignment.vllm_utils import VLLMServer
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
import json


DATA_PATH = "data/gsm8k/test.jsonl"
PROMPT_PATH = "cs336_alignment/prompts/r1_zero.prompt"
MODEL_ID = "allenai/OLMo-2-0425-1B"

sampling_params = {}
sampling_params['stop'] = ["</answer>"]
sampling_params['include_stop_str_in_output'] = True
sampling_params["temperature"] = 1.0
sampling_params["top_p"] = 1.0
sampling_params["max_tokens"] = 512
sampling_params["n"] = 1
sampling_params["seed"] = 0


data = []

with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))

get_base_prompt = open(PROMPT_PATH, "r").read()

model_inputs = []
acutal_outputs = []
for d in data:
    model_inputs.append(get_base_prompt.format(question=d["question"]))
    acutal_outputs.append(d["answer"])

vllm_server = VLLMServer(model_id=MODEL_ID)
vllm_server.start()
model_outputs = vllm_server.generate_completions(
    prompts=model_inputs,
    sampling_params=sampling_params,
    batch_size=256
)
vllm_server.stop()

format_rewards = 0.0
answer_rewards = 0.0
rewards = 0.0

for i in range(len(data)):
    all_rewards = r1_zero_reward_fn(model_outputs[i].text, acutal_outputs[i])
    format_rewards += all_rewards['format_reward']
    answer_rewards += all_rewards['answer_reward']
    rewards += all_rewards['reward']

print("Format Rewards:", format_rewards)
print("Answer Rewards:", answer_rewards)
print("Rewards:", rewards)

n = len(data)

print("Average Format Reward:", format_rewards / n)
print("Average Answer Reward:", answer_rewards / n)
print("Average Reward:", rewards / n)