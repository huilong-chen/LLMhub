import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# from vllm import LLM
#
# llm = LLM(model="/mnt/data/chenhuilong/model/Meta-llama-3-8B-Instruct")  # Name or path of your model
# output = llm.generate("Hello, my name is")
# print(output)

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(model="/mnt/data/chenhuilong/model/Meta-llama-3-8B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("/mnt/data/chenhuilong/model/Meta-llama-3-8B-Instruct")
input_ids = [tokenizer.encode(prompt, add_special_tokens=True) for prompt in prompts]
print(input_ids)
outputs = llm.generate(prompts=None, sampling_params=sampling_params, prompt_token_ids=input_ids)

# Print the outputs.
for output in outputs:
    print(output)
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")