import os
import torch
print(os.getcwd())

import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq

os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

df = pd.read_json("../../data/Code-Feedback/Code-Feedback.jsonl", lines=True)
ds = Dataset.from_pandas(df)
train_test_split = ds.train_test_split(test_size=0.1)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']

model_path = "../../model/Qwen2-7B"
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)


def process_func(example):
    MAX_LENGTH = 4096
    messages = example["messages"]
    messages_len = len(messages)
    system = "<|im_start|>system\nYou are a code master.<|im_end|>\n"
    input = system
    for i in range(messages_len - 1):
        role = messages[i]["role"]
        content = messages[i]["content"]
        input += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    input += "<|im_start|>assistant\n"
    output = messages[messages_len - 1]["content"]

    request = tokenizer(input, add_special_tokens=False)
    response = tokenizer(output, add_special_tokens=False)
    input_ids = request["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = request["attention_mask"] + response["attention_mask"] + [1]  # EOS
    labels = [-100] * len(request["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

tokenized_train_dataset = train_dataset.map(process_func)
tokenized_test_dataset = eval_dataset.map(process_func)

model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16)
model.enable_input_require_grads()

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)

model = get_peft_model(model, config)
model.print_trainable_parameters()


args = TrainingArguments(
    output_dir="../../output/Qwen2_7B_lora",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    logging_steps=10,
    num_train_epochs=1,
    save_steps=100,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True
)
print(args)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

trainer.train()