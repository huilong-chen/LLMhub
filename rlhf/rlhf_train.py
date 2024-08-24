import importlib
import multiprocessing

from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    HfArgumentParser,
    BitsAndBytesConfig,
)
import pandas as pd
import torch
import torch.nn as nn
from trl.trainer.utils import SIMPLE_QUERY_CHAT_TEMPLATE
from rlhf.common_args import CommonArgs
from trl.trainer.ppov2_trainer import PPOv2Trainer


def load_config(args):
    # 根据config_option加载相应的配置
    module_path = args.train_args_path.replace("/", ".").rstrip(".py")
    # 动态导入模块
    module = importlib.import_module(module_path)
    # 每个模块导入的类名均为TrainArgument
    class_name = args.rlhf_type + "Config"
    # 使用getattr获取模块中的类
    argument = getattr(module, class_name)
    train_argument = argument()
    return train_argument


def find_all_linear_names(model):
    """
    找出所有全连接层，为所有全连接添加adapter
    """
    cls = nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    lora_module_names = list(lora_module_names)
    return lora_module_names


def load_data_prompt(tokenizer, train_data_path, eval_samples):
    raw_datasets = pd.read_json(train_data_path, lines=True)
    for i in range(len(raw_datasets)):
        pro = raw_datasets['prompt'][i]
        res = tokenizer.apply_chat_template(pro, tokenize=False)
        raw_datasets.loc[i, 'prompt'] = res
    raw_datasets = Dataset.from_pandas(raw_datasets, preserve_index=False)

    def tokenize(element):
        outputs = tokenizer(
            element['prompt'],
            padding=False,
        )
        return {"input_ids": outputs["input_ids"]}

    raw_datasets = raw_datasets.map(
        tokenize,
        remove_columns=raw_datasets.column_names,
        batched=True,
        num_proc=multiprocessing.cpu_count(),  # multiprocessing.cpu_count(),
        load_from_cache_file=False,
    )
    train_dataset = raw_datasets.select(range(len(raw_datasets) - eval_samples))
    eval_dataset = raw_datasets.select(range(len(raw_datasets) - eval_samples, len(raw_datasets)))
    return train_dataset, eval_dataset

def main():
    parser = HfArgumentParser((CommonArgs,))
    args = parser.parse_args_into_dataclasses()[0]
    # 根据CommonArgs中的config_option动态加载配置
    config = load_config(args)

    # Model & Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.sft_model_path,
        padding_side="left",
        trust_remote_code=True,
    )

    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_QUERY_CHAT_TEMPLATE
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    model_kwargs = dict(trust_remote_code=True)

    if args.train_mode == 'qlora':
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16 if config.fp16 else torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
        model_kwargs.update(quantization_config=quantization_config)
    # 奖励模型
    reward_model = AutoModelForSequenceClassification.from_pretrained(config.reward_model_path, num_labels=1, **model_kwargs)
    # SFT模型
    ref_policy = AutoModelForCausalLM.from_pretrained(config.sft_model_path, **model_kwargs)
    policy = AutoModelForCausalLM.from_pretrained(config.sft_model_path, **model_kwargs)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=find_all_linear_names(policy),
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_dora=args.use_dora
    )
    if args.train_mode == 'lora':
        policy.enable_input_require_grads()
        policy = get_peft_model(policy, lora_config)
    elif args.train_mode == 'qlora':
        policy = prepare_model_for_kbit_training(policy, use_gradient_checkpointing=config.gradient_checkpointing)
        policy = get_peft_model(policy, lora_config)

    # Training
    train_dataset, eval_dataset = load_data_prompt(tokenizer, config.train_data_path, config.eval_samples)
    value_model = AutoModelForSequenceClassification.from_pretrained(config.reward_model_path, num_labels=1, trust_remote_code=True)
    trainer = PPOv2Trainer(
        config=config,
        tokenizer=tokenizer,
        policy=policy,
        ref_policy=ref_policy,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()
    trainer.save_model(config.output_dir)

if __name__ == "__main__":
    main()
