import argparse
import os
from collections import defaultdict
from functools import partial, lru_cache
from typing import List

from transformers import AutoTokenizer
from datasets import Dataset

IGNORE_INDEX = -100

def split_list(input_list, delimiter):
    last_start = 0
    splited = []
    for start in range(len(input_list)):
        if input_list[start: start + len(delimiter)] != delimiter:
            continue
        splited.append(input_list[last_start:start])
        last_start = start
    splited.append(input_list[last_start:])
    return splited

def split_messages_part(
        input_ids, human_token_ids, bot_token_ids, bos_token_ids=None
):
    splited = split_list(input_ids, human_token_ids)
    message_ids = []
    message_label_ids = []
    # user and assistant
    for turn_ids in splited:
        turn_splited = split_list(turn_ids, bot_token_ids)
        split_idx = len(turn_splited[0]) + len(bot_token_ids)
        turn_input_ids = turn_ids[:split_idx]
        turn_output_ids = turn_ids[split_idx:]
        message_ids.append(turn_input_ids + turn_output_ids)
        message_label_ids.append(
            [IGNORE_INDEX] * len(turn_input_ids) + turn_output_ids
        )
    return {
        "message_ids": message_ids,
        "message_label_ids": message_label_ids,
    }


def truncate_fn(max_len, examples):
    """
        按照最大长度切分
    """
    examples = [
        {key: examples[key][i] for key in examples.keys()}
        for i in range(len(examples["id"]))
    ]
    result = defaultdict(list)

    def add_example(input_ids, labels):
        result["input_ids"].append(input_ids)
        result["labels"].append(labels)
        result["length"].append(len(input_ids))

    # 开始切分
    for example in examples:
        input_ids = []
        labels = [IGNORE_INDEX] * len(input_ids)
        for turn_input_ids, turn_label_ids in zip(example["message_ids"], example["message_label_ids"]):
            # 当长度超过最大长度时，后面数据超长的部分直接丢弃
            if len(input_ids) + len(turn_input_ids) > max_len:
                break
            input_ids += turn_input_ids
            labels += turn_label_ids
        add_example(input_ids, labels)
    return result

def process_data(raw_data, tokenizer, human_token_ids, bot_token_ids, bos_token_id, max_length):
    def tokenize_fn(example):
        input_ids = tokenizer.apply_chat_template(
            example["messages"], tokenize=True
        )
        bos_token_ids = None
        if bos_token_id and input_ids[0] != bos_token_id:
            bos_token_ids = [bos_token_id]
        example.update(
            split_messages_part(
                input_ids, human_token_ids, bot_token_ids, bos_token_ids
            )
        )
        return example
    num_proc = max(os.cpu_count() - 2, 1)
    tokenized_data = raw_data.map(tokenize_fn, num_proc=num_proc, desc="[Tokenize]")
    print(f"tokenize 后得到的样本个数 {len(tokenized_data)}")

    truncated_data = tokenized_data.map(
        partial(truncate_fn, max_length),
        batched=True,
        batch_size=None,
        remove_columns=tokenized_data.column_names,
        num_proc=num_proc,
        desc="[Truncate]"
    )
    print(f"truncate 后得到的样本个数 {len(truncated_data)}")
    return truncated_data

def easy_merge_lists(lists, append_token_id, max_length=8192):
    merged_lists = []
    current_list = []
    for lst in lists:
        if len(current_list) + len(lst) <= max_length - 1:
            if len(current_list) > 0:
                current_list.append(append_token_id)
            current_list.extend(lst)
        else:
            if current_list:
                if len(current_list) <= max_length - 1:
                    current_list.append(append_token_id)
                merged_lists.append(current_list)
            current_list = lst.copy()

    if current_list:
        if len(current_list) <= max_length - 1:
            current_list.append(append_token_id)
        merged_lists.append(current_list)
    return merged_lists

def packing_data(dataset, tokenizer, max_length):
    input_ids = dataset["input_ids"]
    new_input_ids = easy_merge_lists(input_ids, tokenizer.eos_token_id, max_length)
    labels = dataset["labels"]
    new_labels = easy_merge_lists(labels, -100, max_length)
    length = [len(i) for i in new_input_ids]
    for i, j in zip(new_labels, new_input_ids):
        assert len(i) == len(j)
    return Dataset.from_dict(
        {"input_ids": new_input_ids, "labels": new_labels, "length": length}
    )

@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a mapping to unicode strings. We specifically avoids mapping to whitespace/control
    characters the bpe code barfs on.

    The reversible bpe codes work on unicode strings. This means you need a large # of unicode characters in your vocab
    if you want to avoid UNKs. When you're at something like a 10B token dataset you end up needing around 5K for
    decent coverage. This is a significant percentage of your normal, say, 32K bpe vocab. To avoid that, we want lookup
    tables between utf-8 bytes and unicode strings.
    """
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

def split_on_value(seq: List[int], value: int):
    # 根据 value 对 seq 进行切割
    results = []
    current_group = []
    for item in seq:
        if item == value:
            if current_group:
                results.append(current_group)
                current_group = []
        else:
            current_group.append(item)
    if current_group:  # Added check to add the last group if it's not empty
        results.append(current_group)
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file_path", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--max_seq_length", required=True, type=int)
    parser.add_argument("--tokenizer_path", required=True, type=str)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True, use_fast=True)

    print(tokenizer.chat_template)

    bos_token_id = tokenizer.bos_token_id
    human_token_ids = (
            tokenizer.convert_tokens_to_ids(["<|start_header_id|>", "user", "<|end_header_id|>"])
            + tokenizer("\n\n")["input_ids"]
    )
    bot_token_ids = (
            tokenizer.convert_tokens_to_ids(["<|start_header_id|>", "assistant", "<|end_header_id|>"])
            + tokenizer("\n\n")["input_ids"]
    )
    print(f"Bos Token Id: {bos_token_id}")
    print(f"Human Token Ids: {human_token_ids}")
    print(f"Bot Token Ids: {bot_token_ids}")

    raw_data = Dataset.from_json(args.input_file_path)
    print(len(raw_data))

    processed_data = process_data(raw_data, tokenizer, human_token_ids, bot_token_ids, bos_token_id, args.max_seq_length)

    print("打印处理后的第一条数据")
    print(processed_data["input_ids"][0])
    print(tokenizer.decode(processed_data["input_ids"][0]))

    packed_data = packing_data(processed_data, tokenizer, args.max_seq_length)

    print(f"Packing 后的样本数量: {len(packed_data)}")
    print("打印 Packing后的第一条数据")
    print(processed_data["input_ids"][0])
    print(tokenizer.decode(processed_data["input_ids"][0]))

    packed_data.save_to_disk(args.output_dir)

if __name__ == "__main__":
    main()