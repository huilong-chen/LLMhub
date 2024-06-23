import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import BertTokenizer
from transformers import DataCollatorWithPadding
from transformers import BertForSequenceClassification
from torch.optim import AdamW

from datasets import load_dataset
model = BertForSequenceClassification.from_pretrained(
        "/mnt/workspace/chenhuilong/model/chinese-bert-wwm-ext", num_labels=2)
tokenizer = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")

def preprocess_function(examples):
    # print(examples) # {'idx': 0, 'sentence': 'hide new secretions from the parental units ', 'label': 0}
    result = tokenizer(
        examples["sentence"],
        padding=False,
        max_length=128,
        truncation=True,
        return_token_type_ids=True, )
    if "label" in examples:
        result["labels"] = [examples["label"]]
    return result
dataset = load_dataset("stanfordnlp/sst2")['train']
print(len(dataset))
dataset = dataset.map(
    preprocess_function,
    batched=False,
    remove_columns=dataset.column_names,
    desc="Running tokenizer on dataset", )
dataset.set_format(
    "np", columns=["input_ids", "token_type_ids", "labels"])

sampler = torch.utils.data.SequentialSampler(dataset)
collate_fn = DataCollatorWithPadding(tokenizer)
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=4,
    sampler=sampler,
    num_workers=0,
    collate_fn=collate_fn, )

criterion = torch.nn.CrossEntropyLoss()
optimizer = AdamW(params=model.parameters(), lr=3e-5)

for batch in data_loader:
    output = model(**batch).logits
    loss = criterion(output, batch['labels'].reshape(-1))
    loss.backward()
    print(loss)
    optimizer.step()
    optimizer.zero_grad()
