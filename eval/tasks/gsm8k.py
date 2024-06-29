import copy
import json
import re
from typing import Any, Dict, List

import pandas as pd
from eval.core.base_task import Sample
from eval.core.base_task import BaseTask
from eval.core.constant import create_assistant_message, create_user_message
from eval.core.constant import COT_CODE


def eval_answer(pred: str, label: str) -> int:
    pred = pred.strip()
    try:
        # 把 7,333 转成 7333
        pred = eval(pred.replace(",", ""))
        label = eval(label.replace(",", ""))
    except:
        return 0

    if pred == label:
        return 1
    else:
        return 0


class Gsm8k(BaseTask):
    """
    gsm8k
    Website: https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard
    Language: English

    Grade school math task.
    """

    task_name = "gsm8k"
    shots = [0, 8]
    task_config = {"max_tokens": 1024, "temperature": 0.0, "stop": ["\nQuestion:"]}

    def load_samples(self) -> List[Sample]:
        """
        Load samples from file, and return a list of `Sample`.
        """

        with open("eval/data/gsm-8k/prompt_flex.json", "r") as f:
            flex_prompt = json.load(f)
        if self.shot == 0:
            prompt_messages = []
        else:
            assert self.shot == 8
            prompt_messages = []

            for prompt in flex_prompt:
                question = prompt["question"] + "\n" + COT_CODE
                answer = "\n".join(prompt["step_answers"])
                prompt_messages.append(create_user_message(question))
                prompt_messages.append(create_assistant_message(answer))

        records = pd.read_json("eval/data/gsm-8k/test.jsonl", lines=True).to_dict(
            orient="records"
        )
        # 清洗分割答案
        for i, record in enumerate(records):
            records[i]["answer"] = record["answer"].split("####")[-1].strip()

        samples = []
        for record in records:
            # TODO: 把 COT-CODE 移到 assistant 后面, 用统一函数处理
            text = record["question"] + "\n" + COT_CODE
            current_messages = copy.deepcopy(prompt_messages)
            current_messages.append(create_user_message(text))
            raw_data = {
                "question": record["question"],
                "answer": record["answer"],
                "messages": current_messages,
            }
            sample = Sample(
                raw_data=raw_data,
                messages=current_messages,
                task_config=self.task_config,
            )
            samples.append(sample)

        return samples

    def extract_answer(self, row: Dict[str, Any]) -> Dict[str, Any]:
        # 极端情况 7,333.33
        row["pred"] = str(row["pred"])
        number_pattern = r"\d+(,\d+)*(\.\d+)?"
        # answer is 后面的答案
        strict_pattern = rf"answer is\D*?({number_pattern})"
        strict_answer = re.search(strict_pattern, row["pred"])
        if strict_answer:
            row["relaxed_answer"] = row["strict_answer"] = strict_answer.group(1)
            return row
        # 如果没有生成 answer is，则取最后一个数字
        relax_pattern = rf"({number_pattern})\D*?$"
        relax_answer = re.search(relax_pattern, row["pred"])
        row["strict_answer"] = ""
        row["relaxed_answer"] = relax_answer.group(1) if relax_answer else ""
        return row

    def calculate_metrics_single(self, row: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate metrics for a single row. Return a dict of metrics.
        Model outputs are in `row["pred"]`.
        """
        if self.shot != 0:
            return {
                "strict": eval_answer(row["strict_answer"], row["answer"]),
                "relaxed": eval_answer(row["relaxed_answer"], row["answer"]),
            }
        else:
            # 0-shot 只返回宽松格式的准确率
            return {
                "relaxed": eval_answer(row["relaxed_answer"], row["answer"]),
            }
