import copy
import json
from typing import Any, Dict, List, Tuple

from eval.core.sample import Sample
from eval.core.base_task import BaseTask
from eval.core.constant import create_assistant_message, create_system_message, create_user_message


def parse_record(record: Dict[str, Any]) -> Tuple[str, str, str]:
    question = record["question"]
    choices = [f"{c}. {record[c]}" for c in "ABCD"]
    choices = "\n".join(choices)
    question = question + "\n" + choices
    answer = record["answer"]
    return question, choices, answer


class CEval(BaseTask):
    """
    C-Eval
    Website: https://cevalbenchmark.com/
    Language: Chinese

    Single-choice question answering task.
    """
    task_name = "ceval"
    shots = [0, 5]
    task_config = {
        "max_tokens": 1,
        "temperature": 0.0,
        "white_list_token": ["A", "B", "C", "D"],
    }

    def load_samples(self) -> List[Sample]:
        """
        Load samples from file, and return a list of `Sample`.
        """
        with open("./eval/data/c-eval/ceval_subject_mapping.json", "r", encoding="utf-8") as f:
            SUBJECT_DICT = json.load(f)

        with open("./eval/data/c-eval/task_info.json", "r", encoding="utf-8") as f:
            task_info = json.load(f)
        TASK2DESC, HARD_SUBSET = task_info["TASK2DESC"], task_info["HARD_SUBSET"]

        samples = []
        for task, subject in TASK2DESC.items():
            category = SUBJECT_DICT[task][2]

            val_data_path = "./eval/data/c-eval/val/" + task + "_val.csv"
            dev_data_path = "./eval/data/c-eval/dev/" + task + "_dev.csv"

            val_records = self.load_file_as_df(val_data_path).to_dict(orient="records")
            dev_records = self.load_file_as_df(dev_data_path).to_dict(orient="records")

            system_prompt = f"以下是中国关于{subject}考试的单项选择题，请选出其中的正确答案。"
            messages = [create_system_message(system_prompt)]

            # 构建 few shot
            for record in dev_records[: self.shot]:
                question, _, answer = parse_record(record)
                messages.append(create_user_message(question))
                messages.append(create_assistant_message(answer))

            # 构建测试集
            for record in val_records:
                question, choices, answer = parse_record(record)
                current_messages = copy.deepcopy(messages)
                current_messages.append(create_user_message(question))
                level = "hard" if task in HARD_SUBSET else "easy"
                raw_data = {
                    "question": question,
                    "choices": choices,
                    "answer": answer,
                    "task": task,
                    "level": level,
                    "category": category,
                }
                sample = Sample(raw_data=raw_data, messages=current_messages, task_config=self.task_config)
                samples.append(sample)
        return samples

    def extract_answer(self, row: Dict[str, Any]) -> Dict[str, Any]:
        pred = row["pred"].replace("'", '"')
        data = json.loads(pred)
        row["pred"] = data[0]['choices'][0]['message']['content']
        return row

    def calculate_metrics_single(self, row: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate metrics for a single row. Return a dict of metrics.
        Model outputs are in `row["pred"]`.
        """
        return {
            "acc": int(row["pred"].strip() == row["answer"]),
        }
