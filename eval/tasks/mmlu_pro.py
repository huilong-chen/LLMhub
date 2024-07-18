import json
import re
import copy

from typing import Dict, Any, List, Hashable

from eval.core.sample import Sample
from eval.core.base_task import BaseTask
from eval.core.utils import create_system_message, create_user_message, create_assistant_message
from eval.core.constant import COT_CODE

subjects = ["math", "health", "physics", "business", "biology", "chemistry", "computer science", "economics", "engineering", "philosophy", "other", "history", "law", "psychology"]
def format_example(question, options, cot_content=COT_CODE):
    if cot_content.startswith("A: "):
        cot_content = cot_content[3:]
    question_options = question
    choice_map = "ABCDEFGHIJ"
    for i, opt in enumerate(options):
        question_options += "{}. {}\n".format(choice_map[i], opt)
    answer = cot_content + "\n\n"
    return question_options, answer

def load_messages(dev_records: List[Dict[Hashable, str]], subject: str, n_shots: int) -> List[Dict[str, str]]:
    system_prompt = (f"The following are multiple-choice questions (with answers) about {subject}. "
                     f"Think step by step and then finish your answer with \"The answer is (X)\" where X is the correct letter choice.\n\n")
    messages = [
        create_system_message(system_prompt),
    ]
    for i, record in enumerate(dev_records[:n_shots]):
      question, answer = format_example(record["question"], record["options"], record["cot_content"])
      messages.append(create_user_message(question))
      messages.append(create_assistant_message(answer))
    return messages

class Mmlu_pro(BaseTask):
    """
        mmlu-pro
        Website: https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro
        Language: English

        Multi_choice question answering task.
        """

    task_name = "mmlu_pro"
    shots = [5]
    task_config = {
        "max_tokens": 1024,
        "temperature": 0.0,
    }

    def load_samples(self) -> List[Sample]:
        with open("./eval/data/MMLU-Pro/data/test.jsonl", "r", encoding="utf-8") as f:
            test_js = f.readlines()
        with open("./eval/data/MMLU-Pro/data/dev.jsonl", "r", encoding="utf-8") as f:
            dev_js = f.readlines()
        test_dt = [json.loads(j) for j in test_js]
        dev_dt = [json.loads(j) for j in dev_js]

        samples = []
        for subject in subjects:
            dev_records = [i for i in dev_dt if i["category"] == subject]
            test_records = [i for i in test_dt if i["category"] == subject]

            messages = load_messages(dev_records, subject, n_shots=self.shot)

            for test_record in test_records:
                question = test_record["question"]
                options = test_record["options"]
                answer = test_record["answer"]
                category = test_record["category"]
                cot_content = test_record["cot_content"]
                raw_data = {
                    "question": question,
                    "choices": options,
                    "answer": answer,
                    "task": subject,
                    "category": category,
                }
                test_format_question, _ = format_example(question, options, cot_content)
                current_messages = copy.deepcopy(messages)
                current_messages.append(create_user_message(test_format_question))
                sample = Sample(
                    raw_data=raw_data,
                    messages=current_messages,
                    task_config=self.task_config,
                )
                samples.append(sample)
        return samples

    def extract_answer(self, row: Dict[str, Any]) -> Dict[str, Any]:
        text = str(row["pred"])
        pattern = r"answer is \(?([A-J])\)?"
        match = re.search(pattern, text)
        if match:
            row["pred_answer"] = match.group(1)
            return row
        else:
            print("1st answer extract failed\n" + text)
            match = re.search(r'.*[aA]nswer:\s*([A-J])', text)
            if match:
                row["pred_answer"] = match.group(1)
                return row
            else:
                pattern = r"[A-J](?=[^A-J]*$)"
                match = re.search(pattern, text)
                if match:
                    row["pred_answer"] = match.group(0)
                    return row
                else:
                    row["pred_answer"] = ""
                    return row

    def calculate_metrics_single(self, row: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate metrics for a single row. Return a dict of metrics.
        Model outputs are in `row["pred"]`, answers are in `row["pred_answer"]`.
        """
        is_correct = 1.0 if str(row["pred_answer"]).strip() == row["answer"] else 0.0
        return {
            "acc": is_correct,
        }

if __name__ == '__main__':
    task = Mmlu_pro()
    task.shot = 5
    samples = task.load_samples()
    print(samples[0])