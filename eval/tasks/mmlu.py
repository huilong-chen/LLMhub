import copy
import json
import os
from typing import Dict, Any, List, Hashable, Tuple

import pandas as pd

from eval.core.base_task import Sample
from eval.core.base_task import BaseTask
from eval.core.constant import create_system_message, create_user_message, create_assistant_message

TASKS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]


def get_formatted_question_and_answer(record: List[str]) -> Tuple[str, str]:
    """
    Note that the returned question has choices in it.
    """
    assert len(record) == 6, "data might be corrupted, each row should have 6 items."

    question = record[0]
    choices = record[1:5]  # 4 choices, e,g, [ 31%, 46%, 61%, 76% ]
    answer = record[5]  # ground truth, e.g. B

    choices_with_abcd = [f"{label}. {choice}" for label, choice in zip("ABCD", choices)]
    choices_str = "\n".join(choices_with_abcd)
    question_with_choices_str = f"{question}\n{choices_str}"

    return question_with_choices_str, answer


def load_messages(dev_records: List[Dict[Hashable, str]], subject: str, n_shots: int) -> List[Dict[str, str]]:
    subject = subject.replace("_", " ")
    system_prompt = f"The following are multiple choice questions (with answers) about {subject}."

    messages = [
        create_system_message(system_prompt),
    ]

    for i, record in enumerate(dev_records[:n_shots]):
        question, answer = get_formatted_question_and_answer(record)
        messages.append(create_user_message(question))
        messages.append(create_assistant_message(answer))
    return messages


class Mmlu(BaseTask):
    """
    MMLU from paper Measuring Massive Multitask Language Understanding. https://arxiv.org/pdf/2009.03300

    Website: https://github.com/hendrycks/test
    Language: Chinese

    This is a massive multitask test consisting of multiple-choice questions from various branches of knowledge. The test spans subjects in the humanities, social sciences, hard sciences, and other areas that are important for some people to learn. This covers 57 tasks including elementary mathematics, US history, computer science, law, and more. To attain high accuracy on this test, models must possess extensive world knowledge and problem solving ability.

    Example:
        As of 2019, about what percentage of Americans agree that the state is run for the benefit of all the people?
            31%
            46%
            61%
            76%
        B

    (example copyed from `data/MMLU/data/dev/global_facts_dev.csv`)
    """

    task_name = "mmlu"
    shots = [5]
    task_config = {
        "max_tokens": 1,
        "temperature": 0.0,
        "white_list_token": ["A", "B", "C", "D"],
    }

    def load_samples(self) -> List[Sample]:
        """
        Load samples from file, and return a list of `Sample`.
        """
        with open("eval/data/MMLU/data/mmlu_categories.json", "r", encoding="utf-8") as f:
            mmlu_categories = json.load(f)
        subcategories, categories = (
            mmlu_categories["subcategories"],
            mmlu_categories["categories"],
        )

        category_map = dict()
        for k_1, v_1 in subcategories.items():
            subject = v_1[0]
            for k_2, v_2 in categories.items():
                if subject in v_2:
                    category_map[k_1] = k_2

        samples = []
        for task in TASKS:
            dev_file_path = os.path.join("eval/data/MMLU/data/dev", task + "_dev.csv")
            dev_records = pd.read_csv(dev_file_path, header=None).values.tolist()
            test_file_path = os.path.join("eval/data/MMLU/data/test", task + "_test.csv")
            test_records = pd.read_csv(test_file_path, header=None).values.tolist()

            messages = load_messages(dev_records, task, self.shot)
            for index, record in enumerate(test_records):
                question = record[0]
                choices = record[1:5]  # 4 choices, e.g., [ 31%, 46%, 61%, 76% ]
                answer = record[5]     # ground truth, e.g., B

                category = category_map[task]
                raw_data = {
                    "question": question,
                    "choices": choices,
                    "answer": answer,
                    "task": task,
                    "category": category,
                    "generation_prompt": messages,
                }

                question, answer = get_formatted_question_and_answer(record)

                current_messages = copy.deepcopy(messages)
                current_messages.append(create_user_message(question))

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
        is_correct = 1.0 if row["pred"].strip() == row["answer"] else 0.0
        return {
            "acc": is_correct,
        }


if __name__ == "__main__":
    task = Mmlu()
    task.shot = 5
    samples = task.load_samples()
    print(task)
    print(samples)
