import copy
import json
import re
from typing import Any, Dict, List

from eval.core.sample import Sample
from eval.core.base_task import BaseTask
from eval.core.utils import create_assistant_message, create_system_message, create_user_message
from eval.core.constant import COT_CODE


class BBH(BaseTask):
    """
    Big-Bench Hard (BBH)
    Website: https://github.com/suzgunmirac/BIG-Bench-Hard
    Language: English

    Single-choice and fill-in-the-blank question answering task.
    """

    task_name = "bbh"
    shots = [3]
    task_config = {
        "max_tokens": 2048,
        "temperature": 0.0,
        "stop": ["\n\nQuestion:", "<|im_end|>"],
    }

    def add_cot_code(self, text: str) -> str:
        return text + COT_CODE if text.endswith("\n") else text + " " + COT_CODE

    def load_prompt(self, file_path: str) -> List[Dict]:
        with open(file_path, "r") as f:
            prompt_data = json.load(f)
        messages = [create_system_message(prompt_data["system"])]
        for i, example in enumerate(prompt_data["examples"]):
            question = "\n".join(example["question"])

            solution_part = "\n".join(example["solution"])
            answer_part = example["answer"][0]
            all_part = [COT_CODE.strip(), solution_part, answer_part]
            print(all_part)

            answer = "\n\n".join(all_part)
            messages.append(create_user_message(question))
            messages.append(create_assistant_message(answer))

        return messages

    def load_samples(self) -> List[Sample]:
        """
        Load samples from file, and return a list of `Sample`.
        """
        with open("./eval/data/BBH/task_info.json", "r", encoding="utf-8") as f:
            task_info = json.load(f)
        free_from_tasks = task_info["FREE_FORM_TASKS"]
        multi_choice_tasks = task_info["MULTIPLE_CHOICE_TASKS"]
        samples = []
        for mode, task_list in zip(["free_form", "multiple_choice"], [free_from_tasks, multi_choice_tasks]):
            for task in task_list:
                task_data = json.load(open(f"eval/data/BBH/data/{task}.json"))
                messages = self.load_prompt(f"eval/data/BBH/lib_prompt_flex/{task}.json")
                for q_ in task_data["examples"]:
                    question = q_["input"]
                    current_messages = copy.deepcopy(messages)
                    current_messages.append(create_user_message(question))
                    if mode == "multiple_choice":
                        a = q_["target"][1]
                    elif mode == "free_form":
                        a = q_["target"]
                    raw_data = {
                        "question": q_["input"],
                        "answer": a,
                        "task": task,
                        "mode": mode,
                    }
                    sample = Sample(
                        raw_data=raw_data, messages=current_messages,
                        task_config=self.task_config,
                        # cpt_force_start=" " + COT_CODE + "\n\n",
                        # sft_force_start=COT_CODE + "\n\n",
                    )
                    samples.append(sample)
        return samples

    def extract_answer(self, row: Dict[str, Any]) -> Dict[str, Any]:
        INVALID_ANSWER = "[invalid]"
        ans = str(row["pred"])
        mode = row["mode"]
        # PROMPT_TYPE = os.environ.get("PROMPT_TYPE")
        # assert PROMPT_TYPE in ["CPT", "SFT"]
        # if PROMPT_TYPE == "CPT":
        #     ans = str(ans).split("\n\nQuestion:")[0]

        ans_line = ans.split("answer is ")
        # Expect to see 'answer is'. If not return whole string
        if len(ans_line) == 1:
            row["pred_answer"] = ans
            return row
        else:
            ans = ans_line[-1].strip()
        if mode == "multiple_choice":
            matches = re.findall(r"\([A-Z]\)", ans)
            if len(matches) != 1:
                # 选项多于 1 个全错
                row["pred_answer"] = INVALID_ANSWER
                return row
            else:
                ans = matches[0][1]  # (A)-> A
                row["pred_answer"] = ans
                return row
        elif mode == "free_form":
            ans = ans.split("\n")[0]
            if not ans:
                row["pred_answer"] = INVALID_ANSWER
                return row
            if ans[-1] == ".":
                ans = ans[:-1]
        row["pred_answer"] = ans
        return row

    def calculate_metrics_single(self, row: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate metrics for a single row. Return a dict of metrics.
        Model outputs are in `row["pred"]`.
        """
        return {
            "acc": int(row["pred_answer"].strip() == row["answer"]),
        }
