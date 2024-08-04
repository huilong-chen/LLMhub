import copy
import pandas as pd
import re
from typing import Any, Dict, List
import os

from eval.core.sample import Sample
from eval.core.base_task import BaseTask
from eval.core.constant import create_system_message, create_user_message, create_assistant_message


class Mbpp(BaseTask):
    """
    MBPP
    Website: https://github.com/google-research/google-research/tree/master/mbpp
    Language: Code

    Code test

    Example：
        {
            "text": "Write a python function to remove first and last occurrence of a given character from the string.",
            "code": "def remove_Occ(s,ch): \r\n    for i in range(len(s)): \r\n        if (s[i] == ch): \r\n            s = s[0 : i] + s[i + 1:] \r\n            break\r\n    for i in range(len(s) - 1,-1,-1):  \r\n        if (s[i] == ch): \r\n            s = s[0 : i] + s[i + 1:] \r\n            break\r\n    return s ",
            "task_id": 11,
            "test_setup_code": "",
            "test_list": [
                "assert remove_Occ(\"hello\",\"l\") == \"heo\"",
                "assert remove_Occ(\"abcda\",\"a\") == \"bcd\"",
                "assert remove_Occ(\"PHP\",\"P\") == \"H\""
            ],
            "challenge_test_list": [
                "assert remove_Occ(\"hellolloll\",\"l\") == \"helollol\"",
                "assert remove_Occ(\"\",\"l\") == \"\""
            ]
        }
        sanitized:
        {
            "source_file": "Benchmark Questions Verification V2.ipynb",
            "task_id": 6,
            "prompt": "Write a python function to check whether the two numbers differ at one bit position only or not.",
            "code": "def is_Power_Of_Two (x): \n    return x and (not(x & (x - 1))) \ndef differ_At_One_Bit_Pos(a,b): \n    return is_Power_Of_Two(a ^ b)",
            "test_imports": [],
            "test_list": [
                "assert differ_At_One_Bit_Pos(13,9) == True",
                "assert differ_At_One_Bit_Pos(15,8) == False",
                "assert differ_At_One_Bit_Pos(2,4) == False",
                "assert differ_At_One_Bit_Pos(2, 3) == True",
                "assert differ_At_One_Bit_Pos(5, 1) == True",
                "assert differ_At_One_Bit_Pos(1, 5) == True"
            ]
        }
    """

    task_name = "mbpp"
    shots = [3]
    task_config = {
        "max_tokens": 512,
        "temperature": 0.0
    }

    user_prompt_format = "You are an expert Python programmer, and here is your task: {prompt} Your code should pass these tests:\n\n{tests}\n"
    assistant_prompt_format = "[BEGIN]\n{code}\n[DONE]"

    def load_samples(self) -> List[Sample]:
        """
        Load samples from file, and return a list of `Sample`.
        """
        df = pd.read_json("./eval/data/mbpp/mbpp.jsonl", lines=True)
        records = df.to_dict(orient="records")

        samples = []
        prompt_messages = []
        for record in records:
            task_id = record["task_id"]
            if task_id in [2, 3, 4]:
                user = self.user_prompt_format.replace("{prompt}", record["text"]) \
                          .replace("{tests}", "\n".join(record["test_list"] + record["challenge_test_list"]))
                assistant = self.assistant_prompt_format.replace("{code}", record["code"])
                prompt_messages.append(create_user_message(user))
                prompt_messages.append(create_assistant_message(assistant))

        for record in records:
            task_id = record["task_id"]
            if task_id not in [2, 3, 4]:
                continue
            user = self.user_prompt_format.replace("{prompt}", record["text"]) \
                .replace("{tests}", "\n".join(record["test_list"] + record["challenge_test_list"]))
            current_messages = copy.deepcopy(prompt_messages)
            current_messages.append(create_user_message(user))
            raw_data = {
                "question": record["text"],
                "task_id": record["task_id"],
                "code": record["code"],
                "test_list": record["test_list"],
                "challenge_test_list": record["challenge_test_list"],
            }
            sample = Sample(raw_data=raw_data, messages=current_messages, task_config=self.task_config)
            samples.append(sample)
        return samples

    def clean_answer(self, code):
        # 如果有``` ```框选的代码，我们只取框选的代码，同时去掉 def 那行
        code_part = re.search(r"```(python\n)*(def.*?\n)*(.*)```", code, re.DOTALL)
        if code_part:
            code = code_part.group(3)

        # 把属于输入的注释部分去掉
        code = code.split('"""')[-1]
        # 我们用``` ``` 框选了代码位置
        code = code.split("```")[0]
        # 这样的代码 tab 会多个空格，所以把多余的空格去掉
        code = re.sub(r" {5,}", lambda x: " " * (len(x.group()) - 1), code)
        # 去掉代码前面的换行
        if code.startswith("\n    "):
            code = code[1:]
        return code

    def valid_task(self, row: Dict[str, str]) -> Dict[str, str]:
        row["completion"] = self.clean_answer(str(row["pred"]))
        return row


    def is_correct_func(self, df: pd.DataFrame, task_name: str, col: str = "is_correct") -> float:
        # 依据指定列计算准确率，约定使用 is_correct 作为列名，仅包含 0 或 1
        right_count = len(df[df[col] == 1])
        all_count = len(df)
        print(f"@{task_name}-ACC:{right_count / all_count:.4f}({right_count}/{all_count})")
        return right_count / all_count

    def calculate_metrics(self, output_dir: str, tokenizer=None) -> Dict[str, float]:
        """
        Calculate average metrics based on the outputs file.
        """
        file_path = self.get_generate_path(output_dir)
        df = self.load_file_as_df(file_path)
        df = df.apply(self.valid_task, axis=1)

        df.to_json("./eval/data/mbpp/mbpp_eval_pred", orient="records", lines=True)
        cmd = f"evaluate_functional_correctness ./eval/data/mbpp/mbpp_pred"
        os.system(cmd)
        res = pd.read_json("./eval/data/mbppl/mbpp_pred_results.jsonl", lines=True).to_dict(orient="records")
        for r in res:
            r["is_correct"] = int(r["passed"])
        output_df = pd.DataFrame(res)

        file_path_with_metrics = os.path.join(output_dir, self.outputs_file_name)
        self.save_df_as_file(output_df, file_path_with_metrics)
        score = self.is_correct_func(output_df, "MBPP", "is_correct")
        return {"acc": score}

    def calculate_metrics_single(self, row: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate metrics for a single row. Return a dict of metrics.
        Model outputs are in `row["pred"]`.
        """
        pass

if __name__ == "__main__":
    mbpp = Mbpp()
    samples = mbpp.load_samples()
    for sample in samples:
        print(sample)