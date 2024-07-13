import copy
import pandas as pd
import re
from typing import Any, Dict, List
import os
import json

from eval.core.sample import Sample
from eval.core.base_task import BaseTask
from eval.core.utils import create_system_message, create_user_message


class HumanEval(BaseTask):
    """
    human-eval
    Website: https://github.com/openai/human-eval  &&  https://hub.opencompass.org.cn/dataset-detail/HumanEval
    Language: Code

    Code test

    Example：
        {
            "task_id": "HumanEval/10",
            "prompt": "\n\ndef is_palindrome(string: str) -> bool:\n    \"\"\" Test if given string is a palindrome \"\"\"\n    return string == string[::-1]\n\n\ndef make_palindrome(string: str) -> str:\n    \"\"\" Find the shortest palindrome that begins with a supplied string.\n    Algorithm idea is simple:\n    - Find the longest postfix of supplied string that is a palindrome.\n    - Append to the end of the string reverse of a string prefix that comes before the palindromic suffix.\n    >>> make_palindrome('')\n    ''\n    >>> make_palindrome('cat')\n    'catac'\n    >>> make_palindrome('cata')\n    'catac'\n    \"\"\"\n",
            "entry_point": "make_palindrome",
            "canonical_solution": "    if not string:\n        return ''\n\n    beginning_of_suffix = 0\n\n    while not is_palindrome(string[beginning_of_suffix:]):\n        beginning_of_suffix += 1\n\n    return string + string[:beginning_of_suffix][::-1]\n",
            "test": "\n\nMETADATA = {\n    'author': 'jt',\n    'dataset': 'test'\n}\n\n\ndef check(candidate):\n    assert candidate('') == ''\n    assert candidate('x') == 'x'\n    assert candidate('xyz') == 'xyzyx'\n    assert candidate('xyx') == 'xyx'\n    assert candidate('jerry') == 'jerryrrej'\n"
        }

        {
            "task_id": "HumanEval/89",
            "prompt": "\ndef encrypt(s):\n    \"\"\"Create a function encrypt that takes a string as an argument and\n    returns a string encrypted with the alphabet being rotated. \n    The alphabet should be rotated in a manner such that the letters \n    shift down by two multiplied to two places.\n    For example:\n    encrypt('hi') returns 'lm'\n    encrypt('asdfghjkl') returns 'ewhjklnop'\n    encrypt('gf') returns 'kj'\n    encrypt('et') returns 'ix'\n    \"\"\"\n",
            "entry_point": "encrypt",
            "canonical_solution": "    d = 'abcdefghijklmnopqrstuvwxyz'\n    out = ''\n    for c in s:\n        if c in d:\n            out += d[(d.index(c)+2*2) % 26]\n        else:\n            out += c\n    return out\n",
            "test": "def check(candidate):\n\n    # Check some simple cases\n    assert candidate('hi') == 'lm', \"This prints if this assert fails 1 (good for debugging!)\"\n    assert candidate('asdfghjkl') == 'ewhjklnop', \"This prints if this assert fails 1 (good for debugging!)\"\n    assert candidate('gf') == 'kj', \"This prints if this assert fails 1 (good for debugging!)\"\n    assert candidate('et') == 'ix', \"This prints if this assert fails 1 (good for debugging!)\"\n\n    assert candidate('faewfawefaewg')=='jeiajeaijeiak', \"This prints if this assert fails 1 (good for debugging!)\"\n    assert candidate('hellomyfriend')=='lippsqcjvmirh', \"This prints if this assert fails 2 (good for debugging!)\"\n    assert candidate('dxzdlmnilfuhmilufhlihufnmlimnufhlimnufhfucufh')=='hbdhpqrmpjylqmpyjlpmlyjrqpmqryjlpmqryjljygyjl', \"This prints if this assert fails 3 (good for debugging!)\"\n\n    # Check some edge cases that are easy to work out by hand.\n    assert candidate('a')=='e', \"This prints if this assert fails 2 (also good for debugging!)\"\n\n"
        }
    """

    task_name = "human_eval"
    task_config = {
        "max_tokens": 512,
        "temperature": 0.0
    }

    def load_samples(self) -> List[Sample]:
        """
        Load samples from file, and return a list of `Sample`.
        """
        df = pd.read_json("./eval/data/human-eval/HumanEval.jsonl", lines=True)
        df = df.rename(columns={"prompt": "question"})
        records = df.to_dict(orient="records")

        # 把 human-eval 的题目加上 markdown 格式，方便分割输出，用于预测
        system_prompt = "Given the initial segment of a Python code, complete the rest ensuring proper syntax and functionality."
        samples = []

        for record in records:
            messages = [create_system_message(system_prompt)]
            question = "```\n" + record["question"]
            current_messages = copy.deepcopy(messages)
            current_messages.append(create_user_message(question))
            raw_data = {
                "question": question,
                "task_id": record["task_id"],
                "entry_point": record["entry_point"],
                "canonical_solution": record["canonical_solution"],
                "test": record["test"]
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

        df.to_json("./eval/data/human-eval/human_eval_pred", orient="records", lines=True)
        cmd = f"evaluate_functional_correctness ./eval/data/human-eval/human_eval_pred"
        os.system(cmd)
        res = pd.read_json("./eval/data/human-eval/human_eval_pred_results.jsonl", lines=True).to_dict(orient="records")
        for r in res:
            r["is_correct"] = int(r["passed"])
        output_df = pd.DataFrame(res)

        file_path_with_metrics = os.path.join(output_dir, self.outputs_file_name)
        self.save_df_as_file(output_df, file_path_with_metrics)
        score = self.is_correct_func(output_df, "Human-Eval", "is_correct")
        return {"acc": score}

    def calculate_metrics_single(self, row: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate metrics for a single row. Return a dict of metrics.
        Model outputs are in `row["pred"]`.
        """
        pass
