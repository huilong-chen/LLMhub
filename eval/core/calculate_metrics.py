import argparse
from typing import List
from transformers import AutoTokenizer
from eval.tasks.task_registry import TASK_REGISTRY

class MetricsCalculator:
    def __init__(self, task_names: List[str], model_path=None):
        self.tokenizer = None
        if model_path is not None:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
            except:
                self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.tasks = TASK_REGISTRY.get_tasks(task_names)

    def calculate(self, output_dir: str):
        all_task_results = []
        for task in self.tasks:
            avg_metrics = task.calculate_metrics(output_dir, tokenizer=self.tokenizer)

            # 指标统一为百分数，然后保留两位小数
            for key in avg_metrics.keys():
                avg_metrics[key] = round(avg_metrics[key] * 100, 2)

            if len(avg_metrics) == 1:
                # 如果只有一个指标，省略指标名称
                metric_value = list(avg_metrics.values())[0]
                all_task_results.append((task.task_name_with_shot, f"{metric_value}"))
                continue

            for metric_name, metric_value in avg_metrics.items():
                all_task_results.append(
                    (
                        f"{task.task_name_with_shot}-{metric_name}",
                        f"{metric_value}",
                    )
                )

        # 按逗号分割，打印结果
        names, values = zip(*all_task_results)
        names = ("任务名称", "输出路径") + names
        values = ("评测指标", output_dir) + values
        print(f"\n-----------评测指标-----------\n")
        print(",".join(names))
        print(",".join(values))
        print(f"\n-----------评测指标-----------\n")

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=False)
    parser.add_argument("--task_names", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()


def main():
    args = arg_parser()

    task_names = args.task_names.split(",")
    metrics_calculator = MetricsCalculator(task_names, args.model_path)
    if args.model_path:
        output_dir = args.model_path + args.output_dir
    else:
        output_dir = args.output_dir
    metrics_calculator.calculate(output_dir)


if __name__ == "__main__":
    main()
