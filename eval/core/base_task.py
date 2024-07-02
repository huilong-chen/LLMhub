import ast
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import pandas as pd
from .sample import Sample

class BaseTask(ABC):
    task_name: str = "base"
    shots: List[int] = []
    task_config: Dict[str, Any] = {}

    def __init__(self, shot: Optional[int] = None):
        self.shot = shot

    @abstractmethod
    def load_samples(self) -> List[Sample]:
        """
        Load samples from file
        """
        ...

    def save_outputs(self, samples: List[Sample], output_dir: str):
        """
        Save results to file, including raw data, prompts, and model outputs.
        """
        all_contents = []
        for sample in samples:
            raw_data = sample.raw_data
            raw_data["messages"] = sample.messages
            raw_data["prompt"] = sample.prompts[0] if len(sample.prompts) == 1 else sample.prompts

            try:
                choice = sample.model_outputs[0]["choices"][0]
                if "logprobs" in choice and choice["logprobs"] is not None:
                    raw_data["logits"] = choice["logprobs"]["token_logprobs"]
                outputs = [choice["text"] for out in sample.model_outputs for choice in out["choices"]]
                raw_data["pred"] = outputs[0] if len(outputs) == 1 else outputs
            except:
                # 返回结果不合法，可能因为输入过长或其他问题，直接保存原始输出
                raw_data["pred"] = sample.model_outputs
            all_contents.append(raw_data)

        df = pd.DataFrame(all_contents)
        file_path = self.get_generate_path(output_dir)
        self.save_df_as_file(df, file_path)
        logging.info(f"Task {self.task_name_with_shot} saved to {file_path}")

    @property
    def outputs_file_name(self):
        if self.shot is None:
            return f"{self.task_name}.csv"
        return f"{self.task_name}_{self.shot}_shot.csv"

    def get_generate_path(self, output_dir):
        output_dir = os.path.join(output_dir, "generate")
        os.makedirs(output_dir, exist_ok=True)
        return os.path.join(output_dir, self.outputs_file_name)

    @property
    def task_name_with_shot(self):
        if self.shot is None:
            return f"{self.task_name}"
        return f"{self.task_name}-{self.shot}"

    @abstractmethod
    def calculate_metrics_single(self, row: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate metrics for a single row. Return a dict of metrics.
        Model outputs are in `row["pred"]`.
        """
        ...

    def extract_answer(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract the answer from the raw data.
        Optional method, only used for some tasks when you need to save intermediate results.
        """
        return row

    def calculate_metrics(self, output_dir: str, tokenizer=None) -> Dict[str, float]:
        """
        Calculate average metrics based on the outputs file.
        """
        file_path = self.get_generate_path(output_dir)
        df = self.load_file_as_df(file_path).fillna("")

        # extract answer for per row
        df = df.apply(lambda row: pd.Series(self.extract_answer(row.to_dict())), axis=1)
        # calculate metrics for per row
        df_metrics = df.apply(lambda row: pd.Series(self.calculate_metrics_single(row.to_dict())), axis=1)
        # calculate mean metrics
        avg_metrics = df_metrics.mean().to_dict()
        df_with_metrics = pd.concat([df, df_metrics], axis=1)
        file_path_with_metrics = os.path.join(output_dir, self.outputs_file_name)
        self.save_df_as_file(df_with_metrics, file_path_with_metrics)
        logging.info(f"Task {self.task_name_with_shot} saved to {file_path_with_metrics}")
        return avg_metrics

    def calculate_loss(self, output_dir: str) -> Dict[str, float]:
        """
        Calculate average metrics based on the outputs file.
        """
        file_path = self.get_generate_path(output_dir)
        df = self.load_file_as_df(file_path).fillna("")
        if self.outputs_file_name.endswith("csv"):
            df["logits"] = df["logits"].apply(ast.literal_eval)
        loss = df["logits"].apply(lambda row: -sum(row[1:]) / len(row[1:])).mean()
        return loss

    def load_file_as_df(self, file_path) -> pd.DataFrame:
        file_extension = file_path.split(".")[-1]
        try:
            if "csv" == file_extension:
                df = pd.read_csv(file_path, escapechar="\\")
            elif "xlsx" == file_extension:
                df = pd.read_excel(file_path)
            elif "json" == file_extension:
                df = pd.read_json(file_path)
            elif "jsonl" == file_extension:
                df = pd.read_json(file_path, lines=True)
            else:
                raise NotImplementedError(
                    f"only support json, xlsx, csv file, got file {file_path} with extension {file_extension}."
                )
            return df
        except Exception as e:
            logging.error(f"Failed to load file {file_path}. Error: {e}")
            return pd.DataFrame()

    def save_df_as_file(self, df, file_path):
        file_extension = file_path.split(".")[-1]
        try:
            if "csv" == file_extension:
                df.to_csv(file_path, escapechar="\\", index=False)
            elif "xlsx" == file_extension:
                df.to_excel(file_path, index=False)
            elif "json" == file_extension:
                df.to_json(file_path, orient="records", force_ascii=False)
            elif "jsonl" == file_extension:
                df.to_json(file_path, orient="records", lines=True, force_ascii=False)
            else:
                raise NotImplementedError(
                    f"only support json, xlsx, csv file, got file {file_path} with extension {file_extension}."
                )
        except Exception as e:
            logging.error(f"Failed to save file {file_path}. Error: {e}")
        return file_path
