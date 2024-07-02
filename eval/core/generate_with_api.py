import argparse
import asyncio
import itertools
import random
from transformers import AutoTokenizer
from typing import List, Dict, Any
from tqdm.asyncio import tqdm
import aiohttp

from eval.tasks.task_registry import TASK_REGISTRY
from eval.core.sample import Sample

class Predictor:
    def __init__(self, model_path: str, task_names: List[str], devices: List[str]):
        self.model_path = model_path
        self.task_names = task_names
        self.devices = devices
        self.tasks = TASK_REGISTRY.get_tasks(self.task_names)

        # 在使用较新的 transformers 时，由于使用了 tiktoken，use_fast 必须设置为 True
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=False, trust_remote_code=True)
        except:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)

    async def predict(self, task_samples):
        samples_to_predict = list(itertools.chain(*task_samples))

        if len(samples_to_predict) == 0:
            raise ValueError("No samples to predict.")

        print(f"Total samples: {len(samples_to_predict)}.")
        random.shuffle(samples_to_predict)

        request_list = []
        for i, sample in enumerate(samples_to_predict):
            prompt = sample.messages[0]["content"]
            sample.prompts.append(prompt)
            request_list.append({"prompt": prompt, **sample.task_config})
            # Print some samples for review
            if i < 10:
                print(f"Sample {i} Prompt: {sample.prompts[-1]}")
        results = await self.batch_request(request_list)
        for sample, output in zip(samples_to_predict, results):
            sample.model_outputs.append(output)
        print("Generate finished.")

    def save_outputs(self, results: List[List[Sample]], output_dir: str):
        for task, result in zip(self.tasks, results):
            task.save_outputs(result, output_dir)

    async def batch_request(self, request_list: List[Dict[str, Any]]):
        connector = aiohttp.TCPConnector(limit=0, limit_per_host=200)
        timeout = aiohttp.ClientTimeout(total=None)
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            tasks = []
            for data in request_list:
                tasks.append(self.single_request(session, data))
            results = await tqdm.gather(*tasks)
        return results

    async def single_request(self, session, data):
        data["model"] = ""
        url = "http://127.0.0.1:6006"
        headers = {'Content-Type': 'application/json'}
        try:
            async with session.post(url, json=data, headers=headers) as response:
                if response.status != 200:
                    raise ValueError(f"Server error: {response.status}")
                return await response.json()
        except Exception as e:
            print(f"Request error")
            return {"error": f"Request error: {repr(e)}"}

    async def load_samples(self):
        task_samples = []
        for task in self.tasks:
            task.task_config["max_tokens"] = 0
            task.task_config["logprobs"] = 0
            task.task_config["echo"] = True
            samples = task.load_samples()
            print(f"Task {task.task_name_with_shot}: {len(samples)} samples.")
            task_samples.append(samples)
        return task_samples


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--task_names", type=str, required=True)
    parser.add_argument("--devices", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()


async def main():
    args = arg_parser()

    task_names = args.task_names.split(",")
    devices = args.devices.split(",")

    predictor = Predictor(args.model_path, task_names, devices)
    load_task = asyncio.create_task(predictor.load_samples())
    task_samples = await load_task

    await predictor.predict(task_samples)
    output_dir = args.model_path + args.output_dir
    predictor.save_outputs(task_samples, output_dir)

if __name__ == "__main__":
    # 启动一个异步函数
    asyncio.run(main())
