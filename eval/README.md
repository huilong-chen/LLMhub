# 大模型评测

## 更新记录🔥
- 2024.07.13 支持 **HumanEval** 代码评测
- 2024.07.08 支持 **CEval** 中文评测
- 2024.06.25 支持 **MMLU** 和 **GSM8K**
- 2024.06.23 增加模型评测模块

## 常见大模型评测数据集分类

1. 代码
- [HumanEval](https://github.com/openai/human-eval)：一个用于评估代码生成模型的基准数据集。它包含一组编程问题和相应的单元测试，模型需要根据问题描述生成正确的 Python 代码。
- [MBPP](https://huggingface.co/datasets/google-research-datasets/mbpp)：Mostly Basic Programming Problems，包含500个基本编程问题，每个问题都附有描述、输入输出示例和参考解决方案。
2. 综合推理
- [BBH](https://github.com/suzgunmirac/BIG-Bench-Hard)：Big-Bench Hard，由多个任务组成的大规模综合推理基准数据集。任务涵盖语言理解、数学推理、常识推理等多个领域。
3. 世界知识
- ✅ [MMLU](https://github.com/hendrycks/test)：Massive Multitask Language Understanding，MMLU 是一个多任务语言理解基准数据集，涵盖57个任务，涉及从初中到大学水平的广泛主题，如数学、历史、生物、法律等。
- [MMLU-Pro](https://github.com/TIGER-AI-Lab/MMLU-Pro?tab=readme-ov-file)：MMLU-Pro 是 MMLU 的扩展版本，包含更多高级和专业领域的任务，如医学、法律等。
4. 阅读理解
- [OpenBookQA](https://allenai.org/data/open-book-qa)：针对开放书本问题回答的基准数据集，问题基于中学科学课程，模型需要结合背景知识进行回答。
5. 数学
- ✅ [GSM8k](https://huggingface.co/datasets/openai/gsm8k)：GSM8K是一个专注于小学数学问题的数据集，它包含了8500个高质量的语言多样化小学数学单词问题。GSM8K数据集中的问题通常需要2到8步的推理才能解决，这使得它能够有效评估模型的数学与逻辑能力。
- ：[MATH](https://github.com/hendrycks/math/) 数据集包含数学竞赛题目，涵盖代数、几何、微积分等高级数学领域。
6. 中文
- [CEval](https://cevalbenchmark.com/index_zh.html): 一个全面的中文基础模型评估套件。它包含了13948个多项选择题，涵盖了52个不同的学科和四个难度级别。
- [CMMLU](https://github.com/haonan-li/CMMLU)：CMMLU 是 MMLU 的中文版本，涵盖类似的多任务和广泛主题，但重点在中文语言理解。


## 评测指标对齐
在这个项目中，主要针对 `Llama3-8B-Instruct` 和 `Qwen2-7B` 这两个开源的模型进行评测，目标是实现与他们公开的技术报告或Blog中的指标对齐，结果如下表。

|    D/M     | Llama3-8B-Instruct(ours) | Llama3-8B-Instruct(report) | Qwen2-7B(ours) | Qwen2-7B(report) |
|:----------:|:------------------------:|:--------------------------:|:--------------:|:----------------:|
| HumanEval  |          56.10           |            62.6            |     62.80      |       51.2       |
|    MBPP    |            -             |             -              |       -        |       65.9       |
|    BBH     |            -             |             -              |       -        |       62.6       |
|    MMLU    |          66.68           |            68.4            |     68.25      |       70.3       |
|  MMLU-Pro  |            -             |             -              |       -        |       40.0       |
| OpenBookQA |            -             |             -              |       -        |        -         |
|   GSM8k    |          72.33           |            79.6            |     79.98      |       79.9       |
|    MATH    |            -             |            30.0            |       -        |       44.2       |
|   CMMLU    |            -             |             -              |       -        |       83.9       |
|   CEval    |          52.08           |             -              |     79.35      |       83.9       |


TODO: 部分指标相差较大，待排查。