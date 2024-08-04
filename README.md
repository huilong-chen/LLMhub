<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/huilong-chen/LLMhub/main/docs/images/logo/logo_light.png">
    <img alt="LLMhub" src="https://raw.githubusercontent.com/huilong-chen/LLMhub/main/docs/images/logo/logo_dark.png" width=55%>
  </picture>
</p>


---


# 自学仓库 - 大模型入门

## 仓库介绍

这个仓库是一个自学资源，旨在帮助初学者了解和掌握大模型的基本技术。内容涵盖了训练、微调、数据构建、评测、推理部署、分词器、长文本等各个方面。

## 关键技术

- **训练**：从头开始训练一个大模型，包括数据准备、模型设计、训练策略等。
- **微调**：基于预训练模型进行微调，以适应特定任务或领域的数据。
- **数据构建**：如何收集、清洗和处理数据，以便用于大模型的训练和评测。
- **评测**：模型性能的评估方法和指标，以及如何进行实验对比。
- **推理部署**：将训练好的模型部署到生产环境中，进行推理和应用。
- ...

[//]: # (## 更新日志)

[//]: # ()
[//]: # (### [v1.0.0] - 2024-08-04)

[//]: # (- 初始版本发布)

[//]: # (- 包含基础的训练、微调、数据构建、评测和推理部署示例)

## 安装和运行指南

### 先决条件

- Python 3.8+
- 必要的库依赖可以通过`requirements.txt`文件安装

### 安装

1. 克隆仓库
    ```bash
    git clone https://github.com/huilong-chen/LLMhub.git
    cd LLMhub
    ```

2. 创建并激活虚拟环境（可选）
    ```bash
    conda create -n llmhub python=3.8
    conda avtivate llmhub
    ```

3. 安装依赖
    ```bash
    pip install -r requirements.txt
    ```

### 运行示例

1. 模型部署
    ```bash
    bash scripts/start_server_llama3_8b.sh
    ```

2. 微调模型
    ```bash
    bash sft_llama3_8b_use_megatron.sh
    ```

3. 评测模型
    ```bash
    bash eval_online.sh
    ```

更多详细信息请参看代码。
