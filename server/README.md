# 模型部署
下文主要记录了 llama3-8b-instruct 模型的部署过程。
## 更新记录
- 2024.07.07 基于vllm实现与OpenAI对话接口类似的API
- 2024.07.02 增加vllm加速推理(单卡)
- 2024.06.30 增加基于FastApi的部署调用

## 基于FastApi的部署调用
参考Repo：https://github.com/datawhalechina/self-llm/blob/master/LLaMA3/

运行 `server/api.py` 在单卡上启动服务
- 使用`Python`代码
  - 运行 `server/inference_api.py` 进行测试。
- 使用`api`接口
  - ```cmd
    curl -X POST "http://127.0.0.1:6006" \
     -H 'Content-Type: application/json' \
     -d '{"prompt": "Hello, what is your name?"}' 
    ```
    
## 基于vllm的推理加速
1. 加载模型
```python
model = LLM(model=model_name_or_path, dtype='float16')
```
2. 调用 `generate` 推理
```python
input_ids = tokenizer.encode(input_str, add_special_tokens=False)
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, repetition_penalty=1.1, max_tokens=512)
outputs = model.generate(prompts=None, sampling_params=sampling_params, prompt_token_ids=[input_ids])
response = outputs[0].outputs[0].text
```
3. 可减少80%的推理时间

