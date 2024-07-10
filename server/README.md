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

## 基于vllm的OpenAI对话接口API

### 对话生成接口
/v1/chat/completions

Ref：https://platform.openai.com/docs/api-reference/chat/create

Request Body 样例：
```json
{
  "model": "chat-model-v1",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "What is the weather like today?"
    }
  ],
  "temperature": 0.9,
  "top_p": 0.95,
  "n": 1,
  "max_tokens": 64,
  "stop": ["\n", "\n\n"],
  "stream": false,
  "presence_penalty": 0.1,
  "frequency_penalty": 0.1,
  "logit_bias": {
    "greeting": -1.0
  },
  "user": "user123",
  "best_of": 1,
  "top_k": 50,
  "ignore_eos": false,
  "use_beam_search": true,
  "stop_token_ids": [50256, 50257],
  "skip_special_tokens": false,
  "spaces_between_special_tokens": false,
  "add_generation_prompt": true,
  "echo": false,
  "repetition_penalty": 1.1,
  "min_p": 0.05,
  "include_stop_str_in_output": true,
  "length_penalty": 1.2,
  "white_list_token_ids": [102, 103, 104],
  "white_list_token": ["hello", "world"],
  "force_prompt": "Always start with a greeting",
  "enable_repeat_stopper": true,
  "add_default_stop": true,
  "conversation_id": 42
}
```

Response Body 样例：
```json
{
  "id": "7a8edfee272a443b817481c5108cdc6a",
  "object": "chat.completion",
  "created": 100108,
  "model": "Meta-llama-3-8B-Instruct",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Here is an implementation of the quicksort algorithm"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 35,
    "total_tokens": 480,
    "completion_tokens": 445
  }
}
```

### 通用文本生成接口
/v1/completions

Ref：https://platform.openai.com/docs/api-reference/completions/create

Request Body 样例：
```json
{
  "model": "text-gen-model-v2",
  "prompt": "Once upon a time, in a land far away,",
  "suffix": ".",
  "max_tokens": 100,
  "temperature": 0.7,
  "top_p": 0.9,
  "n": 5,
  "stream": true,
  "logprobs": 5,
  "echo": true,
  "stop": ["?", "!"],
  "presence_penalty": 0.05,
  "frequency_penalty": 0.05,
  "best_of": 3,
  "logit_bias": {
    "adventure": 0.5
  },
  "user": "user456",
  "top_k": 40,
  "ignore_eos": true,
  "use_beam_search": false,
  "stop_token_ids": [50278],
  "skip_special_tokens": true,
  "spaces_between_special_tokens": true,
  "repetition_penalty": 1.2,
  "min_p": 0.1,
  "include_stop_str_in_output": false,
  "length_penalty": 1.0,
  "white_list_token_ids": [101, 102],
  "white_list_token": ["and", "but"],
  "enable_repeat_stopper": false,
  "add_default_stop": false
}
```

Response Body 样例：
```json
{
  "id": "cmpl-97ff9b2ed98348a19a99cee4f07d2c9b",
  "object": "text_completion",
  "created": 100544,
  "model": "Meta-llama-3-8B-Instruct",
  "choices": [
    {
      "index": 0,
      "text": "xxxxxxxxxx",
      "logprobs": null,
      "finish_reason": "length"
    }
  ],
  "usage": {
    "prompt_tokens": 5,
    "total_tokens": 2053,
    "completion_tokens": 2048
  }
}

```


