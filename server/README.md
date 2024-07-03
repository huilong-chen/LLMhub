# 模型部署
下文主要记录了 llama3-8b-instruct 模型的部署过程。
## 更新记录
- 2024.07.02 增加vllm加速推理
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

