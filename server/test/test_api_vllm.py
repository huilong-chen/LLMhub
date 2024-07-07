import openai

def get_llm_result(conversation) -> str:
    openai.api_base = "http://10.41.36.56:6006/v1"
    openai.api_key = "test"
    chat_completion = openai.ChatCompletion.create(
        model="",
        messages=conversation,
        temperature=0,
        top_p=1,
        max_tokens=2048,
        stream=False,
        enable_repeat_stopper=False,
    )
    result = chat_completion.choices[0].message.content
    return result

if __name__ == "__main__":
    conversation = [{
        "role": "system",
        "content": "你是一个代码专家，请帮用户解决下面的问题。"
    }, {
        "role": "user",
        "content": "使用python实现快速排序算法"
    }]
    print(get_llm_result(conversation))