import openai

def get_llm_result(prompt) -> str:
    openai.api_base = "http://0.0.0.0:6006/v1"
    openai.api_key = "test"
    completion = openai.Completion.create(
        model="",
        prompt=prompt,
        temperature=0,
        top_p=1,
        max_tokens=2048,
        stream=False,
        enable_repeat_stopper=False,
    )
    result = completion.choices[0].text
    return result

if __name__ == "__main__":
    prompt = "使用python实现快速排序"
    print(get_llm_result(prompt))