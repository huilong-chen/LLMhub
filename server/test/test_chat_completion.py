import openai
import asyncio
import aiohttp

from typing import List, Dict, Any
from tqdm.asyncio import tqdm

def get_llm_result(conversation) -> str:
    openai.api_base = "http://0.0.0.0:6006/v1"
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

async def batch_request(request_list: List[Dict[str, Any]]):
    connector = aiohttp.TCPConnector(limit=0, limit_per_host=200)
    timeout = aiohttp.ClientTimeout(total=None)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = []
        for data in request_list:
            tasks.append(single_request(session, data))
        results = await tqdm.gather(*tasks)
    return results

async def single_request(session, data):
    data["model"] = ""
    port = 6006
    if "messages" not in data:
        url = f"http://0.0.0.0:{port}/v1/completions"
    else:
        url = f"http://0.0.0.0:{port}/v1/chat/completions"
    try:
        async with session.post(url, json=data) as response:
            if response.status != 200:
                raise ValueError(f"Server error: {response.status}")
            return await response.json()
    except Exception as e:
        print(f"Request error")
        return {"error": f"Request error: {repr(e)}"}
async def main():
    conversation = [{
        "role": "system",
        "content": "你是一个代码专家，请帮用户解决下面的问题。"
    }, {
        "role": "user",
        "content": "使用python实现快速排序算法"
    }]
    print(get_llm_result(conversation))
    request_list = [
        {'prompt': "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nThere are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?\nLet's think step by step.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nThere are 15 trees originally.\nThen there were 21 trees after some more were planted.\nSo there must have been 21 - 15 = 6.\nThe answer is 6.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nIf there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?\nLet's think step by step.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nThere are originally 3 cars.\n2 more cars arrive.\n3 + 2 = 5.\nThe answer is 5.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nLeah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?\nLet's think step by step.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nOriginally, Leah had 32 chocolates.\nHer sister had 42.\nSo in total they had 32 + 42 = 74.\nAfter eating 35, they had 74 - 35 = 39.\nThe answer is 39.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nJason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?\nLet's think step by step.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nJason started with 20 lollipops.\nThen he had 12 after giving some to Denny.\nSo he gave Denny 20 - 12 = 8.\nThe answer is 8.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nShawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?\nLet's think step by step.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nShawn started with 5 toys.\nIf he got 2 toys each from his mom and dad, then that is 4 more toys.\n5 + 4 = 9.\nThe answer is 9.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nThere were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?\nLet's think step by step.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nThere were originally 9 computers.\nFor each of 4 days, 5 more computers were added.\nSo 5 * 4 = 20 computers were added.\n9 + 20 is 29.\nThe answer is 29.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nMichael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?\nLet's think step by step.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nMichael started with 58 golf balls.\nAfter losing 23 on tues- day, he had 58 - 23 = 35.\nAfter losing 2 more, he had 35 - 2 = 33 golf balls.\nThe answer is 33.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nOlivia has $23. She bought five bagels for $3 each. How much money does she have left?\nLet's think step by step.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nOlivia had 23 dollars.\n5 bagels for 3 dollars each will be 5 x 3 = 15 dollars.\nSo she has 23 - 15 dollars left.\n23 - 15 is 8.\nThe answer is 8.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nSally buys one small pizza at $8 and one family-size pizza that costs 3 times as much as the small pizza. How much does she spend in all?\nLet's think step by step.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", 'max_tokens': 0, 'temperature': 0.0, 'stop': ['\nQuestion:'], 'logprobs': 0, 'echo': True},
        {'prompt': "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nThere are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?\nLet's think step by step.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nThere are 15 trees originally.\nThen there were 21 trees after some more were planted.\nSo there must have been 21 - 15 = 6.\nThe answer is 6.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nIf there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?\nLet's think step by step.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nThere are originally 3 cars.\n2 more cars arrive.\n3 + 2 = 5.\nThe answer is 5.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nLeah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?\nLet's think step by step.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nOriginally, Leah had 32 chocolates.\nHer sister had 42.\nSo in total they had 32 + 42 = 74.\nAfter eating 35, they had 74 - 35 = 39.\nThe answer is 39.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nJason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?\nLet's think step by step.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nJason started with 20 lollipops.\nThen he had 12 after giving some to Denny.\nSo he gave Denny 20 - 12 = 8.\nThe answer is 8.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nShawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?\nLet's think step by step.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nShawn started with 5 toys.\nIf he got 2 toys each from his mom and dad, then that is 4 more toys.\n5 + 4 = 9.\nThe answer is 9.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nThere were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?\nLet's think step by step.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nThere were originally 9 computers.\nFor each of 4 days, 5 more computers were added.\nSo 5 * 4 = 20 computers were added.\n9 + 20 is 29.\nThe answer is 29.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nMichael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?\nLet's think step by step.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nMichael started with 58 golf balls.\nAfter losing 23 on tues- day, he had 58 - 23 = 35.\nAfter losing 2 more, he had 35 - 2 = 33 golf balls.\nThe answer is 33.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nOlivia has $23. She bought five bagels for $3 each. How much money does she have left?\nLet's think step by step.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nOlivia had 23 dollars.\n5 bagels for 3 dollars each will be 5 x 3 = 15 dollars.\nSo she has 23 - 15 dollars left.\n23 - 15 is 8.\nThe answer is 8.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nSally buys one small pizza at $8 and one family-size pizza that costs 3 times as much as the small pizza. How much does she spend in all?\nLet's think step by step.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", 'max_tokens': 0, 'temperature': 0.0, 'stop': ['\nQuestion:'], 'logprobs': 0, 'echo': True},
        {'prompt': "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nThere are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?\nLet's think step by step.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nThere are 15 trees originally.\nThen there were 21 trees after some more were planted.\nSo there must have been 21 - 15 = 6.\nThe answer is 6.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nIf there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?\nLet's think step by step.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nThere are originally 3 cars.\n2 more cars arrive.\n3 + 2 = 5.\nThe answer is 5.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nLeah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?\nLet's think step by step.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nOriginally, Leah had 32 chocolates.\nHer sister had 42.\nSo in total they had 32 + 42 = 74.\nAfter eating 35, they had 74 - 35 = 39.\nThe answer is 39.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nJason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?\nLet's think step by step.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nJason started with 20 lollipops.\nThen he had 12 after giving some to Denny.\nSo he gave Denny 20 - 12 = 8.\nThe answer is 8.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nShawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?\nLet's think step by step.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nShawn started with 5 toys.\nIf he got 2 toys each from his mom and dad, then that is 4 more toys.\n5 + 4 = 9.\nThe answer is 9.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nThere were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?\nLet's think step by step.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nThere were originally 9 computers.\nFor each of 4 days, 5 more computers were added.\nSo 5 * 4 = 20 computers were added.\n9 + 20 is 29.\nThe answer is 29.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nMichael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?\nLet's think step by step.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nMichael started with 58 golf balls.\nAfter losing 23 on tues- day, he had 58 - 23 = 35.\nAfter losing 2 more, he had 35 - 2 = 33 golf balls.\nThe answer is 33.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nOlivia has $23. She bought five bagels for $3 each. How much money does she have left?\nLet's think step by step.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nOlivia had 23 dollars.\n5 bagels for 3 dollars each will be 5 x 3 = 15 dollars.\nSo she has 23 - 15 dollars left.\n23 - 15 is 8.\nThe answer is 8.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nSally buys one small pizza at $8 and one family-size pizza that costs 3 times as much as the small pizza. How much does she spend in all?\nLet's think step by step.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", 'max_tokens': 0, 'temperature': 0.0, 'stop': ['\nQuestion:'], 'logprobs': 0, 'echo': True}
    ]
    results = await batch_request(request_list)
    for result in results:
        print(result)

if __name__ == "__main__":
    asyncio.run(main())
