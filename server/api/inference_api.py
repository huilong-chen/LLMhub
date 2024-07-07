import requests
import json
import time

def get_completion(prompt):
    headers = {'Content-Type': 'application/json'}
    data = {"prompt": prompt}
    response = requests.post(url='http://127.0.0.1:6006', headers=headers, data=json.dumps(data))
    return response.json()['response']

if __name__ == '__main__':
    start_time = time.time()
    print(get_completion('hello')) # Hello! It's nice to meet you. Is there something I can help you with or would you like to chat?
    end_time = time.time()
    print(end_time - start_time)