import argparse
import logging
import uvicorn
import os
import time
import asyncio

from fastapi import FastAPI, Request
from server.pipeline import Pipeline
from server.api.utils import get_api_key
from server.api.serving_chat import OpenAIServingChat
from server.api.serving_completion import OpenAIServingCompletion
from server.api.protocol import ChatCompletionRequest, ErrorResponse, CompletionRequest
from fastapi.routing import APIRoute
from typing import Callable
from http import HTTPStatus
from fastapi.responses import JSONResponse, Response

class LogRequestRoute(APIRoute):
    def get_route_handler(self) -> Callable:
        original_route_handler = super().get_route_handler()

        async def custom_route_handler(request: Request) -> Response:
            api_key = get_api_key(request.headers.get("Authorization", ""))
            request.state.api_key = api_key
            request.state.start_time = time.time()
            request.state.generated_text = None
            request.state.request_id = None
            request.state.finish_reason = None
            request.state.first_scheduled_time = None
            request.state.first_token_time = None
            request.state.streaming_code = None
            try:
                request.state.request_body = await request.json()
            except:
                request.state.request_body = {}
            try:
                response: Response = await original_route_handler(request)
            except asyncio.CancelledError:
                # 当非 Streaming 请求被取消时，会抛出这个异常
                response = Response(status_code=499)
            except ValueError as exc:
                # invalid input (e.g., token too long), return 400 bad request
                logging.exception(exc)
                code = HTTPStatus.BAD_REQUEST.value
                response = JSONResponse(
                    ErrorResponse(message=str(exc), type="invalid_request_error", code=code).model_dump(),
                    status_code=code,
                )
            except Exception as exc:
                code = HTTPStatus.INTERNAL_SERVER_ERROR.value
                logging.exception(exc)
                response = JSONResponse(
                    ErrorResponse(message=str(exc), type="internal_server_error", code=code).model_dump(),
                    status_code=code,
                )
            return response

        return custom_route_handler

app = FastAPI()
app.router.route_class = LogRequestRoute

openai_serving_chat: OpenAIServingChat = None
openai_serving_completion: OpenAIServingCompletion = None

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest, raw_request: Request):
    return await openai_serving_chat.create_chat_completion(request, raw_request)

@app.post("/v1/completions")
async def create_completion(request: CompletionRequest, raw_request: Request):
    return await openai_serving_completion.create_completion(request, raw_request)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--tensor_parallel", type=int, required=True, help="Number of tensor parallel")
    parser.add_argument("--port", type=int, required=True, help="Port of the server")
    parser.add_argument("--env", type=str, default=os.environ.get("ENV", "dev"), help="Environment")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    pipeline = Pipeline.create_pipeline(
        args.model_path,
        tensor_parallel_size=args.tensor_parallel,
    )
    openai_serving_chat = OpenAIServingChat(pipeline, args.model_name)
    openai_serving_completion = OpenAIServingCompletion(pipeline, args.model_name)

    logging.info("Starting api server at port: " + str(args.port))
    uvicorn.run(app, host="0.0.0.0", port=args.port, workers=1, timeout_keep_alive=5)