import asyncio
import logging
import time
from http import HTTPStatus
from typing import AsyncGenerator, AsyncIterator

from server.api.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    DeltaMessage,
    UsageInfo,
)
from server.api.serving_engine import OpenAIServing
from server.api.utils import random_uuid
from fastapi import Request
from server.pipeline import PipelineOutput
from starlette.responses import JSONResponse, StreamingResponse


class OpenAIServingChat(OpenAIServing):
    async def create_chat_completion(self, request: ChatCompletionRequest, raw_request: Request):
        """Completion API similar to OpenAI's API.

        See  https://platform.openai.com/docs/api-reference/chat/create
        for the API specification. This API mimics the OpenAI ChatCompletion API.

        """

        input_ids = self.pipeline.pre_process(
            request.messages, add_generation_prompt=request.add_generation_prompt, force_prompt=request.force_prompt
        )
        request_id = random_uuid()
        self._validate_prompt_and_tokenize(request, prompt_ids=input_ids)
        pipeline_params = request.to_pipeline_params(self.pipeline, raw_request.state.api_key)
        result_generator = self.pipeline.stream(input_ids, pipeline_params, request_id)
        raw_request.state.request_id = request_id
        if request.stream:
            stream_generator = self.chat_completion_stream_generator(request, raw_request, result_generator, request_id)
            return StreamingResponse(content=stream_generator, media_type="text/event-stream")
        else:
            response = await self.chat_completion_full_generator(request, raw_request, result_generator, request_id)
            return JSONResponse(content=response.model_dump())

    async def chat_completion_stream_generator(
        self,
        request: ChatCompletionRequest,
        raw_request: Request,
        result_generator: AsyncIterator[PipelineOutput],
        request_id: str,
    ) -> AsyncGenerator[str, None]:
        model_name = self.model_name
        created_time = int(time.monotonic())
        chunk_object_type = "chat.completion.chunk"

        # Send first response for each request.n (index) with the role
        role = "assistant"
        for i in range(request.n):
            choice_data = ChatCompletionResponseStreamChoice(index=i, delta=DeltaMessage(role=role), finish_reason=None)
            chunk = ChatCompletionStreamResponse(
                id=request_id, object=chunk_object_type, created=created_time, choices=[choice_data], model=model_name
            )
            data = chunk.model_dump_json(exclude_unset=True)
            yield f"data: {data}\n\n"

        # Send response to echo the input portion of the last message
        if request.echo:
            last_msg_content = ""
            if (
                request.messages
                and isinstance(request.messages, list)
                and request.messages[-1].get("content")
                and request.messages[-1].get("role") == role
            ):
                last_msg_content = request.messages[-1]["content"]
            if last_msg_content:
                for i in range(request.n):
                    choice_data = ChatCompletionResponseStreamChoice(
                        index=i, delta=DeltaMessage(content=last_msg_content), finish_reason=None
                    )
                    chunk = ChatCompletionStreamResponse(
                        id=request_id,
                        object=chunk_object_type,
                        created=created_time,
                        choices=[choice_data],
                        model=model_name,
                    )
                    data = chunk.model_dump_json(exclude_unset=True)
                    yield f"data: {data}\n\n"

        # Send response for each token for each request.n (index)
        previous_texts = [""] * request.n
        previous_num_tokens = [0] * request.n
        finish_reason_sent = [False] * request.n
        finish_reason_list = [None] * request.n
        raw_request.state.generated_text = previous_texts
        try:
            async for res in result_generator:
                res: PipelineOutput
                raw_request.state.first_scheduled_time = res.first_scheduled_time
                raw_request.state.first_token_time = res.first_token_time

                if await raw_request.is_disconnected():
                    # Abort the request if the client disconnects.
                    await self.pipeline.abort_stream_request(request_id)
                    raise asyncio.CancelledError

                for output in res.outputs:
                    i = output.index

                    if finish_reason_sent[i]:
                        continue

                    delta_text = output.text[len(previous_texts[i]) :]
                    previous_texts[i] = output.text
                    previous_num_tokens[i] = len(output.token_ids)

                    if output.finish_reason is None:
                        # Send token-by-token response for each request.n
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=i, delta=DeltaMessage(content=delta_text), finish_reason=None
                        )
                        chunk = ChatCompletionStreamResponse(
                            id=request_id,
                            object=chunk_object_type,
                            created=created_time,
                            choices=[choice_data],
                            model=model_name,
                        )
                        data = chunk.model_dump_json(exclude_unset=True)
                        yield f"data: {data}\n\n"
                    else:
                        finish_reason_list[i] = output.finish_reason
                        # Send the finish response for each request.n only once
                        prompt_tokens = len(res.prompt_token_ids)
                        final_usage = UsageInfo(
                            prompt_tokens=prompt_tokens,
                            completion_tokens=previous_num_tokens[i],
                            total_tokens=prompt_tokens + previous_num_tokens[i],
                        )
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=i, delta=DeltaMessage(content=delta_text), finish_reason=output.finish_reason
                        )
                        chunk = ChatCompletionStreamResponse(
                            id=request_id,
                            object=chunk_object_type,
                            created=created_time,
                            choices=[choice_data],
                            model=model_name,
                        )
                        if final_usage is not None:
                            chunk.usage = final_usage
                        data = chunk.model_dump_json(exclude_unset=True, exclude_none=True)
                        yield f"data: {data}\n\n"
                        finish_reason_sent[i] = True
        except asyncio.CancelledError:
            raw_request.state.streaming_code = 499
        except ValueError as e:
            code = HTTPStatus.BAD_REQUEST.value
            raw_request.state.streaming_code = code
            logging.exception(e)
            data = self.create_streaming_error_response(str(e), type="invalid_request_error", code=code)
            yield f"data: {data}\n\n"
        except BaseException as e:
            code = HTTPStatus.INTERNAL_SERVER_ERROR.value
            raw_request.state.streaming_code = code
            logging.exception(e)
            await self.pipeline.abort_stream_request(request_id)
            data = self.create_streaming_error_response(str(e), type="internal_server_error", code=code)
            yield f"data: {data}\n\n"

        raw_request.state.finish_reason = finish_reason_list
        # Send the final done message after all response.n are finished
        yield "data: [DONE]\n\n"

    async def chat_completion_full_generator(
        self,
        request: ChatCompletionRequest,
        raw_request: Request,
        result_generator: AsyncIterator[PipelineOutput],
        request_id: str,
    ) -> ChatCompletionResponse:
        model_name = self.model_name
        created_time = int(time.monotonic())
        final_res: PipelineOutput = None

        async for res in result_generator:
            if await raw_request.is_disconnected():
                # Abort the request if the client disconnects.
                await self.pipeline.abort_stream_request(request_id)
                raise asyncio.CancelledError("Client disconnected")
            final_res = res
        assert final_res is not None

        choices = []
        role = "assistant"
        raw_request.state.first_scheduled_time = final_res.first_scheduled_time
        raw_request.state.first_token_time = final_res.first_token_time
        raw_request.state.generated_text = []
        for i, output in enumerate(final_res.outputs):
            choice_data = ChatCompletionResponseChoice(
                index=output.index,
                message=ChatMessage(role=role, content=output.text),
                finish_reason=output.finish_reason,
            )
            choices.append(choice_data)
            raw_request.state.generated_text.append(output.text)

        if request.echo:
            last_msg_content = ""
            if (
                request.messages
                and isinstance(request.messages, list)
                and request.messages[-1].get("content")
                and request.messages[-1].get("role") == role
            ):
                last_msg_content = request.messages[-1]["content"]

            for choice in choices:
                full_message = last_msg_content + choice.message.content
                choice.message.content = full_message

        num_prompt_tokens = len(final_res.prompt_token_ids)
        num_generated_tokens = sum(len(output.token_ids) for output in final_res.outputs)
        usage = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_generated_tokens,
            total_tokens=num_prompt_tokens + num_generated_tokens,
        )
        response = ChatCompletionResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            choices=choices,
            usage=usage,
        )

        return response

    async def compute_chat_tokens(self, request: ChatCompletionRequest):
        input_ids = self.pipeline.pre_process(request.messages, add_generation_prompt=request.add_generation_prompt)
        return {"token_nums": len(input_ids)}
