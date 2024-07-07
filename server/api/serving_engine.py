import json
import logging
from typing import Dict, List, Optional, Union

from server.api.protocol import (
    ChatCompletionRequest,
    CompletionRequest,
    LogProbs,
    ModelCard,
    ModelList,
    ModelPermission,
    ErrorResponse,
)
from server.pipeline import Pipeline

class OpenAIServing:
    def __init__(self, pipeline: Pipeline, model_name: str):
        self.pipeline = pipeline
        self.model_name = model_name
        self.max_model_len = pipeline.max_model_len
        self.tokenizer = pipeline.tokenizer

    async def show_available_models(self) -> ModelList:
        """Show available models. Right now we only have one model."""
        model_cards = [ModelCard(id=self.model_name, root=self.model_name, permission=[ModelPermission()])]
        return ModelList(data=model_cards)

    def _create_logprobs(
        self,
        token_ids: List[int],
        top_logprobs: Optional[List[Optional[Dict[int, float]]]] = None,
        num_output_top_logprobs: Optional[int] = None,
        initial_text_offset: int = 0,
    ) -> LogProbs:
        """Create OpenAI-style logprobs."""
        logprobs = LogProbs()
        last_token_len = 0
        if num_output_top_logprobs:
            logprobs.top_logprobs = []
        for i, token_id in enumerate(token_ids):
            step_top_logprobs = top_logprobs[i]
            if step_top_logprobs is not None:
                token_logprob = step_top_logprobs[token_id]
            else:
                token_logprob = None
            token = self.tokenizer.convert_ids_to_tokens(token_id)
            logprobs.tokens.append(token)
            logprobs.token_logprobs.append(token_logprob)
            if len(logprobs.text_offset) == 0:
                logprobs.text_offset.append(initial_text_offset)
            else:
                logprobs.text_offset.append(logprobs.text_offset[-1] + last_token_len)
            last_token_len = len(token)

            if num_output_top_logprobs:
                logprobs.top_logprobs.append(
                    {self.tokenizer.convert_ids_to_tokens(i): p for i, p in step_top_logprobs.items()}
                    if step_top_logprobs
                    else None
                )
        return logprobs

    def _validate_prompt_and_tokenize(
        self,
        request: Union[ChatCompletionRequest, CompletionRequest],
        prompt: Optional[str] = None,
        prompt_ids: Optional[List[int]] = None,
    ) -> List[int]:
        if not (prompt or prompt_ids):
            raise ValueError("Either prompt or prompt_ids should be provided.")
        if prompt and prompt_ids:
            raise ValueError("Only one of prompt or prompt_ids should be provided.")

        input_ids = prompt_ids if prompt_ids is not None else self.tokenizer(prompt).input_ids
        token_num = len(input_ids)

        if token_num > self.max_model_len:
            raise ValueError(
                f"This model's maximum context length is {self.max_model_len} tokens. "
                f"However, you requested {token_num} tokens "
                f"Please reduce the length of the messages.",
            )
        elif token_num + request.max_tokens > self.max_model_len:
            logging.info(
                f"token_num: {token_num}, request.max_tokens: {request.max_tokens} (Before); {self.max_model_len - token_num} (Modified)"
            )
            request.max_tokens = self.max_model_len - token_num
        return input_ids

    def create_streaming_error_response(self, message: str, type: str, code: int) -> str:
        json_str = json.dumps({"error": ErrorResponse(message=message, type=type, code=code).model_dump()})
        return json_str
