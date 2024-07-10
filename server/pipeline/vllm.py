import asyncio
import os
from typing import AsyncIterator, List, Union

from server.api.utils import random_uuid
from server.pipeline import Pipeline, PipelineCompletionOutput, PipelineOutput, PipelineParams
from tqdm import tqdm
from vllm import AsyncEngineArgs, RequestOutput, SamplingParams
from vllm.utils import Counter
from server.vllm_utils.my_vllm_engine import MyAsyncLLMEngine
from vllm.inputs import TokensPrompt


def _get_sampling_params(pipeline_params: PipelineParams) -> SamplingParams:
    sampling_params = SamplingParams(
        n=pipeline_params.n,
        best_of=pipeline_params.best_of,
        presence_penalty=pipeline_params.presence_penalty,
        frequency_penalty=pipeline_params.frequency_penalty,
        repetition_penalty=pipeline_params.repetition_penalty,
        temperature=pipeline_params.temperature,
        top_p=pipeline_params.top_p,
        top_k=pipeline_params.top_k,
        min_p=pipeline_params.min_p,
        use_beam_search=pipeline_params.use_beam_search,
        length_penalty=pipeline_params.length_penalty,
        early_stopping=pipeline_params.early_stopping,
        stop=pipeline_params.stop,
        stop_token_ids=pipeline_params.stop_token_ids,
        include_stop_str_in_output=pipeline_params.include_stop_str_in_output,
        ignore_eos=pipeline_params.ignore_eos,
        max_tokens=pipeline_params.max_tokens,
        logprobs=pipeline_params.logprobs,
        prompt_logprobs=pipeline_params.prompt_logprobs,
        skip_special_tokens=pipeline_params.skip_special_tokens,
        spaces_between_special_tokens=pipeline_params.spaces_between_special_tokens,
        logits_processors=pipeline_params.logits_processors,
    )
    if pipeline_params.api_key:
        sampling_params.api_key = pipeline_params.api_key
    return sampling_params


def _get_pipeline_output(request_output: RequestOutput) -> PipelineOutput:
    return PipelineOutput(
        request_id=request_output.request_id,
        prompt=request_output.prompt,
        prompt_token_ids=request_output.prompt_token_ids,
        prompt_logprobs=request_output.prompt_logprobs,
        outputs=[
            PipelineCompletionOutput(
                index=output.index,
                text=output.text,
                token_ids=output.token_ids,
                cumulative_logprob=output.cumulative_logprob,
                logprobs=output.logprobs,
                finish_reason=output.finish_reason,
            )
            for output in request_output.outputs
        ],
        finished=request_output.finished,
        first_scheduled_time=request_output.metrics.first_scheduled_time,
        first_token_time=request_output.metrics.first_token_time,
    )


class VLLMPipeline(Pipeline):
    def _init_engine(self, model_path: str, tensor_parallel_size: int) -> None:
        self._request_counter = Counter()
        self._async_engine = MyAsyncLLMEngine.from_engine_args(
            AsyncEngineArgs(
                model=model_path,
                tokenizer_mode="auto",
                trust_remote_code=True,
                disable_log_requests=True,
                dtype="half",
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=float(os.environ.get("VLLM_GPU_UTIL", "0.9")),
                enforce_eager=False,
                max_model_len=int(os.environ.get("MAX_MODEL_LEN", 4096)),
            ),
        )

    def _get_max_model_len(self) -> int:
        engine_model_config = asyncio.run(self._async_engine.get_model_config())
        return engine_model_config.max_model_len

    def generate(
        self,
        batch_input_ids: List[List[int]],
        pipeline_params: Union[PipelineParams, List[PipelineParams]],
        use_tqdm: bool = False,
    ) -> List[PipelineOutput]:
        """
        根据 batch_input_ids 批量生成
        同步方法，当所有请求都完成时返回
        :param batch_input_ids: 输入的 batch input_ids
        :param pipeline_params: 采样参数
        :param use_tqdm: 是否使用 tqdm 进度条
        :return: 生成的结果
        """
        # handle different params, can be removed when all use of pipeline_params is a list
        if isinstance(pipeline_params, List):
            batch_pipeline_params = pipeline_params
        else:
            batch_pipeline_params = [pipeline_params for _ in range(len(batch_input_ids))]
        assert len(batch_pipeline_params) == len(batch_input_ids)
        batch_sampling_params = [_get_sampling_params(param) for param in batch_pipeline_params]

        for input_ids, sampling_params in zip(batch_input_ids, batch_sampling_params):
            request_id = str(next(self._request_counter))
            self._async_engine.engine.add_request(
                request_id=request_id,
                prompt=None,
                sampling_params=sampling_params,
                prompt_token_ids=input_ids,
            )

        if use_tqdm:
            num_requests = self._async_engine.engine.get_num_unfinished_requests()
            pbar = tqdm(total=num_requests, desc="Processed prompts")
        # Run the engine.
        outputs = []
        while self._async_engine.engine.has_unfinished_requests():
            step_outputs = self._async_engine.engine.step()
            for output in step_outputs:
                if output.finished:
                    outputs.append(output)
                    if use_tqdm:
                        pbar.update(1)
        if use_tqdm:
            pbar.close()

        outputs = sorted(outputs, key=lambda x: int(x.request_id))
        outputs = [_get_pipeline_output(output) for output in outputs]
        return outputs

    async def stream(
        self,
        input_ids: List[int],
        pipeline_params: PipelineParams,
        request_id: str = None,
    ) -> AsyncIterator[PipelineOutput]:
        """
        根据 input_ids 生成结果
        异步方法，使用异步生成器
        :param input_ids:
        :param pipeline_params:
        :param request_id:
        :return:
        """
        sampling_params = _get_sampling_params(pipeline_params)
        if not request_id:
            request_id = random_uuid()
        tokens_prompt = TokensPrompt(prompt_token_ids=input_ids)
        result_generator = self._async_engine.generate(
            inputs=tokens_prompt,
            sampling_params=sampling_params,
            request_id=request_id,
        )
        async for vllm_output in result_generator:
            yield _get_pipeline_output(vllm_output)

    async def abort_stream_request(self, request_id: str) -> None:
        return await self._async_engine.abort(request_id)

    async def do_log_stats(self):
        await self._async_engine.do_log_stats()
