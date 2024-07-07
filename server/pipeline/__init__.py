from abc import ABC
from dataclasses import InitVar, dataclass, field
from typing import AsyncIterator, Callable, Dict, List, Optional, Set, Union
from typing_extensions import Annotated
from pydantic import Field

import torch
from transformers import AutoTokenizer

LogitsProcessor = Callable[[List[int], torch.Tensor], torch.Tensor]
Conversation = List[Dict[str, str]]


@dataclass
class PipelineParams:
    """
    Pipeline parameters
    Copied from vllm SamplingParams

    vllm:
        Overall, we follow the sampling parameters from the OpenAI text completion
    API (https://platform.openai.com/docs/api-reference/completions/create).
    In addition, we support beam search, which is not supported by OpenAI.

    """

    n: int = 1 # 要为给定提示返回的输出序列数
    best_of: Optional[int] = None # 从提示生成的输出序列数。从这些best_of序列中，返回顶部n个序列。best_of必须大于或等于n。当use_beam_search为True时，这被视为束宽。默认情况下，best_of设置为n。
    presence_penalty: float = 0.0 # 根据新token在生成的文本中的出现频率对新token进行惩罚的浮点数。大于0的值鼓励模型使用新token，小于0的值鼓励模型重复token。
    frequency_penalty: float = 0.0 # 根据新token在迄今为止生成的文本中的频率对新token进行惩罚的浮点数。大于0的值鼓励模型使用新token，小于0的值鼓励模型重复token。
    repetition_penalty: float = 1.0 # 根据新token是否出现在提示和迄今为止生成的文本中对新token进行惩罚的浮点数。大于1的值鼓励模型使用新token，小于1的值鼓励模型重复token。
    temperature: float = 1.0 # 控制抽样随机性的浮点数。较低的值让模型更确定性，较高的值让模型更随机。零意味着贪婪抽样。
    top_p: float = 1.0 # 控制要考虑的顶部token的累积概率的浮点数。必须在(0, 1]之间。设为1考虑所有token。
    top_k: int = -1 # 控制要考虑的顶部token数的整数。设为-1考虑所有token。
    min_p: float = 0.0 # 代表要考虑的token的最小概率（相对于最有可能的token的概率）。必须在[0, 1]之间。设为0禁用此功能。
    use_beam_search: bool = False # 是否使用束搜索而不是抽样。
    length_penalty: float = 1.0 # 根据序列的长度对序列进行惩罚的浮点数。在束搜索中使用。
    early_stopping: Union[bool, str] = False # 控制束搜索的停止条件。接受以下值：“True”，生成在有best_of个完成候选时停止；“False”，应用启发式并在找到更好候选者变得非常不可能时停止；"never"，束搜索程序仅在不能有更好候选者时停止（规范束搜索算法）。
    stop: Optional[Union[str, List[str]]] = None # 生成停止时生成的字符串列表。返回的输出不会包含停止字符串。
    stop_token_ids: Optional[List[int]] = None # 生成停止时生成的token列表。返回的输出将包含停止token，除非停止token是特殊token。
    include_stop_str_in_output: bool = False # 是否将停止字符串包含在输出文本中。默认为False。
    ignore_eos: bool = False # 是否忽略EOS token并在生成EOS token后继续生成token。
    max_tokens: int = 16 # 每个输出序列生成的最大token数。
    logprobs: Optional[int] = None # 返回每个输出token的对数概率数。注意，实现遵循OpenAI API：返回结果包括最可能token的对数概率，以及所选token的对数概率。API总是返回采样token的对数概率，因此响应中可能有多达logprobs+1个元素。
    prompt_logprobs: Optional[int] = None # 返回每个提示token的对数概率数
    skip_special_tokens: bool = True # 是否在输出中跳过特殊token。
    spaces_between_special_tokens: bool = True # 输出中是否在特殊token之间添加空格。默认为True。
    logits_processors: List[LogitsProcessor] = field(default_factory=list) # 基于之前生成的token（以及作为第一个参数的提示token，如果有的话）修改对数几率的函数列表。
    truncate_prompt_tokens: Optional[Annotated[int, Field(ge=1)]] = None, # 如果设置为整数k，将仅使用提示的最后k个token（即，左截断）。默认为None（即，不截断）。

    # 新增
    api_key: Optional[str] = None
    do_sample: InitVar[bool] = True
    add_default_stop: InitVar[bool] = False

    def __post_init__(self, do_sample: bool, add_default_stop: bool):
        if add_default_stop and not self.ignore_eos:
            self._add_default_stop()
        if not do_sample:
            self.temperature = 0
            self.top_p = 1
            self.top_k = -1

    def _add_default_stop(self):
        # 强制加上 stop 标记, 默认添加 <|im_end|> 和 <|eot_id|> 作为终止符（后者适配 Llama 3）
        default_stop_tokens = ["<|im_end|>", "<|eot_id|>"]
        if not self.stop:
            self.stop = default_stop_tokens
        else:
            if isinstance(self.stop, str):
                self.stop = [self.stop]
            self.stop.extend(default_stop_tokens)

PromptLogprobs = List[Optional[Dict[int, float]]]
SampleLogprobs = List[Dict[int, float]]

@dataclass
class PipelineCompletionOutput:
    """
    Completion output
    Copied from vllm CompletionOutput
    """

    index: int # 请求中输出的索引。
    text: str # 生成的输出文本。
    token_ids: List[int] # 生成输出文本的token ID。
    cumulative_logprob: float # 生成输出文本的累计对数概率。
    logprobs: Optional[SampleLogprobs] # 如果请求了对数概率，每个位置上概率最高的词的对数概率。
    finish_reason: Optional[str] = (None,) # 序列完成的原因。
    stop_reason: Union[int, str, None] = None # 导致完成停止的停止字符串或token id，如果完成因为遇到EOS token或其他原因而完成，则为None。

@dataclass
class PipelineOutput:
    """
    Pipeline output
    Copied from vllm RequestOutput
    """

    request_id: str # 请求的唯一ID。
    prompt: str # 请求的提示字符串。
    prompt_token_ids: List[int] # 提示的token ID。
    prompt_logprobs: Optional[PromptLogprobs] # 返回每个提示token的对数概率。
    outputs: List[PipelineCompletionOutput] # 请求的输出序列。
    finished: bool # 整个请求是否已完成。

    # 新增
    first_scheduled_time: Optional[float]
    first_token_time: Optional[float]


class Pipeline(ABC):
    def __init__(
            self,
            model_path: str,
            tensor_parallel_size=1,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
        self._init_engine(model_path, tensor_parallel_size=tensor_parallel_size)
        self.max_model_len = self._get_max_model_len()

    @staticmethod
    def create_pipeline(*args, **kwargs):
        from server.pipeline.vllm import VLLMPipeline
        return VLLMPipeline(*args, **kwargs)

    def _init_engine(self, model_path: str, tensor_parallel_size: int) -> None:
        raise NotImplementedError

    def _get_max_model_len(self) -> int:
        raise NotImplementedError

    def pre_process(
            self,
            conversation: Conversation,
            return_token_ids: bool = True,
            add_generation_prompt: bool = True,
            force_prompt: str = None,
    ) -> List[int]:
        """
        将 conversation 处理为 input_ids
        会自动在最后添加 assistant prompt
        conversation 包含若干个 dict，每个 dict 包含 role 和 content
        role 可以是 system, user, assistant
        content 是对话内容

        force_prompt: assistant 的回复强制用 force_prompt 的字符串开头
        """
        role_set = {"system", "user", "assistant"}
        for conv in conversation:
            if conv["role"] not in role_set:
                raise ValueError(f"role {conv['role']} not in {role_set}")
        result = self.tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=add_generation_prompt
        )
        if force_prompt:
            result += force_prompt
        if not return_token_ids:
            return result
        return self.tokenizer.encode(result, add_special_tokens=True)

    def __call__(
            self,
            conversations: Optional[List[Conversation]] = None,
            texts: Optional[List[str]] = None,
            use_tqdm: bool = False,
            **kwargs,
    ) -> List[str]:
        if (conversations is None) == (texts is None):
            raise ValueError("You should only input the conversations or texts, not both or neither.")
        if conversations is not None:
            batch_input_ids = [self.pre_process(conv, return_token_ids=True) for conv in conversations]
        if texts is not None:
            batch_input_ids = self.tokenizer.batch_encode_plus(texts, add_special_tokens=True)["input_ids"]

        outputs = self.generate(
            batch_input_ids,
            self.create_pipeline_params(**kwargs),
            use_tqdm,
        )
        return [output.outputs[0].text for output in outputs]

    def stream(
            self,
            input_ids: List[int],
            sampling_params: PipelineParams,
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
        raise NotImplementedError

    def generate(
            self,
            batch_input_ids: List[List[int]],
            pipeline_params: Union[PipelineParams, List[PipelineParams]],
            use_tqdm: bool = False,
    ) -> List[PipelineOutput]:
        """
        根据 input_ids 生成
        同步方法，当所有请求都完成时返回
        :param batch_input_ids: 输入的 batch input_ids
        :param pipeline_params: 采样参数
        :param use_tqdm: 是否使用 tqdm 进度条
        :return:
        """
        raise NotImplementedError

    async def abort_stream_request(self, request_id: str) -> None:
        raise NotImplementedError

    async def do_log_stats(self):
        raise NotImplementedError

    def create_pipeline_params(self, **kwargs):
        params = PipelineParams(**kwargs)
        return params
