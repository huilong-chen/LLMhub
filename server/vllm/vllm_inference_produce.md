# 基于 VLLM 梳理推理流程

Repo：https://github.com/vllm-project/vllm 

【2024.07.04】本次记录只讨论单卡的情况

## vllm 的简单使用
从下面的代码入手
```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# 创建一个采样对象
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# 创建一个LLM
llm = LLM(model="facebook/Meta-llama-3-8B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("facebook/Meta-llama-3-8B-Instruct")
input_ids = [tokenizer.encode(prompt, add_special_tokens=True) for prompt in prompts]

# 方法一：根据 token id 生成文本，返回一个 RequestOutput 对象
outputs = llm.generate(prompts=None, sampling_params=sampling_params, prompt_token_ids=input_ids)

# 方法二：根据 prompt 文本生成，同样返回一个 RequestOutput 对象
outputs = llm.generate(prompts=prompts, sampling_params=sampling_params, prompt_token_ids=None)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

从上面的代码可以看到这里创建了一个LLM对象，然后调用了LLM对象的 `generate` 函数。这就是vllm的入口点，接下来我们对LLM这个类的 `generaet` 过程进行解析. LLM这个类可以在 [llm.py](https://github.com/vllm-project/vllm/blob/56b325e977435af744f8b3dca7af0ca209663558/vllm/entrypoints/llm.py#L24) 中找到.

```python
class LLM:
    """这是一个名为LLM（语言模型）的Python类，这个类用于从给定的提示和采样参数生成文本。
    类的主要部分包括tokenizer（用于将输入文本分词）、语言模型（可能分布在多个GPU上执行）
    以及为中间状态分配的GPU内存空间（也被称为KV缓存）。给定一批提示和采样参数，
    该类将使用智能批处理机制和高效的内存管理从模型中生成文本。

    这个类设计用于离线推理。在线服务的话，应使用AsyncLLMEngine类。
    对于参数列表，可以参见EngineArgs。

    Args:
        model: HuggingFace Transformers模型的名称或路径。
        tokenizer: HuggingFace Transformers标记器的名称或路径。
        tokenizer_mode:标记器模式。"auto"将使用快速标记器
            如果可用，则“slow”将始终使用慢标记器。
        skip_tokenizer_init:如果为true，跳过tokenizer和detokenizer的初始化。
            对于prompt，期望有效的prompt_token_ids和None从输入。
        trust_remote_code: 信任远程代码(例如，来自HuggingFace)，当下载模型和标记器。
        tensor_parallel_size:用于分布式的gpu数量张量并行执行。
        dtype:模型权重和激活的数据类型。目前支持' float32 '， ' float16 '和' bfloat16 '。
            如果是auto，我们用在模型配置文件中指定的' torch_dtype '属性。
            但是，如果配置中的' torch_dtype '是' float32 '，我们将使用' float16 '代替。
        quantization: 将模型权重量化的方法。目前, 我们支持“awq”，“gptq”，“squeezellm”和“fp8”(实验)。
            如果None，我们首先检查模型配置文件。如果该值为None，我们假设模型权重为不量化，使用' dtype '来确定的数据类型权重。
        revision:要使用的具体模型版本。可以是分支名称，标记名或提交id。
        tokenizer_revision:要使用的特定标记器版本。它可以是分支名称、标记名称或提交id。
        seed:初始化用于采样的随机数生成器的种子。
        gpu_memory_utilization: GPU内存占比，取值范围为0 ~ 1
            为模型权重、激活和KV缓存预留。更高的值将增加KV缓存大小，从而改进模型吞吐量。
            但是，如果该值过高，则可能导致OOM错误。
        swap_space:每个GPU用作交换空间的CPU内存大小(GiB)。
            这可以用于临时存储请求的状态当它们的best_of抽样参数大于1时。如果所有
            请求将具有' best_of=1 '，您可以安全地将其设置为0。否则，太小的值可能会导致内存不足(OOM)错误。
            强制执行:是否强制执行。如果是真的，我们会禁用CUDA图形，并始终在等待模式下执行模型。
            如果为False，我们将混合使用CUDA图形和渴望执行。
        max_context_len_to_capture: CUDA图形覆盖的最大上下文len。
            当一个序列的上下文长度大于这个长度时，我们就退一步
            切换到渴望模式(已弃用)。使用' max_seq_len_to_capture '代替)。
        max_seq_len_to_capture: CUDA图形覆盖的最大序列len。
            当一个序列的上下文长度大于这个长度时，我们就退一步
            切换到渴望模式。
        disable_custom_all_reduce:参见ParallelConfig
        **kwargs: class: ' ~vllm. engineeargs '的参数。
    """

    def __init__(
        self,
        model: str,
        tokenizer: Optional[str] = None,
        tokenizer_mode: str = "auto",
        skip_tokenizer_init: bool = False,
        trust_remote_code: bool = False,
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        quantization: Optional[str] = None,
        revision: Optional[str] = None,
        tokenizer_revision: Optional[str] = None,
        seed: int = 0,
        gpu_memory_utilization: float = 0.9,
        swap_space: int = 4,
        enforce_eager: bool = False,
        max_context_len_to_capture: Optional[int] = None,
        max_seq_len_to_capture: int = 8192,
        disable_custom_all_reduce: bool = False,
        **kwargs,
    ) -> None:
        # 在初始化函数中，首先检查kwargs中是否包含"disable_log_stats"键，
        # 如果没有，则在kwargs中添加该键并设置其值为True。
        if "disable_log_stats" not in kwargs:
            kwargs["disable_log_stats"] = True
        # 使用所有给定的参数（包括通过kwargs传递的任何额外参数）来初始化EngineArgs对象，
        # 然后使用这些参数来初始化LLMEngine对象
        engine_args = EngineArgs(
            model=model,
            tokenizer=tokenizer,
            tokenizer_mode=tokenizer_mode,
            skip_tokenizer_init=skip_tokenizer_init,
            trust_remote_code=trust_remote_code,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            quantization=quantization,
            revision=revision,
            tokenizer_revision=tokenizer_revision,
            seed=seed,
            gpu_memory_utilization=gpu_memory_utilization,
            swap_space=swap_space,
            enforce_eager=enforce_eager,
            max_context_len_to_capture=max_context_len_to_capture,
            max_seq_len_to_capture=max_seq_len_to_capture,
            disable_custom_all_reduce=disable_custom_all_reduce,
            **kwargs,
        )
        self.llm_engine = LLMEngine.from_engine_args(engine_args, usage_context=UsageContext.LLM_CLASS)
        # 初始化一个名为request_counter的Counter对象，用于请求计数。
        self.request_counter = Counter()
```
可以看到LLM类似于对`LLMEngine`进行了封装，一个LLM对象对应了一个`LLMEngine`对象。下面解析一下`EngineArgs`和`LLMEngine`。
首先来看`EngineArgs`，这个类可以在 [arg_utils.py](https://github.com/vllm-project/vllm/blob/56b325e977435af744f8b3dca7af0ca209663558/vllm/engine/arg_utils.py) 中找到：

TL;DR

下面解析`LLMEngine`，这个类可以在 [llm_engine.py](https://github.com/vllm-project/vllm/blob/56b325e977435af744f8b3dca7af0ca209663558/vllm/engine/llm_engine.py) 中找到：

```python
class LLMEngine:
    """这段代码定义了一个名为 LLMEngine 的类，它是一个接收请求并生成文本的语言模型(LLM)引擎。

	  这个类是vLLM引擎的主要类，它从客户端接收请求，并从LLM生成文本。
	  这个类包含了一个分词器，一个语言模型（可能在多个GPU之间切分），
	  以及为中间状态（也称为KV缓存）分配的GPU内存空间。
	  此类使用了迭代级别的调度和有效的内存管理来最大化服务吞吐量。

    LLM 类将此类封装用于离线批量推理，而 AsyncLLMEngine 类将此类封装用于在线服务

    注意：配置参数源自 EngineArgs 类。有关参数的完整列表，请参见 EngineArgs。

    Args:
        model_config: 与LLM模型相关的配置。
        cache_config: 与KV缓存内存管理相关的配置。
        parallel_config: 与分布式执行相关的配置。
        scheduler_config: 与分布式执行相关的配置。
        device_config: 与分布式执行设备相关的配置。
        lora_config (Optional): 服务于multi-LoRA的相关配置。
        multimodal_config (Optional): The configuration related to multimodal 
            models.
        speculative_config (Optional): The configuration related to speculative
            decoding.
        executor_class: The model executor class for managing distributed
            execution.
        log_stats: Whether to log statistics.
        usage_context: Specified entry point, used for usage info collection.
    """
    def __init__(
        self,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        load_config: LoadConfig,
        lora_config: Optional[LoRAConfig],
        multimodal_config: Optional[MultiModalConfig],
        speculative_config: Optional[SpeculativeConfig],
        decoding_config: Optional[DecodingConfig],
        observability_config: Optional[ObservabilityConfig],
        executor_class: Type[ExecutorBase],
        log_stats: bool,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: Optional[Dict[str, StatLoggerBase]] = None,
    ) -> None:

        # TODO(woosuk): Print more configs in debug mode.

        self.model_config = model_config
        self.cache_config = cache_config
        self.lora_config = lora_config
        self.multimodal_config = multimodal_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.device_config = device_config
        self.speculative_config = speculative_config
        self.load_config = load_config
        self.decoding_config = decoding_config or DecodingConfig()
        self.observability_config = observability_config or ObservabilityConfig(
        )
        self.log_stats = log_stats

        # 设置tokenizer
        if not self.model_config.skip_tokenizer_init:
            self.tokenizer = self._init_tokenizer()
            self.detokenizer = Detokenizer(self.tokenizer)
        else:
            self.tokenizer = None
            self.detokenizer = None

        self.seq_counter = Counter()
        self.generation_config_fields = _load_generation_config_dict(
            model_config)

        self.input_processor = INPUT_REGISTRY.create_input_processor(
            self.model_config)

        self.model_executor = executor_class(
            model_config=model_config,
            cache_config=cache_config,
            parallel_config=parallel_config,
            scheduler_config=scheduler_config,
            device_config=device_config,
            lora_config=lora_config,
            multimodal_config=multimodal_config,
            speculative_config=speculative_config,
            load_config=load_config,
        )

        if not self.model_config.embedding_mode:
            self._initialize_kv_caches()

        # If usage stat is enabled, collect relevant info.
        if is_usage_stats_enabled():
            from vllm.model_executor.model_loader import (
                get_architecture_class_name)
            usage_message.report_usage(
                get_architecture_class_name(model_config),
                usage_context,
                extra_kvs={
                    # Common configuration
                    "dtype":
                    str(model_config.dtype),
                    "tensor_parallel_size":
                    parallel_config.tensor_parallel_size,
                    "block_size":
                    cache_config.block_size,
                    "gpu_memory_utilization":
                    cache_config.gpu_memory_utilization,

                    # Quantization
                    "quantization":
                    model_config.quantization,
                    "kv_cache_dtype":
                    cache_config.cache_dtype,

                    # Feature flags
                    "enable_lora":
                    bool(lora_config),
                    "enable_prefix_caching":
                    cache_config.enable_prefix_caching,
                    "enforce_eager":
                    model_config.enforce_eager,
                    "disable_custom_all_reduce":
                    parallel_config.disable_custom_all_reduce,
                })

        if self.tokenizer:
            # Ping the tokenizer to ensure liveness if it runs in a
            # different process.
            self.tokenizer.ping()

        # Create the scheduler.
        # NOTE: the cache_config here have been updated with the numbers of
        # GPU and CPU blocks, which are profiled in the distributed executor.
        self.scheduler = [
            Scheduler(scheduler_config, cache_config, lora_config,
                      parallel_config.pipeline_parallel_size)
            for _ in range(parallel_config.pipeline_parallel_size)
        ]

        # Metric Logging.
        ...

        self.tracer = None
        if self.observability_config.otlp_traces_endpoint:
            self.tracer = init_tracer(
                "vllm.llm_engine",
                self.observability_config.otlp_traces_endpoint)

        # Create sequence output processor, e.g. for beam search or
        # speculative decoding.
        self.output_processor = (
            SequenceGroupOutputProcessor.create_output_processor(
                self.scheduler_config,
                self.detokenizer,
                self.scheduler,
                self.seq_counter,
                self.get_tokenizer_for_seq,
                stop_checker=StopChecker(
                    self.scheduler_config.max_model_len,
                    self.get_tokenizer_for_seq,
                ),
            ))
```
从`LLMEngine`的定义可以知道，它做了初始化 tokenizer，创建并行的 worker 信息以及初始化 KV Cache 等事情，这里的 worker 是每个 GPU 对应一个。

下面是`LLMEngine`的其他函数：
```python
    # TODO

```

回到LLM类的`generate`过程解析：

```python

    def get_tokenizer(
            self) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
        return self.llm_engine.tokenizer.tokenizer

    def set_tokenizer(
        self,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    ) -> None:
        # While CachedTokenizer is dynamic, have no choice but
        # compare class name. Misjudgment will arise from
        # user-defined tokenizer started with 'Cached'
        if tokenizer.__class__.__name__.startswith("Cached"):
            self.llm_engine.tokenizer.tokenizer = tokenizer
        else:
            self.llm_engine.tokenizer.tokenizer = get_cached_tokenizer(
                tokenizer)

    @deprecate_kwargs("prompts",
                      "prompt_token_ids",
                      is_deprecated=lambda: LLM.DEPRECATE_LEGACY,
                      additional_message="Please use the 'inputs' parameter "
                      "instead.")
    
    """
        省略了一些generate的虚函数定义
    """
    
    def generate(
        self,
        prompts: Union[Union[PromptStrictInputs, Sequence[PromptStrictInputs]],
                       Optional[Union[str, List[str]]]] = None,
        sampling_params: Optional[Union[SamplingParams,
                                        Sequence[SamplingParams]]] = None,
        prompt_token_ids: Optional[Union[List[int], List[List[int]]]] = None,
        use_tqdm: bool = True,
        lora_request: Optional[Union[List[LoRARequest], LoRARequest]] = None,
    ) -> List[RequestOutput]:
        """Generates the completions for the input prompts.

        This class automatically batches the given prompts, considering
        the memory constraint. For the best performance, put all of your prompts
        into a single list and pass it to this method.

        Args:
            inputs: A list of inputs to generate completions for.
            sampling_params: The sampling parameters for text generation. If
                None, we use the default sampling parameters. 
                When it is a single value, it is applied to every prompt. 
                When it is a list, the list must have the same length as the 
                prompts and it is paired one by one with the prompt.
            use_tqdm: Whether to use tqdm to display the progress bar.
            lora_request: LoRA request to use for generation, if any.

        Returns:
            A list of `RequestOutput` objects containing the
            generated completions in the same order as the input prompts.

        Note:
            Using ``prompts`` and ``prompt_token_ids`` as keyword parameters is
            considered legacy and may be deprecated in the future. You should
            instead pass them via the ``inputs`` parameter.
        """
        if self.llm_engine.model_config.embedding_mode:
            raise ValueError(
                "LLM.generate() is only supported for generation models "
                "(XForCausalLM).")

        if prompt_token_ids is not None:
            inputs = self._convert_v1_inputs(
                prompts=cast(Optional[Union[str, List[str]]], prompts),
                prompt_token_ids=prompt_token_ids,
            )
        else:
            inputs = cast(
                Union[PromptStrictInputs, Sequence[PromptStrictInputs]],
                prompts)

        if sampling_params is None:
            # Use default sampling params.
            sampling_params = SamplingParams()

        self._validate_and_add_requests(
            inputs=inputs,
            params=sampling_params,
            lora_request=lora_request,
        )

        outputs = self._run_engine(use_tqdm=use_tqdm)
        return LLMEngine.validate_outputs(outputs, RequestOutput)
    
    """
        省略了一些encode虚函数的定义
    """

    @deprecate_kwargs("prompts",
                      "prompt_token_ids",
                      is_deprecated=lambda: LLM.DEPRECATE_LEGACY,
                      additional_message="Please use the 'inputs' parameter "
                      "instead.")
    def encode(
        self,
        prompts: Union[Union[PromptStrictInputs, Sequence[PromptStrictInputs]],
                       Optional[Union[str, List[str]]]] = None,
        pooling_params: Optional[Union[PoolingParams,
                                       Sequence[PoolingParams]]] = None,
        prompt_token_ids: Optional[Union[List[int], List[List[int]]]] = None,
        use_tqdm: bool = True,
        lora_request: Optional[Union[List[LoRARequest], LoRARequest]] = None,
    ) -> List[EmbeddingRequestOutput]:
        """Generates the completions for the input prompts.

        This class automatically batches the given prompts, considering
        the memory constraint. For the best performance, put all of your prompts
        into a single list and pass it to this method.

        Args:
            inputs: The inputs to the LLM. You may pass a sequence of inputs for
                batch inference. See :class:`~vllm.inputs.PromptStrictInputs`
                for more details about the format of each input.
            pooling_params: The pooling parameters for pooling. If None, we
                use the default pooling parameters.
            use_tqdm: Whether to use tqdm to display the progress bar.
            lora_request: LoRA request to use for generation, if any.

        Returns:
            A list of `EmbeddingRequestOutput` objects containing the
            generated embeddings in the same order as the input prompts.

        Note:
            Using ``prompts`` and ``prompt_token_ids`` as keyword parameters is
            considered legacy and may be deprecated in the future. You should
            instead pass them via the ``inputs`` parameter.
        """
        if not self.llm_engine.model_config.embedding_mode:
            raise ValueError(
                "LLM.encode() is only supported for embedding models (XModel)."
            )

        if prompt_token_ids is not None:
            inputs = self._convert_v1_inputs(
                prompts=cast(Optional[Union[str, List[str]]], prompts),
                prompt_token_ids=prompt_token_ids,
            )
        else:
            inputs = cast(
                Union[PromptStrictInputs, Sequence[PromptStrictInputs]],
                prompts)

        if pooling_params is None:
            # Use default pooling params.
            pooling_params = PoolingParams()

        self._validate_and_add_requests(
            inputs=inputs,
            params=pooling_params,
            lora_request=lora_request,
        )

        outputs = self._run_engine(use_tqdm=use_tqdm)
        return LLMEngine.validate_outputs(outputs, EmbeddingRequestOutput)


    def _validate_and_add_requests(
        self,
        inputs: Union[PromptStrictInputs, Sequence[PromptStrictInputs]],
        params: Union[SamplingParams, Sequence[SamplingParams], PoolingParams,
                      Sequence[PoolingParams]],
        lora_request: Optional[Union[Sequence[LoRARequest], LoRARequest]],
    ) -> None:
        if isinstance(inputs, (str, dict)):
            # Convert a single prompt to a list.
            inputs = [inputs]

        num_requests = len(inputs)

        if isinstance(params, list) and len(params) != num_requests:
            raise ValueError("The lengths of prompts and params "
                             "must be the same.")
        if isinstance(lora_request,
                      list) and len(lora_request) != num_requests:
            raise ValueError("The lengths of prompts and lora_request "
                             "must be the same.")

        # Add requests to the engine.
        for i, request_inputs in enumerate(inputs):
            self._add_request(
                request_inputs,
                params[i] if isinstance(params, Sequence) else params,
                lora_request=lora_request[i] if isinstance(
                    lora_request, Sequence) else lora_request,
            )

    def _add_request(
        self,
        inputs: PromptInputs,
        params: Union[SamplingParams, PoolingParams],
        lora_request: Optional[Union[List[LoRARequest], LoRARequest]] = None,
    ) -> None:
        request_id = str(next(self.request_counter))
        self.llm_engine.add_request(request_id,
                                    inputs,
                                    params,
                                    lora_request=lora_request)

    def _run_engine(
            self, *, use_tqdm: bool
    ) -> List[Union[RequestOutput, EmbeddingRequestOutput]]:
        # Initialize tqdm.
        if use_tqdm:
            num_requests = self.llm_engine.get_num_unfinished_requests()
            pbar = tqdm(
                total=num_requests,
                desc="Processed prompts",
                dynamic_ncols=True,
                postfix=(f"est. speed input: {0:.2f} toks/s, "
                         f"output: {0:.2f} toks/s"),
            )
        # Run the engine.
        outputs: List[Union[RequestOutput, EmbeddingRequestOutput]] = []
        total_in_toks = 0
        total_out_toks = 0
        while self.llm_engine.has_unfinished_requests():
            step_outputs = self.llm_engine.step()
            for output in step_outputs:
                if output.finished:
                    outputs.append(output)
                    if use_tqdm:
                        if isinstance(output, RequestOutput):
                            # Calculate tokens only for RequestOutput
                            total_in_toks += len(output.prompt_token_ids)
                            in_spd = total_in_toks / pbar.format_dict["elapsed"]
                            total_out_toks += sum(
                                len(stp.token_ids) for stp in output.outputs)
                            out_spd = total_out_toks / pbar.format_dict[
                                "elapsed"]
                            pbar.postfix = (
                                f"est. speed input: {in_spd:.2f} toks/s, "
                                f"output: {out_spd:.2f} toks/s")
                        pbar.update(1)
        if use_tqdm:
            pbar.close()
        # Sort the outputs by request ID.
        # This is necessary because some requests may be finished earlier than
        # its previous requests.
        return sorted(outputs, key=lambda x: int(x.request_id))
```

现在基本走完了vllm根据prompt，特定模型架构和特定采样参数去生成结果的全流程，下面再对这个流程总结一下。

首先，vllm进来之后先实例化一个LLM对象即：`llm = LLM(model="xxx/xxx")`。然后调用`llm.generate`函数，这个函数的输入是prompts（List[str]类型），采样参数，然后返回 `List[RequestOutput]`，对应`outputs = llm.generate(prompts, sampling_params)`这行代码。从`llm.generate`的实现来看，对于每一个 prompt 都会生成一个`request`喂给`llm_engine`，然后执行`_run_engine`（这个函数负责运行 `llm_engine.step` 函数，并收集已完成的请求的输出，函数结束。

~~llm_engine.step函数首先从scheduler获取当前的输入seq_group_metadata_list ，同时生成一个 scheduler_outputs，接下来会调用 workers 的 execute_model来指定模型的前向推理过程，拿到这个结果之后再进行解码（对应self._decode_sequences(seq_groups)这行）。最后scheduler再更新已经解码完毕的序列的状态，并释放序列占用的内存。~~