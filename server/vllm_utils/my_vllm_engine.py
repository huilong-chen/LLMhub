import atexit
import logging
import os
import signal
import threading
import time
from collections import defaultdict
from typing import Type, Optional

from vllm import AsyncLLMEngine
from vllm.core.scheduler import SchedulerOutputs
from vllm.engine.async_llm_engine import _AsyncLLMEngine
from vllm.engine.metrics import StatLogger, Stats
from typing import List
from vllm.sequence import SamplerOutput

from server.vllm_utils.vllm_scheduler_patch import patch_scheduler_add_seq_group


class MyLLMEngine(_AsyncLLMEngine):

    def _get_stats(
            self,
            scheduler_outputs: Optional[SchedulerOutputs],
            model_output: Optional[List[SamplerOutput]] = None) -> Stats:
        """Get Stats to be Logged to Prometheus."""
        now = time.monotonic()
        # KV Cache Usage in %
        num_total_gpu = self.cache_config.num_gpu_blocks
        gpu_cache_usage_sys = 0.
        if num_total_gpu is not None:
            num_free_gpu = self.scheduler.block_manager.get_num_free_gpu_blocks(
            )
            gpu_cache_usage_sys = 1.0 - (num_free_gpu / num_total_gpu)

        num_total_cpu = self.cache_config.num_cpu_blocks
        cpu_cache_usage_sys = 0.
        if num_total_cpu is not None and num_total_cpu > 0:
            num_free_cpu = self.scheduler.block_manager.get_num_free_cpu_blocks()
            cpu_cache_usage_sys = 1.0 - (num_free_cpu / num_total_cpu)

        stats_by_api_key = defaultdict(lambda: Stats(
            now=0,
            num_running_sys=0,
            num_waiting_sys=0,
            num_swapped_sys=0,
            gpu_cache_usage_sys=0,
            cpu_cache_usage_sys=0,
            num_prompt_tokens_iter=0,
            num_generation_tokens_iter=0,
            time_to_first_tokens_iter=[],
            time_per_output_tokens_iter=[],
            num_preemption_iter=0,
            time_e2e_requests=[],
            num_prompt_tokens_requests=[],
            num_generation_tokens_requests=[],
            best_of_requests=[],
            n_requests=[],
            finished_reason_requests=[],
        ))

        for seq_group in self.scheduler.running:
            api_key = getattr(seq_group.sampling_params, "api_key", "unknown")
            stats_by_api_key[api_key].num_running_sys += 1

        for seq_group in self.scheduler.swapped:
            api_key = getattr(seq_group.sampling_params, "api_key", "unknown")
            stats_by_api_key[api_key].num_swapped_sys += 1

        for seq_group in self.scheduler.waiting:
            api_key = getattr(seq_group.sampling_params, "api_key", "unknown")
            stats_by_api_key[api_key].num_swapped_sys += 1

        if scheduler_outputs is not None:
            for idx, scheduled_seq_group in enumerate(scheduler_outputs.scheduled_seq_groups):
                api_key = getattr(scheduled_seq_group.seq_group.sampling_params, "api_key", "unknown")
                group_was_prefill = idx < scheduler_outputs.num_prefill_groups
                seq_group = scheduled_seq_group.seq_group
                if group_was_prefill:
                    # Number of prompt tokens.
                    stats_by_api_key[api_key].num_prompt_tokens_iter += scheduled_seq_group.token_chunk_size

                    # If the seq_group just finished the prefill state
                    # get TTFT.
                    if not seq_group.is_prefill():
                        latency = seq_group.get_last_latency(now)
                        stats_by_api_key[api_key].time_to_first_tokens_iter.append(latency)
                else:
                    # TPOTs.
                    latency = seq_group.get_last_latency(now)
                    stats_by_api_key[api_key].time_per_output_tokens_iter.append(latency)

                stats_by_api_key[api_key].num_generation_tokens_iter += seq_group.num_seqs()

        stat = Stats(
            now=now,
            num_running_sys=0,
            num_waiting_sys=0,
            num_swapped_sys=0,
            gpu_cache_usage_sys=gpu_cache_usage_sys,
            cpu_cache_usage_sys=cpu_cache_usage_sys,
            num_prompt_tokens_iter=0,
            num_generation_tokens_iter=0,
            time_to_first_tokens_iter=[],
            time_per_output_tokens_iter=[],
            num_preemption_iter=0,
            time_e2e_requests=[],
            num_prompt_tokens_requests=[],
            num_generation_tokens_requests=[],
            best_of_requests=[],
            n_requests=[],
            finished_reason_requests=[],
        )
        stat.stats_by_api_key = stats_by_api_key
        return stat

class MyAsyncLLMEngine(AsyncLLMEngine):
    _engine_class: Type[_AsyncLLMEngine] = MyLLMEngine

    @classmethod
    def from_engine_args(cls, *args, **kwargs) -> "AsyncLLMEngine":

        if os.environ.get("PATCH_ADD_SEQ_GROUP", "1"):
            patch_scheduler_add_seq_group()

        async_engine = super().from_engine_args(*args, **kwargs)

        return async_engine
