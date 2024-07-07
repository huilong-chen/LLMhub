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
from vllm.engine.ray_utils import RayWorkerVllm

from server.vllm_utils.vllm_scheduler_patch import patch_scheduler_add_seq_group, patch_scheduler_with_tool


class MyLLMEngine(_AsyncLLMEngine):
    def _init_workers_ray(self, *args, **kwargs):
        RayWorkerVllm.ping = lambda *_, **__: True

        super()._init_workers_ray(*args, **kwargs)

        self.driver_worker.ping = lambda *_, **__: True

        # start worker monitor
        self.worker_monitor_stopped = threading.Event()
        self.worker_monitor = threading.Thread(target=self._worker_monitor, daemon=True)
        self.worker_monitor.start()

        def shutdown():
            self.worker_monitor_stopped.set()

        atexit.register(shutdown)

    def _worker_monitor(self):
        while not self.worker_monitor_stopped.wait(15):
            try:
                self._run_workers("ping")
            except:
                logging.exception("worker monitor error")
                os.kill(os.getpid(), signal.SIGKILL)
                break

    def _get_stats(self, scheduler_outputs: Optional[SchedulerOutputs]) -> Stats:
        """Get Stats to be Logged to Prometheus."""
        now = time.monotonic()

        # KV Cache Usage in %.
        num_total_gpu = self.cache_config.num_gpu_blocks
        num_free_gpu = self.scheduler.block_manager.get_num_free_gpu_blocks()
        gpu_cache_usage = 1.0 - (num_free_gpu / num_total_gpu)

        num_total_cpu = self.cache_config.num_cpu_blocks
        cpu_cache_usage = 0.0
        if num_total_cpu > 0:
            num_free_cpu = self.scheduler.block_manager.get_num_free_cpu_blocks()
            cpu_cache_usage = 1.0 - (num_free_cpu / num_total_cpu)

        stats_by_api_key = defaultdict(
            lambda: Stats(
                now=0,
                num_running=0,
                num_swapped=0,
                num_waiting=0,
                gpu_cache_usage=0,
                cpu_cache_usage=0,
                num_prompt_tokens=0,
                num_generation_tokens=0,
                time_to_first_tokens=[],
                time_per_output_tokens=[],
                time_e2e_requests=[],
            )
        )

        for seq_group in self.scheduler.running:
            api_key = getattr(seq_group.sampling_params, "api_key", "unknown")
            stats_by_api_key[api_key].num_running += 1

        for seq_group in self.scheduler.swapped:
            api_key = getattr(seq_group.sampling_params, "api_key", "unknown")
            stats_by_api_key[api_key].num_swapped += 1

        for seq_group in self.scheduler.waiting:
            api_key = getattr(seq_group.sampling_params, "api_key", "unknown")
            stats_by_api_key[api_key].num_waiting += 1

        if scheduler_outputs is not None:
            prompt_run = scheduler_outputs.prompt_run

            for seq_group in scheduler_outputs.scheduled_seq_groups:
                api_key = getattr(seq_group.sampling_params, "api_key", "unknown")
                time_last_iters = seq_group.get_last_latency(now)

                if prompt_run:
                    stats_by_api_key[api_key].num_prompt_tokens += len(seq_group.prompt_token_ids)
                    stats_by_api_key[api_key].time_to_first_tokens.append(time_last_iters)
                else:
                    stats_by_api_key[api_key].time_per_output_tokens.append(time_last_iters)

                stats_by_api_key[api_key].num_generation_tokens += seq_group.num_seqs()

        stat = Stats(
            now=now,
            num_running=0,
            num_swapped=0,
            num_waiting=0,
            gpu_cache_usage=gpu_cache_usage,
            cpu_cache_usage=cpu_cache_usage,
            num_prompt_tokens=0,
            num_generation_tokens=0,
            time_to_first_tokens=[],
            time_per_output_tokens=[],
            time_e2e_requests=[],
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
