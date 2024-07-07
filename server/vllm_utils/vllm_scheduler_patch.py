from collections import defaultdict

from vllm.core.scheduler import Scheduler
from vllm.sequence import SequenceGroup


def patch_scheduler_with_tool(tools):
    origin_schedule_fn = Scheduler._schedule

    def call_tool(scheduled_seq_groups):
        filtered_seq_groups = []
        for seq_group in scheduled_seq_groups:
            has_not_done_seq = False
            for seq in seq_group.seqs_dict.values():
                output_token_ids = seq.data.output_token_ids
                if not tools.check_ends_with_result_token(output_token_ids):
                    # 如果没有调用工具
                    continue
                tool_ids = tools.get_tool_ids(output_token_ids)
                if tool_ids is None or tools.check_in_cache(tool_ids):
                    # 如果已经在缓存中
                    continue
                has_not_done_seq = True
                tools.call_tool_async(tool_ids)
            if not has_not_done_seq:
                # 过滤掉需要等待工具返回结果的 seq_group
                filtered_seq_groups.append(seq_group)
        if len(filtered_seq_groups) == 0:
            # 如果全被过滤，则等待直到至少有一个工具调用结果返回
            tools.wait()

        return filtered_seq_groups

    def patched_schedule(self):
        scheduler_outputs = origin_schedule_fn(self)
        if not scheduler_outputs.scheduled_seq_groups:
            return scheduler_outputs
        filtered_seq_groups = []
        while len(filtered_seq_groups) == 0:
            filtered_seq_groups = call_tool(scheduler_outputs.scheduled_seq_groups)

        scheduler_outputs.scheduled_seq_groups = filtered_seq_groups

        return scheduler_outputs

    Scheduler._schedule = patched_schedule


def patch_scheduler_add_seq_group():

    def patched_add_seq_group(self, seq_group: SequenceGroup) -> None:
        """
        默认先来的请求都进入队尾，修改后支持根据 api_key 插队
        比如队列目前为 [A1 B1 A2 B2 A3 A4 A5], 请求 B3 入队
        则插队后队列为 [A1 B1 A2 B2 A3 *B3* A4 A5]
        时间复杂度 O(n)
        """

        request_api_key = getattr(seq_group.sampling_params, "api_key", None)
        if not request_api_key or request_api_key == "unknown":
            self.waiting.append(seq_group)
            return

        # 统计每个 api_key 的请求数量，当发生不平衡时插队
        api_key_counter = defaultdict(int)
        i = 0
        for seq in self.waiting:
            api_key = getattr(seq.sampling_params, "api_key", None)
            if not api_key or api_key == "unknown":
                break
            if api_key_counter[api_key] > api_key_counter[request_api_key]:
                break
            api_key_counter[api_key] += 1
            i += 1
        self.waiting.insert(i, seq_group)

    Scheduler.add_seq_group = patched_add_seq_group
