from collections import defaultdict

from vllm.core.scheduler import Scheduler
from vllm.sequence import SequenceGroup

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
