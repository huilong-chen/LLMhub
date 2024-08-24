# DoRA: Weight-Decomposed Low-Rank Adaptation

https://zhuanlan.zhihu.com/p/683368968

https://arxiv.org/pdf/2402.09353

Dora是基于LoRA的变体.

DoRA可以分两步描述，其中第一步是将预训练的权重矩阵分解为幅度向量（m）和方向矩阵（V）。第二步是将LoRA应用于方向矩阵V并单独训练幅度向量m。

