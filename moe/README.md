# 什么是 MoE ？

MOE主要由两个关键点组成：

一是将传统Transformer中的FFN（前馈网络层）替换为多个稀疏的专家层（Sparse MoE layers）。每个专家本身是一个独立的神经网络，实际应用中，这些专家通常是前馈网络 (FFN)，但也可以是更复杂的网络结构。

二是门控网络或路由：此部分用来决定输入的token分发给哪一个专家。

# 怎么实现一个 MoE ？

## 创建一个专家模型

等价于创建一个 MLP

```python
from torch import nn
dropout = 0.1
class Expert(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
```

## 创建 TopKRouter

即创建MOE的路由部分。

假设我们定义了4个专家，路由取前2名专家，即expert=4， top_k=2。

接收注意力层的输出作为输入X，即将输入从（Batch size，Tokens，n_embed）的形状（2，4，32）投影到对应于（Batch size，Tokens，num_experts）的形状（2，4，4）其中num_experts即expert=4。

其中返回的indices可以理解为对于每个token的4个专家来说，选的两个专家的序号索引。

## 添加噪声 & 设置专家容量

从本质上讲，我们不希望所有token都发送给同一组“受青睐”的expert。需要一个良好平衡，因此，将标准正态噪声添加到来自门控线性层的logits。

为了防止所有tokens都被一个或几个expert处理，我们需要设置一个专家容量。如果某个专家处理超过容量的tokens后就会给他截断。

## 构建一个完整的 sparse moe module

前面的操作主要是获取了router分发的结果，获取到这些结果后我们就可以将router乘给对应的token。这种选择性加权乘法最终构成了稀疏MOE。

以专家为单位遍历每个专家，抽取每个专家对于所有token中在前top_k的tokens。

## 将 MoE 与 transformer 结合

即用moe替代MLP层。

```python
class Block(nn.Module):
    def __init__(self, n_embed, n_head, num_experts, top_k):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.smoe = SparseMoE(n_embed, num_experts, top_k)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.smoe(self.ln2(x))
        return x
```



