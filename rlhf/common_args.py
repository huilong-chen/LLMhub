from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CommonArgs:
    """
    一些常用的自定义参数
    """
    train_args_path: str = field(default="rlhf/args/ppo_config.py", metadata={"help": "当前模式训练参数,目前支持 PPO "})
    # 微调方法相关选择与配置
    train_mode: str = field(default='lora', metadata={"help": "选择采用的训练方式：[qlora, lora, full]"})
    use_dora: bool = field(default=False, metadata={"help": "仅在 train_mode==lora 时可以使用。是否使用Dora(一个基于Lora的变体)"})
    rlhf_type: str = field(default="PPO", metadata={"help": "选择使用的RLHF方法，PPO"})

    # lora相关配置
    lora_rank: Optional[int] = field(default=64, metadata={"help": "lora rank"})
    lora_alpha: Optional[int] = field(default=16, metadata={"help": "lora alpha"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "lora dropout"})
