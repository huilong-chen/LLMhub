from dataclasses import dataclass, field
from typing import Any, Dict, List

@dataclass
class Sample:
    """
    One sample is a single row in task data.
    """

    # Raw data of a single row.
    # This will be saved to csv file with the prediction results.
    raw_data: Dict[str, Any]
    # Messages to build the prompt.
    messages: List[Dict[str, str]]
    # Task config used in `Pipeline.generate()`.
    task_config: Dict[str, Any]

    use_template: bool = field(default=True)
    # These fields are filled during the generation process.
    prompts: List[str] = field(default_factory=list, init=False)
    model_outputs: List[Any] = field(default_factory=list, init=False)

    def __post_init__(self):
        self.raw_data = self.raw_data.copy()

