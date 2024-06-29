import glob
import importlib
from os.path import basename, dirname, isfile, join
from typing import List

from eval.core.base_task import BaseTask


class TaskRegistry:
    _task_registries = {}

    def __init__(self):
        print("---- TaskRegistry ----")
        for f in glob.glob(join(dirname(__file__), "*.py")):
            module_name = basename(f)[:-3]
            print(module_name)
            if not isfile(f) or module_name.startswith("_") or module_name.startswith("task"):
                continue
            module = importlib.import_module(f"eval.tasks.{module_name}")
            for attr in dir(module):
                task_cls = getattr(module, attr)
                if (
                    isinstance(task_cls, type)
                    and issubclass(task_cls, BaseTask)
                    and hasattr(task_cls, "task_name")
                    and task_cls.task_name != "base"
                ):
                    if task_cls.task_name == "custom":
                        for custom_task_name in task_cls.task_name_list:
                            self._task_registries[custom_task_name] = task_cls
                    else:
                        self.register(task_cls)

    def register(self, task_cls):
        if task_cls.task_name in self._task_registries:
            if self._task_registries[task_cls.task_name] == task_cls:
                # skip identical task class
                return task_cls
            raise ValueError(
                f"Task name {task_cls.task_name} has already been registered."
            )
        self._task_registries[task_cls.task_name] = task_cls
        return task_cls

    def get_tasks(self, task_names: List[str]) -> List[BaseTask]:
        task_instances = []
        for task_name in task_names:
            if task_name not in self._task_registries:
                raise ValueError(f"Task name {task_name} has not been registered.")
            task_cls = self._task_registries[task_name]

            if not task_cls.shots:
                if hasattr(task_cls, "task_name_list"):
                    task_instances.append(task_cls(shot=None, assign_task_name=task_name))
                else:
                    task_instances.append(task_cls())
                continue

            for shot in task_cls.shots:
                task_instances.append(task_cls(shot))
        return task_instances

TASK_REGISTRY = TaskRegistry()
