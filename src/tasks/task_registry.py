from typing import Type, Dict

from src.tasks.task_factory import TaskFactory
from src.tasks.isic_task_factory import ISICTaskFactory
from src.tasks.ich_task_factory import ICHTaskFactory

class TaskRegistry:
    _registry: Dict[str, Type[TaskFactory]] = {}

    @classmethod
    def register_task_factory(cls, task_type: str, factory: Type[TaskFactory]) -> None:
        cls._registry[task_type] = factory

    @classmethod
    def get_factory(cls, task_type: str) -> TaskFactory:
        if task_type in cls._registry:
            return cls._registry[task_type]()
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

# register tasks
TaskRegistry.register_task_factory("isic", ISICTaskFactory)
TaskRegistry.register_task_factory("ich", ICHTaskFactory)

