from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from src.evaluation.evaluation_strategy import EvaluationStrategy
from src.modules.defaults import TrainerBase

class TaskFactory(ABC):
    
    @abstractmethod
    def create_dataset(self, cfg) -> Dataset:
        pass
    
    @abstractmethod
    def create_evaluation_strategy(self, cfg) -> EvaluationStrategy:
        pass

    @abstractmethod
    def create_loss_fn(self):
        pass