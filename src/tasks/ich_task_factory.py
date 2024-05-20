import torch.nn as nn

from src.evaluation.binary_evaluation import BinaryEvaluationStrategy
from src.tasks.task_factory import TaskFactory
from src.datasets.dataset_ich import ICHDataset

class ICHTaskFactory(TaskFactory):
    def create_evaluation_strategy(self, cfg):
        return BinaryEvaluationStrategy(cfg)
    
    def create_dataset(self, mode, cfg):
        return ICHDataset(mode=mode, cfg=cfg)
    
    def create_loss_fn(self):
        return nn.CrossEntropyLoss()