import torch.nn as nn

from src.evaluation.multiclass_evaluation import MulticlassEvaluationStrategy
from src.tasks.task_factory import TaskFactory
from src.datasets.dataset_isic import ISICDataset

class ISICTaskFactory(TaskFactory):
    def create_evaluation_strategy(self, cfg):
        return MulticlassEvaluationStrategy(cfg)
    
    def create_dataset(self, mode, cfg):
        return ISICDataset(mode=mode, cfg=cfg)
    
    def create_loss_fn(self):
        return nn.CrossEntropyLoss()