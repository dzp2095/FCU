import copy
import torch
import abc
import logging
from src.tasks.task_registry import TaskRegistry
from src.utils.device_selector import get_free_device_name

from src.modules.trainer import Trainer
class Client(abc.ABC):
    def __init__(self, name, args, cfg, poisoned=False, unlearn = False, post_train = False):
        self._name = name
        self.cfg = copy.deepcopy(cfg)
        self.poisoned = poisoned
        self._is_unlearn = unlearn
        self.setup()
        self.global_discriminator = None
        self.trainer = Trainer(args, self.cfg, post_train)
        if unlearn:
            self.trainer.initialize_unlearn()

        self.round = 0
        self._train_data_num = self.trainer.train_data_num
        self.trainer.client_label = int(name.split('_')[-1]) + 1

    def load_model(self, model_weights):
        self.trainer.load_model(model_weights)

    def setup(self):
        self.cfg["wandb"]["run_name"] = f"{self.cfg['wandb']['run_name']}_rounds_{self.cfg['fl']['rounds']}_{self.name}"
        if self.poisoned:
            self.cfg['dataset']['train'] = f"{self.cfg['dataset']['train']}/{self.name}/poisoned_train.csv"
        else:
            self.cfg['dataset']['train'] = f"{self.cfg['dataset']['train']}/{self.name}/train.csv"        
        self._train_path = self.cfg['dataset']['train']
        logging.info(f"{self.name}: Training data path: {self._train_path}")
        self.total_rounds = self.cfg['fl']['rounds']
        self.iter_per_round = self.cfg['fl']['local_iter']
        max_iter =  self.total_rounds * self.iter_per_round
        self.cfg["train"]["max_iter"] = max_iter
    
    def run(self):
        """train the model """
        logging.info(f"{self.name}: Starting training from round {self.round}")
        self.trainer.train(self.iter_per_round)
        self.round+=1

    def unlearn(self, iter):
        """unlearn the model """
        orginial_model = self.trainer.unlearn(iter)
        return orginial_model
    

    def unlearn_eval(self, model, metric_prefix):
        factory = TaskRegistry.get_factory(self.cfg['task'])
        test_dataset = factory.create_dataset(mode='test', cfg=self.cfg)
        data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.cfg["train"]["batch_size"], shuffle=False, 
                                                  num_workers=8, pin_memory=True)
        
        loss_fn = factory.create_loss_fn()
        device = get_free_device_name(gpu_exclude_list=self.cfg["train"]["gpu_exclude"])
        evaluation_strategy = factory.create_evaluation_strategy(self.cfg)

        test_metrics = evaluation_strategy.custom_eval(model, data_loader, device, loss_fn, metric_prefix)

        self.cfg['dataset']['val'] = self.train_path
        forget_set = factory.create_dataset(mode='val', cfg=self.cfg)
        data_loader = torch.utils.data.DataLoader(forget_set, batch_size=self.cfg["train"]["batch_size"], shuffle=False, 
                                                  num_workers=8, pin_memory=True)
        forget_metrics = evaluation_strategy.custom_eval(model, data_loader, device, loss_fn, metric_prefix)

        return test_metrics, forget_metrics

    @property
    def model(self):
        return self.trainer.model

    @property
    def name(self):
        return self._name

    @property
    def train_data_num(self):
        return self._train_data_num
        
    @property
    def class_nums(self):
        return self.trainer._class_nums
    
    @property
    def is_unlearn(self):
        return self._is_unlearn

    @property
    def train_path(self):
        return self._train_path

