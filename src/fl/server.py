import os
import logging
import torch
import copy
import weakref
from typing import List
import wandb
import numpy as np

from src.utils.device_selector import get_free_device_name
from src.utils.metric_logger import MetricLogger
from src.fl.client import Client
from src.tasks.task_registry import TaskRegistry

class Server:
    def __init__(self, clients: List[Client], is_unlearn, cfg):
        self.clients = clients
        self.r = 0
        for client in self.clients:
            client.server = weakref.proxy(self)
        self.factory = TaskRegistry.get_factory(cfg['task'])
        self.evaluation_strategy = self.factory.create_evaluation_strategy(cfg)

        self.save_checkpoints = True  # turn on to save the checkpoints
        self.rounds = cfg['fl']['rounds']
        self.is_unlearn = is_unlearn
        if is_unlearn:
            self.unlearn_clients = [client for client in clients if client.is_unlearn]
            self.clients = [client for client in clients if not client.is_unlearn]

        # weight ratio is the same for all clients, as the iteration number is the same
        self.weights_ratio = [1.0 / len(self.clients)] * len(self.clients) 
        self.eval_start_round = cfg['fl']['eval_start_round']
        self.test_start_round = cfg['fl']['test_start_round']
        self.save_start_round = cfg['fl']['save_start_round']
        self.metric_logger = MetricLogger()
        self.cfg = copy.deepcopy(cfg)

        if self.cfg["fl"]["wandb_global"]:
            self.wandb_init()
        self.load_model()
       
    def wandb_init(self):
        wandb.login(key=self.cfg["wandb"]["key"])
        self.wandb_id = wandb.util.generate_id()
        self.experiment = wandb.init(project=f'{self.cfg["wandb"]["project"]}', resume='allow', id=self.wandb_id, name=self.cfg["wandb"]["run_name"])
        self.experiment.config.update(
            dict(steps=self.cfg["train"]["max_iter"], batch_size=  self.cfg["train"]["batch_size"],
                 learning_rate = self.cfg["train"]["optimizer"]["lr"]), allow_val_change=True)
       
        logging.info("######## Running wandb logger")

    def wandb_upload(self, metric_logger):
        self.experiment.log(metric_logger._dict)

    def aggregate(self):
        # FedAvg
        w_avg = copy.deepcopy(self.clients[0].model.state_dict())
        for k in w_avg.keys():
            w_avg[k] = w_avg[k].to('cpu') * self.weights_ratio[0]

        for i in range(1, len(self.clients)):
            client_state_dict = {k: v.to('cpu') for k, v in self.clients[i].model.state_dict().items()}
            for k in w_avg.keys():
                w_avg[k] += client_state_dict[k] * self.weights_ratio[i]
        return w_avg

    def load_model(self):
        resume_path = self.cfg['train']['resume_path']
        if resume_path is not None and os.path.isfile(resume_path):
            logging.info(f"Resume from: {resume_path}")
            w = torch.load(resume_path)
            # send global model
            for client in self.clients:
                client.load_model(w)

    def save_model(self, w_avg, name):
        torch.save(w_avg,
            os.path.join(self.cfg["train"]["checkpoint_dir"], name + '.pth')
        )

    def run(self):
        self.best_metric = 0
        if self.is_unlearn:
            unlearned_model = self.run_unlearn()
            self.distribute_global_model(unlearned_model.state_dict())
            logging.info("Evluation result for the initial unlearned model")
            self.run_evaluation(unlearned_model.state_dict())

        for self.r in range(self.rounds):
            self.run_clients()
            w_avg = self.aggregate()
            self.distribute_global_model(w_avg)
            self.run_evaluation(w_avg)
        
        if self.cfg["fl"]["wandb_global"]:
            self.experiment.finish()

    def run_unlearn(self):
        unlearn_iter = self.cfg['fl']['unlearn']['unlearn_iter']
        fusion_interval = self.cfg['fl']['unlearn']['fusion_interval']
        unlearn_round = unlearn_iter // fusion_interval
        # initialize dataset
        dataset = self.factory.create_dataset(mode='test', cfg=self.cfg)
        test_data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.cfg["train"]["batch_size"], shuffle=False, 
                                                  num_workers=8, pin_memory=True)
        loss_fn = self.factory.create_loss_fn()
        device = get_free_device_name(gpu_exclude_list=self.cfg["train"]["gpu_exclude"])
        dataset = self.factory.create_dataset(mode='forgotten', cfg=self.cfg)
        forgotten_data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.cfg["train"]["batch_size"], shuffle=False, 
                                                  num_workers=8, pin_memory=True)
        dataset = self.factory.create_dataset(mode='remembered', cfg=self.cfg)
        remembered_data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.cfg["train"]["batch_size"], shuffle=False,
                                                    num_workers=8, pin_memory=True)
        
        logging.info(f"######## Unlearn start, total rounds: {unlearn_round}")
        for i in range(unlearn_round):
            logging.info(f"######## Unlearn round {i}")
            for client in self.unlearn_clients:
                if i == 0:
                    client.trainer.initialize_unlearn()
                    metrics = self.evaluation_strategy.unlearn_eval(client.model, test_data_loader, forgotten_data_loader, remembered_data_loader, device, loss_fn)
                    logging.info(f"######## Unlearn eval for pretrained mdoel, client {client.name}, metrics: {metrics}")
                client.unlearn(fusion_interval)
                metrics = self.evaluation_strategy.unlearn_eval(client.model, test_data_loader, forgotten_data_loader, remembered_data_loader, device, loss_fn)
                logging.info(f"######## Unlearn eval for unlearn mdoel, client {client.name}, metrics: {metrics}")
        # to do: support multiple unlearn clients
        return self.unlearn_clients[0].model

    def run_clients(self):
        for client in self.clients:
            client.run()
    
    def distribute_global_model(self, w_avg):
        for client in self.clients:
            client.load_model(w_avg)
        
    def run_evaluation(self, w_avg):
        
        # save the best model on the validation set
        if (self.r >= self.eval_start_round):
            self.global_validate(w_avg)
            self.save_best_model(w_avg, self.cfg['eval']['metric'])
        
        # save the global model automatically
        if (self.r >= self.save_start_round):
            self.save_model(w_avg, f"global_model_round_{self.r}")

        if (self.r >= self.test_start_round):
            self.global_test(w_avg)
        
        # upload the metrics to wandb
        if self.cfg["fl"]["wandb_global"]:
            self.wandb_upload(self.metric_logger)
    
    def global_test(self, w_avg):
        model = copy.deepcopy(self.clients[0].model)
        model.load_state_dict(w_avg)
        loss_fn = self.factory.create_loss_fn()
        device = get_free_device_name(gpu_exclude_list=self.cfg["train"]["gpu_exclude"])
        if self.is_unlearn:
            # initialize dataset
            dataset = self.factory.create_dataset(mode='test', cfg=self.cfg)
            test_data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.cfg["train"]["batch_size"], shuffle=False, 
                                                    num_workers=8, pin_memory=True)

            dataset = self.factory.create_dataset(mode='forgotten', cfg=self.cfg)
            forgotten_data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.cfg["train"]["batch_size"], shuffle=False, 
                                                    num_workers=8, pin_memory=True)
            dataset = self.factory.create_dataset(mode='remembered', cfg=self.cfg)
            remembered_data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.cfg["train"]["batch_size"], shuffle=False,
                                                        num_workers=8, pin_memory=True)
            metrics = self.evaluation_strategy.unlearn_eval(model, test_data_loader, forgotten_data_loader, remembered_data_loader, device, loss_fn)
            self.metric_logger.update(**metrics)
            logging.info(f"######## Unlearn eval for unlearn mdoel, round {self.r}, metrics: {metrics}")
        else:

            dataset = self.factory.create_dataset(mode='test', cfg=self.cfg)
            data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.cfg["train"]["batch_size"], shuffle=False, 
                                                    num_workers=8, pin_memory=True)
            metrics = self.evaluation_strategy.test(model, data_loader, device, loss_fn)
            self.metric_logger.update(**metrics)
            logging.info(f"######## Global test: {metrics}")

    def global_validate(self, w_avg):
        model = copy.deepcopy(self.clients[0].model)
        model.load_state_dict(w_avg)
        dataset = self.factory.create_dataset(mode='val', cfg=self.cfg)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.cfg["train"]["batch_size"], shuffle=False, 
                                                  num_workers=8, pin_memory=True)
        loss_fn = self.factory.create_loss_fn()
        device = get_free_device_name(gpu_exclude_list=self.cfg["train"]["gpu_exclude"])
        metrics = self.evaluation_strategy.validate(model, data_loader, device, loss_fn)
        self.metric_logger.update(**metrics)
        logging.info(f"######## Global validate: {metrics}")

    def save_best_model(self, w_avg, metric):
        if metric in self.metric_logger._dict:
            eval_metric = self.metric_logger._dict[metric]
            if self.best_metric < eval_metric:
                self.best_metric = eval_metric
                self.save_model(w_avg, f"best_{metric}".replace('/',"_"))
                logging.info(f"######## New best {metric}: {self.best_metric}")
        else:
            logging.info(f"######## No {metric} in metric_logger")