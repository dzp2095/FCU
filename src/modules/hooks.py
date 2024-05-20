
import logging
import torch 
import wandb
import copy
from datetime import datetime

from src.modules.defaults import HookBase

class Timer(HookBase):

    def before_train(self):
        self.tick = datetime.now()
        logging.info("Begin training at: {}".format(self.tick.strftime("%Y-%m-%d %H:%M:%S")))
        logging.info("######## Running Timer")

    def after_train(self):
        tock = datetime.now()
        logging.info("\nBegin training at: {}".format(self.tick.strftime("%Y-%m-%d %H:%M:%S")))
        logging.info("Finish training at: {}".format(tock.strftime("%Y-%m-%d %H:%M:%S")))
        logging.info("Time spent: {}\n".format(str(tock - self.tick).split('.')[0]))

class WAndBUploader(HookBase):
    def __init__(self, cfg):
        self.cfg = copy.deepcopy(cfg)
        wandb.login(key=self.cfg["wandb"]["key"])
        self.wandb_id = wandb.util.generate_id()

    def before_train(self):
        self.experiment = wandb.init(project=f'{self.cfg["wandb"]["project"]}', resume='allow', id=self.wandb_id, name=self.cfg["wandb"]["run_name"])
        self.val_interval = self.cfg["train"]["eval_interval"]
        self.experiment.config.update(
            dict(steps=self.trainer.max_iter, batch_size=self.cfg["train"]["batch_size"],
                 learning_rate = self.cfg["train"]["optimizer"]["lr"]), allow_val_change=True)
       
        logging.info("######## Running wandb logger")

    def after_step(self):
        wandb_dict = {}
        metric_names = {"loss", "val_loss"}
        for metric_name in metric_names:
            if metric_name in self.trainer.metric_logger._dict:
                wandb_dict.update({metric_name: self.trainer.metric_logger._dict[metric_name]})

        if self.trainer.iter % self.val_interval == 0:
            wandb_dict.update(self.trainer.metric_logger._dict)
        self.experiment.log(wandb_dict)

    def after_train(self):
        self.experiment.finish()