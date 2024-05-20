import logging
import torch 
import numpy as np
import copy
from tqdm import tqdm

from src.modules.defaults import TrainerBase
from src.modules import hooks
from src.model.net import DenseNet121, frequency_guided_model_fusion
from src.tasks.task_registry import TaskRegistry
from src.datasets.sampler import TrainingSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau

class Trainer(TrainerBase):

    def __init__(self, args, cfg, is_post_train) -> None:
        super().__init__(args, cfg)
        self.register_hooks(self.build_hooks())
        self.is_post_train = is_post_train
        # if is_post_train:
        #     self.pretrained_model = DenseNet121(cfg=self.cfg)
        #     try:
        #         self.pretrained_model.load_state_dict(torch.load(self.cfg["model"]["model_path"]))
        #         logging.info("Pretrained model loaded for post training, path: " + self.cfg["model"]["model_path"])
        #     except:
        #         logging.info("pretrained model not found, which is necessary for post training")
        #         raise FileNotFoundError("pretrained model not found")
        #     self.model = copy.deepcopy(self.pretrained_model)

    def build_model(self):
        self.model = DenseNet121(cfg=self.cfg)
    
    def init_dataloader(self):
        batch_size = self.cfg["train"]["batch_size"]
        factory = TaskRegistry.get_factory(self.cfg['task'])
        dataset = factory.create_dataset(mode='train', cfg=self.cfg)
        self._loss_fn = factory.create_loss_fn()
        self._train_data_num = len(dataset)
        self.iter_per_epoch = self._train_data_num // batch_size
        
        batch_size = self.cfg["train"]["batch_size"]
        num_workers = self.cfg["train"]["num_workers"]
        seed = self.cfg["dataset"]["seed"]
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,
                                                  sampler = TrainingSampler(len(dataset),seed=seed))
        self._data_iter = iter(data_loader)

    def load_model(self, model_weights):
        self.model.load_state_dict(model_weights, strict=False)
        # need to construct a new optimizer for the new network
        old_scheduler = copy.deepcopy(self.lr_scheduler.state_dict())
        old_optimizer = copy.deepcopy(self.optimizer.state_dict())
        self.build_optimizer()
        self.optimizer.load_state_dict(old_optimizer)
        self.build_schedular(self.optimizer)
        self.lr_scheduler.load_state_dict(old_scheduler)

    def build_hooks(self):
        ret = [hooks.Timer()]
        if self.cfg["hooks"]["wandb"]:
            ret.append(hooks.WAndBUploader(self.cfg))
        return ret
    
    def before_train(self):
        return super().before_train()

    def after_train(self):
        return super().after_train()
    
    def run_step(self):
        self.model.train()
        self.model.to(self.device)
        _, image, label = next(self._data_iter)
        image = image.to(self.device)
        label = label.to(self.device)
        # class_indices = label.argmax(dim=1)
        _, output = self.model(image)
        loss = self.loss_fn(output, label)
        
        self.loss_logger.update(loss=loss)
        self.metric_logger.update(loss=loss)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def initialize_unlearn(self):
        try:
            self.model.load_state_dict(torch.load(self.cfg["model"]["model_path"]))
            self.pretrained_model = copy.deepcopy(self.model)
            logging.info("Pretrained model loaded for unlearn, path: " + self.cfg["model"]["model_path"])
        except:
            logging.info("Target model not found, which is necessary for unlearn")
            raise FileNotFoundError("Target unlearn model not found")

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg["fl"]["unlearn"]["optimizer"]['lr'],
            betas=(self.cfg["fl"]["unlearn"]["optimizer"]['beta1'],self.cfg["fl"]["unlearn"]["optimizer"]['beta2']), 
            weight_decay=self.cfg["fl"]["unlearn"]["optimizer"]['weight_decay'])
        
        self.lr_scheduler = ReduceLROnPlateau(self.optimizer, factor=self.cfg["fl"]["unlearn"]["lr_scheduler"]["factor"], 
        patience=self.cfg["fl"]["unlearn"]["lr_scheduler"]["patience"], verbose=True, min_lr=self.cfg["fl"]["unlearn"]["lr_scheduler"]["min_lr"])

    def unlearn(self, iter):
        
        self.model.to(self.device)
        self.downgraded_model = DenseNet121(cfg=self.cfg)
        
        self.downgraded_model.to(self.device)
        self.pretrained_model.to(self.device)

        self.model.train()
        self.downgraded_model.eval()

        with tqdm(total=iter) as pbar:
            for _ in range(iter):
                _, image, label = next(self._data_iter)
                image = image.to(self.device)
                label = label.to(self.device)
                output, _ = self.model(image)
                with torch.no_grad(): 
                    downgraded_output, _ = self.downgraded_model(image)
                    pretrained_output, _ = self.pretrained_model(image)
                loss = self.contrastive_loss(output, pretrained_output, downgraded_output)                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                pbar.update(1)
                pbar.set_description(f"Unlearning: Loss {loss.item()}")
        
        # keep the low frequency part of the pretrained model
        low_freq_ratio = self.cfg['fl']['unlearn']['low_freq']
        fusion_model = frequency_guided_model_fusion(self.model, self.pretrained_model, low_freq_ratio)
        self.load_model(fusion_model.state_dict())

    # pull close to hs1 and push away from hs0
    def contrastive_loss(self, hs, hs0, hs1):
        cs = torch.nn.CosineSimilarity(dim=-1)
        sims0 = cs(hs, hs0)
        sims1 = cs(hs, hs1)

        sims = 2.0 * torch.stack([sims0, sims1], dim=1)
        labels = torch.LongTensor([1] * hs.shape[0])
        labels = labels.to(hs.device)

        criterion = torch.nn.CrossEntropyLoss()
        ct_loss = criterion(sims, labels)
        return ct_loss

    @property
    def train_data_num(self):
        return self._train_data_num
    
    @property
    def loss_fn(self):
        return self._loss_fn
    
