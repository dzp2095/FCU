import yaml
import logging

import random
import numpy as np
import torch
from torch.backends import cudnn
from pathlib import Path
from datetime import datetime

from src.fl.client import Client
from src.fl.server import Server
from src.utils.args_parser import args

if args.deterministic:
    cudnn.benchmark = True
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

now = str(datetime.timestamp(datetime.now()))

Path(f'./log').mkdir(parents=True, exist_ok=True)
logging.basicConfig(filename=f'log/log_fl_{now}.txt',
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level=logging.INFO)

def initialize_client(i, args, cfg, status='normal'):
    client_id = f'client_{i}'
    if status == 'unlearn':
        client = Client(client_id, args, cfg, unlearn=True)
        logging.info(f'Client {i} needs to be unlearned')
    else:  # Normal or post-train client
        post_train = status == 'post_train'
        client = Client(client_id, args, cfg, post_train=post_train)
    return client

if __name__ == "__main__":

    try:
        cfg = yaml.safe_load(open(args.config))
        cfg["train"]["checkpoint_dir"] = f'{cfg["train"]["checkpoint_dir"]}{now}'
        excluded_clients = args.excluded_clients
        poisoned_clients = args.poisoned_clients
        is_unlearn = args.is_unlearn
        if excluded_clients is not None and len(excluded_clients) > 0:
            cfg["wandb"]["project"] = cfg["wandb"]["project"] + '_exclude_' + '_'.join([str(i) for i in excluded_clients])
        
        if args.eval_only:
            server = Server([], cfg)
            cfg["train"]["resume_path"] = args.resume_path
            server.global_test()
        else:
            client_num = cfg['fl']['num_clients']
            clients = []
            if is_unlearn:
                for i in range(client_num):
                    if str(i) in excluded_clients:
                        client = initialize_client(i, args, cfg, status='unlearn')
                    else:
                        client = initialize_client(i, args, cfg, status='post_train')
                    clients.append(client)
            else:
                for i in range(client_num):
                    if str(i) in excluded_clients:
                        logging.info(f'Client {i} is excluded')
                        continue
                    else:
                        client = initialize_client(i, args, cfg, status='normal')
                    clients.append(client)
            server = Server(clients, is_unlearn, cfg)
            server.run()
    except Exception as e:
        logging.critical(e, exc_info=True)
