import logging
import torch
import os
import numpy as np
from src.utils.args_parser import args

def get_free_gpu(gpu_exclude_list=[]):
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    memory_available = [memory_available[i] for i in range(len(memory_available)) if i not in gpu_exclude_list]
    os.remove('tmp')
    return np.argmax(memory_available)

def get_free_device_name(gpu_exclude_list=[]):
    if (torch.cuda.is_available() and args.gpu):
        gpu = get_free_gpu(gpu_exclude_list=gpu_exclude_list)
        logging.info(f'Using GPU: {gpu}')
        return f'cuda:{gpu}'
    else:
        return 'cpu'