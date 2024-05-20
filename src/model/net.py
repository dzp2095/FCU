# encoding: utf-8

"""
The main CheXpert models implementation.
Including:
    DenseNet-121
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model import densenet
import copy
from functools import reduce
import numpy as np

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

def get_module_by_name(module, access_string):
    names = access_string.split(sep='.')
    return reduce(getattr, names, module)

def set_module_by_name(module, access_string, target):
    names = access_string.split(sep='.')
    last = names[-1]
    names = names[:-1]
    module = reduce(getattr, names, module)
    setattr(module, last, target)

class DenseNet121(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self, cfg):
        super(DenseNet121, self).__init__()
        num_classes = cfg['model']['num_classes']
        pretrained = cfg['model']['pretrained']
        drop_rate = cfg['model']['drop_rate']

        self._num_classes = num_classes
        self.densenet121 = densenet.densenet121(pretrained=pretrained, drop_rate=drop_rate)
        
        num_ftrs = self.densenet121.classifier.in_features
        self._num_ftrs = num_ftrs
        # delete original classifer layer 
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, num_classes),
            #nn.Sigmoid()
        )
        
        # Official init from torch repo.
        for m in self.densenet121.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

        self.drop_rate = drop_rate
        self.drop_layer = nn.Dropout(p=drop_rate)

    def forward(self, x):
        # feature shape 1024
        features = self.densenet121.features(x)
        
        features = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(features, (1, 1)).view(features.size(0), -1)

        if self.drop_rate>0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        logits = self.densenet121.classifier(out)
        return out, logits

    @property
    def num_ftrs(self):
        return self._num_ftrs
    
    @property
    def num_classes(self):
        return self._num_classes
    
def frequency_guided_model_fusion(low_freq_model: DenseNet121, high_freq_model: DenseNet121, low_freq_ratio: float = 0.5):
    fusion_model = copy.deepcopy(high_freq_model)
    for key in low_freq_model.state_dict().keys():
        # skip batch norm layers
        if 'norm' in key:
            continue
        if 'weight' in key:
            if 'classifier' in key:
                fusion_weight = weight_level_fusion(low_freq_model.state_dict()[key], high_freq_model.state_dict()[key], is_conv=False, low_freq_ratio=low_freq_ratio)
            elif 'conv' in key:
                fusion_weight = weight_level_fusion(low_freq_model.state_dict()[key], high_freq_model.state_dict()[key], is_conv=True, low_freq_ratio=low_freq_ratio)
            fusion_model.state_dict()[key].data.copy_(fusion_weight)
    return fusion_model

def weight_level_fusion(low_freq_weight, high_freq_weight, is_conv, low_freq_ratio):
    if is_conv:
        N, C, D1, D2 = low_freq_weight.size()
    else:
        N , C  = 1, 1
        D1, D2 = low_freq_weight.size()
    
    if is_conv:
        low_freq_weight = low_freq_weight.permute(1, 2, 3, 0).reshape((C*D1, D2*N))
        high_freq_weight = high_freq_weight.permute(1, 2, 3, 0).reshape((C*D1, D2*N))
    low_freq_weight = low_freq_weight.cpu().numpy()
    high_freq_weight = high_freq_weight.cpu().numpy()

    # FFT
    low_freq_weight_fft = np.fft.fft2(low_freq_weight, axes=(-2, -1))
    amp_fft, pha_fft = np.abs(low_freq_weight_fft), np.angle(low_freq_weight_fft) 
    low_freq_weight_fft = np.fft.fftshift(amp_fft, axes=(-2, -1))

    high_freq_weight_fft = np.fft.fft2(high_freq_weight, axes=(-2, -1))
    amp_fft, pha_fft = np.abs(high_freq_weight_fft), np.angle(high_freq_weight_fft)
    high_freq_weight_fft = np.fft.fftshift(amp_fft, axes=(-2, -1))

    fusion_weight_fft = copy.deepcopy(high_freq_weight_fft)

    # fusion
    h, w = low_freq_weight_fft.shape
    b_h = (np.floor(h *low_freq_ratio / 2)).astype(int)
    b_w = (np.floor(w *low_freq_ratio / 2)).astype(int)
    c_h = np.floor(h/2.0).astype(int)
    c_w = np.floor(w/2.0).astype(int)

    h1 = c_h-b_h
    h2 = c_h+b_h
    w1 = c_w-b_w
    w2 = c_w+b_w
    
    fusion_weight_fft[h1:h2, w1:w2] = low_freq_weight_fft[h1:h2, w1:w2]

    # IFFT
    fusion_weight_fft = np.fft.ifftshift(fusion_weight_fft, axes=(-2, -1))
    fusion_weight_fft = fusion_weight_fft * np.exp(1j * pha_fft)
    fusion_weight = np.fft.ifft2(fusion_weight_fft, axes=(-2, -1))

    fusion_weight = torch.FloatTensor(fusion_weight)
    if is_conv:
        fusion_weight = fusion_weight.reshape((C, D1, D2, N)).permute(3, 0, 1, 2)
    return fusion_weight

if __name__ == "__main__":
    import yaml
    path = "/home/user/fedmu/configs/ich/run_conf.yaml"
    cfg = yaml.safe_load(open(path))

    model = DenseNet121(cfg)
    print(model)