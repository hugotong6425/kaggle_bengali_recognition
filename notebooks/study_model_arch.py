# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import MemoryEfficientSwish
import torch
from torch import nn

eff_b0 = EfficientNet.from_pretrained('efficientnet-b0', num_classes=168)

eff_b0


class EfficientNet_0(nn.Module):
    def __init__(self):
        super(EfficientNet_0, self).__init__()
        #eff_b0 = EfficientNet.from_pretrained('efficientnet-b0', num_classes=168)
        eff_b0 = EfficientNet.from_name('efficientnet-b0')

        self._conv_stem = eff_b0._conv_stem
        self._bn0 = eff_b0._bn0
        self._blocks = eff_b0._blocks
        self._conv_head = eff_b0._conv_head
        self._bn1 = eff_b0._bn1
        self._avg_pooling = eff_b0._avg_pooling
        self._dropout = eff_b0._dropout
        
        self._root_fc = nn.Linear(1280, 168)
        # self._root_swish = MemoryEfficientSwish()
        
        self._vowel_fc = nn.Linear(1280, 11)
        # self._vowel_swish = MemoryEfficientSwish()
        
        self._consonant_fc = nn.Linear(1280, 7)
        # self._consonant_swish = MemoryEfficientSwish()
        
    def forward(self, x):
        x = self._conv_stem(x)
        x = self._bn0(x)
        for m in self._blocks:
            x = m(x)
        x = self._conv_head(x)
        x = self._bn1(x)
        x = self._avg_pooling(x)
        x = self._dropout(x)
        x = x.view(-1, 1280)
        
        print(x.shape)
        x_root = self._root_fc(x)
        x_root = self._root_swish(x_root)
        
        x_vowel = self._vowel_fc(x)
        x_vowel = self._vowel_swish(x_vowel)
        
        x_consonant = self._consonant_fc(x)
        x_consonant = self._consonant_swish(x_consonant)
            
        return x_root, x_vowel, x_consonant



eff_b0

sample = torch.ones(1, 3, 448, 448)

eff_b0(sample)

eff_seq = EfficientNet_0()

eff_seq

eff_seq(sample)



