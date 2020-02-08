from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import MemoryEfficientSwish
from torch import nn


class EfficientNet_0(nn.Module):
    def __init__(self, pretrain=False):
        super(EfficientNet_0, self).__init__()
        if pretrain:
            eff_net = EfficientNet.from_pretrained(f"efficientnet-b0")
        else:
            eff_net = EfficientNet.from_name(f"efficientnet-b0")

        self._conv_stem = eff_net._conv_stem
        self._bn0 = eff_net._bn0
        self._blocks = eff_net._blocks
        self._conv_head = eff_net._conv_head
        self._bn1 = eff_net._bn1
        self._avg_pooling = eff_net._avg_pooling
        self._dropout = eff_net._dropout
        
        self._root_fc = nn.Linear(1280, 168)
        # self._root_swish = MemoryEfficientSwish()
        
        self._vowel_fc = nn.Linear(1280, 11)
        # self._vowel_swish = MemoryEfficientSwish()
        
        self._consonant_fc = nn.Linear(1280, 7)
        # self._consonant_swish = MemoryEfficientSwish()
        
    def forward(self, x):
        # print(x.shape)
        x = self._conv_stem(x)
        x = self._bn0(x)
        for m in self._blocks:
            x = m(x)
        x = self._conv_head(x)
        x = self._bn1(x)
        x = self._avg_pooling(x)
        x = self._dropout(x)
        x = x.view(-1, 1280)
        
        # print(x.shape)
        x_root = self._root_fc(x)
        # print(x_root.shape)
        #x_root = self._root_swish(x_root)
        
        x_vowel = self._vowel_fc(x)
        #x_vowel = self._vowel_swish(x_vowel)
        
        x_consonant = self._consonant_fc(x)
        #x_consonant = self._consonant_swish(x_consonant)
            
        return x_root, x_vowel, x_consonant
    