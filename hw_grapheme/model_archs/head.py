import torch
import torch.nn as nn
import torch.nn.functional as F


class Head_1fc(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(Head_1fc, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(input_dim, out_dim)

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# +
class AdaptiveConcatPool2d(nn.Module):
    "Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`."
    def __init__(self, sz=None):
        super(AdaptiveConcatPool2d, self).__init__()
        "Output will be 2*sz or 2 if sz is None"
        self.output_size = sz or 1
        self.ap = nn.AdaptiveAvgPool2d(self.output_size)
        self.mp = nn.AdaptiveMaxPool2d(self.output_size)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)

    
class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        #inlining this saves 1 second per epoch (V100 GPU) vs having a temp x and then returning x(!)
        return x *( torch.tanh(F.softplus(x)))
    
    
def bn_drop_lin(n_in: int, n_out: int, bn: bool = True, p: float = 0.0, actn=None):
    "Sequence of batchnorm (if `bn`), dropout (with `p`) and linear (`n_in`,`n_out`) layers followed by `actn`."
    layers = [nn.BatchNorm1d(n_in)] if bn else []
    if p != 0:
        layers.append(nn.Dropout(p))
    layers.append(nn.Linear(n_in, n_out))
    if actn is not None:
        layers.append(actn)
    return layers


class Flatten(nn.Module):
    "Flatten `x` to a single dimension, often used at the end of a model. `full` for rank-1 tensor"

    def __init__(self, full: bool = False):
        super(Flatten, self).__init__()
        self.full = full

    def forward(self, x):
        return x.view(-1) if self.full else x.view(x.size(0), -1)


class Head_3fc(nn.Module):
    def __init__(self, input_dim, out_dim, ps=0.5):
        super().__init__()
        layers = (
            [AdaptiveConcatPool2d(), Mish(), Flatten()]
            + bn_drop_lin(input_dim * 2, 1024, True, ps, Mish())
            + bn_drop_lin(1024, out_dim, True, ps)
        )
        self.fc = nn.Sequential(*layers)
        self._init_weight()
        
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()

    def forward(self, x):
        return self.fc(x)
