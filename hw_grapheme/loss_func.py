import torch

import torch.nn.functional as F

from torch import nn
from functools import partial



def cutmix_criterion(preds1, preds2, preds3, targets):
    targets1, targets2, targets3, targets4, targets5, targets6, lam = targets[0], targets[1], targets[2], targets[3], targets[4], targets[5], targets[6]
    criterion = nn.CrossEntropyLoss(reduction='mean')
    return lam * criterion(preds1, targets1) + (1 - lam) * criterion(preds1, targets2) + lam * criterion(preds2, targets3) + (1 - lam) * criterion(preds2, targets4) + lam * criterion(preds3, targets5) + (1 - lam) * criterion(preds3, targets6)


def mixup_criterion(preds1, preds2, preds3, targets):
    targets1, targets2, targets3, targets4, targets5, targets6, lam = targets[0], targets[1], targets[2], targets[3], targets[4], targets[5], targets[6]
    criterion = nn.CrossEntropyLoss(reduction='mean')
    return lam * criterion(preds1, targets1) + (1 - lam) * criterion(preds1, targets2) + lam * criterion(preds2, targets3) + (1 - lam) * criterion(preds2, targets4) + lam * criterion(preds3, targets5) + (1 - lam) * criterion(preds3, targets6)


class Loss_combine(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, input, root_target, vowel_target, consonant_target, reduction='mean'):
        root_pred, vowel_pred, consonant_pred = input
        root_pred = root_pred.float()
        vowel_pred = vowel_pred.float()
        consonant_pred = consonant_pred.float()
        
#         print("In loss combine")
        
#         print("root_pred.shape", root_pred.shape)
#         print("vowel_pred.shape", vowel_pred.shape)
#         print("consonant_pred.shape", consonant_pred.shape)
        
#         print("root_target.shape", root_target.shape)
#         print("vowel_target.shape", vowel_target.shape)
#         print("consonant_target.shape", consonant_target.shape)
        
#         print("root_target.max", max(root_target))
#         print("vowel_target.max", max(vowel_target))
#         print("consonant_target.max", max(consonant_target))
        
#         print("root_target.min", min(root_target))
#         print("vowel_target.min", min(vowel_target))
#         print("consonant_target.min", min(consonant_target))
 
        return (
            0.7*F.cross_entropy(root_pred, root_target, reduction=reduction) + 
            0.1*F.cross_entropy(vowel_pred, vowel_target, reduction=reduction) + 
            0.2*F.cross_entropy(consonant_pred, consonant_target, reduction=reduction)
        )

# class MixUpLoss(nn.Module):
#     "Adapt the loss function `crit` to go with mixup."
    
#     def __init__(self, crit, reduction='mean'):
#         super().__init__()
#         if hasattr(crit, 'reduction'): 
#             self.crit = crit
#             self.old_red = crit.reduction
#             setattr(self.crit, 'reduction', 'none')
#         else: 
#             self.crit = partial(crit, reduction='none')
#             self.old_crit = crit
#         self.reduction = reduction
        
#     def forward(self, output, target):
#         if len(target.shape) == 2 and target.shape[1] == 7:
#             loss1, loss2 = self.crit(output,target[:,0:3].long()), self.crit(output,target[:,3:6].long())
#             d = loss1 * target[:,-1] + loss2 * (1-target[:,-1])
#         else:  d = self.crit(output, target)
#         if self.reduction == 'mean':    return d.mean()
#         elif self.reduction == 'sum':   return d.sum()
#         return d
    
    def get_old(self):
        if hasattr(self, 'old_crit'):  return self.old_crit
        elif hasattr(self, 'old_red'): 
            setattr(self.crit, 'reduction', self.old_red)
            return self.crit

