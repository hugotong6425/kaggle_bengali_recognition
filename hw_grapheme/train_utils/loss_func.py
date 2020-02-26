# -*- coding: utf-8 -*-
import torch

import torch.nn.functional as F

from torch import nn
from functools import partial


class CombineLabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, ε: float = 0.1, reduction="mean"):
        super().__init__()
        self.ε, self.reduction = ε, reduction

    def forward(self, input, root_target, vowel_target, consonant_target):
        root_pred, vowel_pred, consonant_pred = input
        root_pred = root_pred.float()
        vowel_pred = vowel_pred.float()
        consonant_pred = consonant_pred.float()

        c_root = root_pred.size()[-1]
        c_vowel = vowel_pred.size()[-1]
        c_consonant = consonant_pred.size()[-1]

        log_preds_root = F.log_softmax(root_pred, dim=-1)
        log_preds_vowel = F.log_softmax(vowel_pred, dim=-1)
        log_preds_consonant = F.log_softmax(consonant_pred, dim=-1)

        loss_root = reduce_loss(-log_preds_root.sum(dim=-1), self.reduction)
        loss_vowel = reduce_loss(-log_preds_vowel.sum(dim=-1), self.reduction)
        loss_consonant = reduce_loss(-log_preds_consonant.sum(dim=-1), self.reduction)

        nll_root = F.nll_loss(log_preds_root, root_target, reduction=self.reduction)
        nll_vowel = F.nll_loss(log_preds_vowel, vowel_target, reduction=self.reduction)
        nll_consonant = F.nll_loss(
            log_preds_consonant, consonant_target, reduction=self.reduction
        )

        l1 = lin_comb(loss_root / c_root, nll_root, self.ε)
        l2 = lin_comb(loss_vowel / c_vowel, nll_vowel, self.ε)
        l3 = lin_comb(loss_consonant / c_consonant, nll_consonant, self.ε)
        return combine_loss(l1, l2, l3)


class Weighted_entropy(nn.Module):
    def __init__(self, weights=[0.5, 0.25, 0.25]):
        super().__init__()
        self.weights = weights

    def forward(
        self, input, root_target, vowel_target, consonant_target, reduction="mean"
    ):
        root_pred, vowel_pred, consonant_pred = input
        root_pred = root_pred.float()
        vowel_pred = vowel_pred.float()
        consonant_pred = consonant_pred.float()

        l1 = F.cross_entropy(root_pred, root_target, reduction=reduction)
        l2 = F.cross_entropy(vowel_pred, vowel_target, reduction=reduction)
        l3 = F.cross_entropy(consonant_pred, consonant_target, reduction=reduction)
        return combine_loss(l1, l2, l3, self.weights)


def cross_entropy_criterion(preds1, preds2, preds3, targets, weights=[0.5, 0.25, 0.25]):
    targets1, targets2, targets3 = targets[0], targets[1], targets[2]
    criterion = nn.CrossEntropyLoss(reduction="mean")
    l1 = criterion(preds1, targets1)
    l2 = criterion(preds2, targets2)
    l3 = criterion(preds3, targets3)
    return combine_loss(l1, l2, l3, weights)


def cutmix_criterion(preds1, preds2, preds3, targets, weights=[0.5, 0.25, 0.25]):
    targets1, targets2, targets3, targets4, targets5, targets6, lam = (
        targets[0],
        targets[1],
        targets[2],
        targets[3],
        targets[4],
        targets[5],
        targets[6],
    )
    criterion = nn.CrossEntropyLoss(reduction="mean")
    l1 = lam * criterion(preds1, targets1) + (1 - lam) * criterion(preds1, targets2)
    l2 = lam * criterion(preds2, targets3) + (1 - lam) * criterion(preds2, targets4)
    l3 = lam * criterion(preds3, targets5) + (1 - lam) * criterion(preds3, targets6)
    return combine_loss(l1, l2, l3, weights)


def mixup_criterion(preds1, preds2, preds3, targets, weights=[0.5, 0.25, 0.25]):
    targets1, targets2, targets3, targets4, targets5, targets6, lam = (
        targets[0],
        targets[1],
        targets[2],
        targets[3],
        targets[4],
        targets[5],
        targets[6],
    )
    criterion = nn.CrossEntropyLoss(reduction="mean")
    l1 = lam * criterion(preds1, targets1) + (1 - lam) * criterion(preds1, targets2)
    l2 = lam * criterion(preds2, targets3) + (1 - lam) * criterion(preds2, targets4)
    l3 = lam * criterion(preds3, targets5) + (1 - lam) * criterion(preds3, targets6)

    return combine_loss(l1, l2, l3, weights)


def lin_comb(a, b, t):
    return t * a + (1 - t) * b


def reduce_loss(loss, reduction="mean"):
    return (
        loss.mean()
        if reduction == "mean"
        else loss.sum()
        if reduction == "sum"
        else loss
    )


def combine_loss(l1, l2, l3, weights):
    return weights[0] * l1 + weights[1] * l2 + weights[2] * l3


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
#
#     def get_old(self):
#         if hasattr(self, 'old_crit'):  return self.old_crit
#         elif hasattr(self, 'old_red'):
#             setattr(self.crit, 'reduction', self.old_red)
#             return self.crit
