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


def ohem_loss(cls_pred, cls_target, ohem_rate):
    batch_size = cls_pred.size(0)
    ohem_cls_loss = F.cross_entropy(
        cls_pred, cls_target, reduction="none", ignore_index=-1
    )

    sorted_ohem_loss, idx = torch.sort(ohem_cls_loss, descending=True)
    keep_num = min(sorted_ohem_loss.size()[0], int(batch_size * ohem_rate))
    if keep_num < sorted_ohem_loss.size()[0]:
        keep_idx_cuda = idx[:keep_num]
        ohem_cls_loss = ohem_cls_loss[keep_idx_cuda]
    cls_loss = ohem_cls_loss.sum() / keep_num
    return cls_loss


# def ohem_criterion(preds1, preds2, preds3, targets, ohem_rate, head_weights=[0.5, 0.25, 0.25]):
#     targets1, targets2, targets3 = targets[0], targets[1], targets[2]

#     l1 = ohem_loss(preds1, targets1, ohem_rate)
#     l2 = ohem_loss(preds2, targets2, ohem_rate)
#     l3 = ohem_loss(preds3, targets3, ohem_rate)

#     return combine_loss(l1, l2, l3, head_weights)


def no_extra_augmentation_criterion(
    preds1,
    preds2,
    preds3,
    targets,
    loss_criteria,
    loss_criteria_paras,
    head_weights=[0.5, 0.25, 0.25],
):
    targets1, targets2, targets3 = targets[0], targets[1], targets[2]

    root_arg = loss_criteria_paras["root"]
    vowel_arg = loss_criteria_paras["vowel"]
    consonant_arg = loss_criteria_paras["consonant"]

    l1 = loss_criteria(preds1, targets1, **root_arg)
    l2 = loss_criteria(preds2, targets2, **vowel_arg)
    l3 = loss_criteria(preds3, targets3, **consonant_arg)

    return combine_loss(l1, l2, l3, head_weights).mean()


def cutmix_criterion(
    preds1,
    preds2,
    preds3,
    targets,
    loss_criteria,
    loss_criteria_paras,
    head_weights=[0.5, 0.25, 0.25],
):
    targets1, shuffle_targets1, targets2, shuffle_targets2, targets3, shuffle_targets3, lam = (
        targets[0],
        targets[1],
        targets[2],
        targets[3],
        targets[4],
        targets[5],
        targets[6],
    )

    # root_criterion = loss_criteria()
    # vowel_criterion = loss_criteria(**loss_criteria_paras["vowel"])
    # consonant_criterion = loss_criteria(**loss_criteria_paras["consonant"])

    root_arg = loss_criteria_paras["root"]
    vowel_arg = loss_criteria_paras["vowel"]
    consonant_arg = loss_criteria_paras["consonant"]

    root_unshuffle = lam * loss_criteria(preds1, targets1, **root_arg)     
    vowel_unshuffle = lam * loss_criteria(preds2, targets2, **vowel_arg)
    consonant_unshuffle = lam * loss_criteria(preds3, targets3, **consonant_arg)
    
    root_shuffle = (1-lam) * loss_criteria(preds1, shuffle_targets1, **root_arg)
    vowel_shuffle = (1-lam) * loss_criteria(preds2, shuffle_targets2, **vowel_arg)
    consonant_shuffle = (1-lam) * loss_criteria(preds3, shuffle_targets3, **consonant_arg)
    
    l1 = root_unshuffle + root_shuffle
    l2 = vowel_unshuffle + vowel_shuffle
    l3 = consonant_unshuffle + consonant_shuffle
       
    return combine_loss(l1, l2, l3, head_weights).mean()


def mixup_criterion(
    preds1,
    preds2,
    preds3,
    targets,
    loss_criteria,
    loss_criteria_paras,
    head_weights=[0.5, 0.25, 0.25],
):
    targets1, shuffle_targets1, targets2, shuffle_targets2, targets3, shuffle_targets3, lam = (
        targets[0],
        targets[1],
        targets[2],
        targets[3],
        targets[4],
        targets[5],
        targets[6],
    )

    root_arg = loss_criteria_paras["root"]
    vowel_arg = loss_criteria_paras["vowel"]
    consonant_arg = loss_criteria_paras["consonant"]

    root_unshuffle = lam * loss_criteria(preds1, targets1, **root_arg)     
    vowel_unshuffle = lam * loss_criteria(preds2, targets2, **vowel_arg)
    consonant_unshuffle = lam * loss_criteria(preds3, targets3, **consonant_arg)
    
    root_shuffle = (1-lam) * loss_criteria(preds1, shuffle_targets1, **root_arg)
    vowel_shuffle = (1-lam) * loss_criteria(preds2, shuffle_targets2, **vowel_arg)
    consonant_shuffle = (1-lam) * loss_criteria(preds3, shuffle_targets3, **consonant_arg)
    
    l1 = root_unshuffle + root_shuffle
    l2 = vowel_unshuffle + vowel_shuffle
    l3 = consonant_unshuffle + consonant_shuffle

    return combine_loss(l1, l2, l3, head_weights).mean()


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

# class Weighted_entropy(nn.Module):
#     def __init__(self, weights=[0.5, 0.25, 0.25]):
#         super().__init__()
#         self.weights = weights

#     def forward(
#         self, input, root_target, vowel_target, consonant_target, reduction="mean"
#     ):
#         root_pred, vowel_pred, consonant_pred = input
#         root_pred = root_pred.float()
#         vowel_pred = vowel_pred.float()
#         consonant_pred = consonant_pred.float()

#         l1 = F.cross_entropy(root_pred, root_target, reduction=reduction)
#         l2 = F.cross_entropy(vowel_pred, vowel_target, reduction=reduction)
#         l3 = F.cross_entropy(consonant_pred, consonant_target, reduction=reduction)
#         return combine_loss(l1, l2, l3, self.weights)
