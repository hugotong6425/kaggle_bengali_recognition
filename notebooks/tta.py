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

# +
import torch

import numpy as np
import pandas as pd

from torch import nn
from torchvision import models
from torch.utils.data import Dataset

from fastai.layers import AdaptiveConcatPool2d, Flatten, bn_drop_lin
from hw_grapheme.dl_utils.mish_activation import Mish

# +
TEST = [
#      '../data/train_image_data_0.parquet',
#      '../data/train_image_data_2.parquet',
    'test_image_data_0.parquet',
    'test_image_data_1.parquet',
    'test_image_data_2.parquet',
    'test_image_data_3.parquet'
]
model_names = ["fold_0/eff_0_high_acc.pth"]
# model_dir = "/kaggle/input/bengalicv19trainedmodels"
model_dir = "../model_weights/eff_0_with_mixup_cutmix/"
# data_dir = "/kaggle/input/bengaliai-cv19"
data_dir = "/kaggle/input/bengaliai-cv19"


bs = 128

row_id,target = [],[]

for fname in TEST:
    # get dataloader
    data_path = os.path.join(data_dir, fname)
    ds = GraphemeDataset(data_path)
    dl = DataLoader(ds, batch_size=bs, num_workers=nworkers, shuffle=False)
    
    model_results_root = []
    model_results_vowel = []
    model_results_consonant = []
    
    for i, model_name in enumerate(model_names):
        # load model
        print(f"Loading model {i}...")
        model = Dnet_1ch(pre=False).cuda()
        model_path = os.path.join(model_dir, model_name)
        load_parallel_trained_model(model, model_path)
        
        # get full dataset, multi-model pred
        probit_root, probit_vowel, probit_consonant, file_name = predit_in_mini_batch(model, dl)
        model_results_root.append(probit_root)
        model_results_vowel.append(probit_vowel)
        model_results_consonant.append(probit_consonant)
        
    # get final pred = highest summed probit 
    probit_root_sum = model_results_root[0].copy()
    for r in model_results_root[1:]:
        probit_root_sum += r
        
    probit_vowel_sum = model_results_vowel[0].copy()
    for r in model_results_vowel[1:]:
        probit_vowel_sum += r
        
    probit_consonant_sum = model_results_consonant[0].copy()
    for r in model_results_consonant[1:]:
        probit_consonant_sum += r
    
    pred_ensemble_root = probit_root_sum.argmax(axis=1)
    pred_ensemble_vowel = probit_vowel_sum.argmax(axis=1)
    pred_ensemble_consonant = probit_consonant_sum.argmax(axis=1)
    
    # turn pred into df
    file_name = [x for batch_x in file_name for x in batch_x]
    for idx, name in enumerate(file_name):
        row_id += [f'{name}_grapheme_root',f'{name}_vowel_diacritic',
                   f'{name}_consonant_diacritic']
        target += [pred_ensemble_root[idx],pred_ensemble_vowel[idx],pred_ensemble_consonant[idx]]
