# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
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

# %load_ext autoreload
# %autoreload 2

# +
import csv
import os

import numpy as np
import torch.optim as optim

from pathlib import Path
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.optim import lr_scheduler

from hw_grapheme.train import generate_stratified_k_fold_index, train_model
from hw_grapheme.utils import load_model_weight
from hw_grapheme.data_pipeline import create_dataloaders, load_data
from hw_grapheme.model.se_resnext50 import se_resnext50 
from hw_grapheme.loss_func import Loss_combine

# from torchtools.optim import RangerLars, RAdam
# from one_cycle import OneCycleLR
from torch.optim import Optimizer

# from warmup_scheduler import GradualWarmupScheduler


# +
# load data 
pickle_paths = [
    "../data/processed_data/size_224/train_data_0.pickle",
    "../data/processed_data/size_224/train_data_1.pickle",
    "../data/processed_data/size_224/train_data_2.pickle",
    "../data/processed_data/size_224/train_data_3.pickle",
]

image_data, name_data, label_data = load_data(pickle_paths)
# -

configs = {
    "name": 'se-resnext50 baseline',
    "model": "se-resnext50",
    "pretrain": False,
    "head_info": "1 fc",
    "input_size": "224X224",
    "optimizer": "adam",
    "image_processing": "rotate(-10,10), scale(1.0, 1.15)",
    'batch_size': 64,
    'num_workers': 8,
    'pin_memory': True,
    'n_epoch': 48,
    'n_splits': 5,
    'random_seed': 2020,
    'mix_precision': False
}

# +
batch_size = configs['batch_size']
num_workers = configs['num_workers'] 
pin_memory = configs['pin_memory']
n_epoch = configs['n_epoch']
n_splits = configs['n_splits']
random_seed = configs['random_seed']
mixed_precision = configs['mix_precision']

train_idx_list, test_idx_list = generate_stratified_k_fold_index(
    image_data, label_data, n_splits, random_seed
)

# create loss function
# criterion = nn.CrossEntropyLoss()
# criterion = Loss_combine()

# for discriminative lr
# my_list = ['module._fc.weight', 'module._fc.bias']
# params = list(filter(lambda kv: kv[0] in my_list, eff_b0.named_parameters()))
# base_params = list(filter(lambda kv: kv[0] not in my_list, eff_b0.named_parameters()))
# params = [kv[1] for kv in params]
# base_params = [kv[1] for kv in base_params]

# create data_transforms
data_transforms = {
    'train': transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomAffine(degrees=10, scale=(1.0, 1.15)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        # transforms.Normalize([0.0692], [0.2051]),
        # transforms.ToPILImage(),
    ]),
    'val': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        # transforms.Normalize([0.0692], [0.2051])
    ]),
}
# -

from datetime import datetime
now = datetime.now()
current_time = now.strftime("%Y%m%d_%H%M%S")
current_time

# + run_control={"marked": true}
# import os
# import wandb
# # os.environ['WANDB_NOTEBOOK_NAME'] = 'model_3_efficient_net'
# # %env WANDB_NOTEBOOK_NAME=model_3_efficient_net

# +
pretrain = configs['pretrain']
criterion = Loss_combine()

for i, (train_idx, valid_idx) in enumerate(zip(train_idx_list, test_idx_list)):
    if i != 0:
        continue
        
    print(f"Training fold {i}")
    MODEL_DIR = Path(f"../model_weights/seresnext50_baseline/fold_{i}")
    MODEL_DIR.mkdir(exist_ok=True)
    
    # create model
    model = se_resnext50()

    if mixed_precision:
        model = apex.parallel.DistributedDataParallel(model)
        model.to("cuda")
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[0, 1], output_device=0
        )
        model, optimizer_ft = amp.initialize(
            model, optimizer_ft, opt_level="O1"
        )
    else:
        model.to("cuda")
        model = nn.DataParallel(model)
        # Add W&B logging

    # create optimizer
    optimizer_ft = optim.Adam(model.parameters(), weight_decay=1e-5)

    # create data loader
    data_loaders = create_dataloaders(
        image_data, name_data, label_data, train_idx, valid_idx,
        data_transforms, batch_size, num_workers, pin_memory
    )
    
    callbacks = {}
    callbacks = train_model(
        model, criterion, optimizer_ft, data_loaders,
        mixed_precision, callbacks, num_epochs=n_epoch,
        epoch_scheduler=None, save_dir=MODEL_DIR
    )

# +
save_root_dir = Path("../model_weights/eff_0_baseline")
save_root_dir.mkdir(exist_ok=True)


config_save_path = save_root_dir/"config.csv"

with open(config_save_path, "w") as f:
    for key in configs.keys():
        f.write(f"{key},{configs[key]}\n")
# -



