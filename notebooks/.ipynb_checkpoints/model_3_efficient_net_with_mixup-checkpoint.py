# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
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

from efficientnet_pytorch import EfficientNet
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.optim import lr_scheduler

from hw_grapheme.train_mixup import generate_stratified_k_fold_index, train_model
from hw_grapheme.utils import load_model_weight
from hw_grapheme.data_pipeline import create_dataloaders, load_data
from hw_grapheme.model import EfficientNet_grapheme, EfficientNet_0
from hw_grapheme.loss_func import Loss_combine

from torchtools.optim import RangerLars, RAdam
# from one_cycle import OneCycleLR
from torch.optim import Optimizer

# from warmup_scheduler import GradualWarmupScheduler


# -

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# +
# load data 
pickle_paths = [
    "../data/processed_data/size_224/train_data_0.pickle",
#     "../data/processed_data/size_224/train_data_1.pickle",
#     "../data/processed_data/size_224/train_data_2.pickle",
#     "../data/processed_data/size_224/train_data_3.pickle",
]

image_data, name_data, label_data = load_data(pickle_paths)
# -

image_data[-5][100:110, 100:110]

# +
batch_size = 64
num_workers = 6
pin_memory = True
n_epoch = 120

n_splits = 5
random_seed = 2020

mixed_precision = False

train_idx_list, test_idx_list = generate_stratified_k_fold_index(
    image_data, label_data, n_splits, random_seed
)

# create loss function
# criterion = nn.CrossEntropyLoss()
criterion = Loss_combine()

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
        # transforms.RandomAffine(degrees=10, scale=(1.0, 1.15)),
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

# pretrain = False
# eff_version = "4"
#
# eff_b0 = EfficientNet_grapheme(eff_version, pretrain)

# eff_b0







# +
pretrain = False
# eff_version = "4"
mixup_alpha = 0.1

for i, (train_idx, valid_idx) in enumerate(zip(train_idx_list, test_idx_list)):
    if i != 0:
        continue
    print(f"Training fold {i}")
    
    # create model 
    # eff_b0 = EfficientNet_grapheme(eff_version, pretrain)    
    eff_b0 = EfficientNet_0(pretrain)

    #########################
    # load_model_weight(eff_b0, "../model_weights/eff_0_with_mixup_cutmix/fold_0/eff_0_low_loss.pth")
    #########################
    if mixed_precision:
        eff_b0 = apex.parallel.DistributedDataParallel(eff_b0)
        eff_b0.to("cuda")
        eff_b0 = torch.nn.parallel.DistributedDataParallel(
            eff_b0, device_ids=[0,1], output_device=0
        )
        eff_b0, optimizer_ft = amp.initialize(eff_b0, optimizer_ft, opt_level="O1")
    else:
        eff_b0.to("cuda")
        eff_b0 = nn.DataParallel(eff_b0)
        
    # create optimizer
    # optimizer_ft = RangerLars(eff_b0.parameters())
    optimizer_ft = optim.Adam(eff_b0.parameters(), weight_decay=1e-5)

    # create data loader
    data_loaders = create_dataloaders(
        image_data, name_data, label_data, train_idx, valid_idx, 
        data_transforms, batch_size, num_workers, pin_memory
    )
    
    # create lr scheduler
    # exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(
    #     optimizer_ft, factor=0.5, patience=5,
    # )
#     cos_lr_scheduler = lr_scheduler.CosineAnnealingLR(
#         optimizer_ft, T_max=n_epoch,
#     )
#     # one_cycle_lr_scheduler = OneCycleLR(
#     #     optimizer_ft, max_lr=0.01, steps_per_epoch=len(data_loaders["train"]), epochs=n_epoch
#     # )   
    
#     scheduler_warmup = GradualWarmupScheduler(
#         optimizer_ft, multiplier=1, total_epoch=10, after_scheduler=cos_lr_scheduler
#     )

    
    callbacks = {}

    callbacks = train_model(
        eff_b0, optimizer_ft, data_loaders,
        mixed_precision, callbacks, mixup_alpha, num_epochs=n_epoch,
        epoch_scheduler=None, save_dir=f"../model_weights/eff_0_with_mixup_cutmix_alpha_0.1/fold_{i}"
    )

# -

configs = {
    "model": "efficient 0",
    "pretrain": pretrain,
    "head_info": "1 fc",
    "input_size": "224X224",
    "optimizer": "adam",
    "n_fold": n_splits,
    "split_seed": random_seed,
    "batch_size": batch_size,
    "epoch": n_epoch,
    "mixed_precision": mixed_precision,
    "image_processing": "mixup, cutmix",
    "cutmix/mixup alpha": 0.1,
}

# +
save_root_dir = "../model_weights/eff_0_baseline"


config_save_path = os.path.join(save_root_dir, "config.csv")

with open(config_save_path, "w") as f:
    for key in configs.keys():
        f.write(f"{key},{configs[key]}\n")


# +
# Get a batch of training data for demo

# visual_loader = DataLoader(
#     train_dataset, batch_size=4,
#     num_workers=num_workers, pin_memory=True,
# )

# inputs, a,b,c = next(iter(visual_loader))

# # Make a grid from batch
# out = torchvision.utils.make_grid(inputs)

# imshow(out)
# -




