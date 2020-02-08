import numpy as np
import torch.optim as optim
import torch

from efficientnet_pytorch import EfficientNet
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.optim import lr_scheduler

from hw_grapheme.train import generate_stratified_k_fold_index, train_model
from hw_grapheme.utils import load_model_weight
from hw_grapheme.data_pipeline import create_dataloaders, load_data
from hw_grapheme.model import EfficientNet_0
from hw_grapheme.loss_func import Loss_combine

from torchtools.optim import RangerLars, RAdam
from torch.optim import Optimizer

import argparse
from apex import amp
from apex.parallel import DistributedDataParallel


def test_apex(mixed_precision, cuda_parallel, batch_size, opt_level):
    # not support in nb
    if mixed_precision and cuda_parallel:
        parser = argparse.ArgumentParser()
        parser.add_argument("--local_rank", type=int)
        args = parser.parse_args()
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method='env://')

    # load data
    pickle_paths = [
        "../data/processed_data/size_224/train_data_0.pickle",
        #     "../data/processed_data/size_224/train_data_1.pickle",
        #     "../data/processed_data/size_224/train_data_2.pickle",
        #     "../data/processed_data/size_224/train_data_3.pickle",
    ]

    image_data, name_data, label_data = load_data(pickle_paths)

    batch_size = batch_size
    num_workers = 6

    pin_memory = True
    n_epoch = 120

    n_splits = 5
    random_seed = 2020

    train_idx_list, valid_idx_list = generate_stratified_k_fold_index(
        image_data, label_data, n_splits, random_seed
    )

    train_idx = train_idx_list[0]
    valid_idx = valid_idx_list[0]

    # create loss function
    criterion = Loss_combine()

    # create data_transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ]),
        'val': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ]),
    }

    # create model
    eff_b0 = EfficientNet_0()

    # create optimizer
    optimizer_ft = optim.Adam(eff_b0.parameters())

    # create data loader
    data_loaders = create_dataloaders(
        image_data, name_data, label_data, train_idx, valid_idx,
        data_transforms, batch_size, num_workers, pin_memory
    )

    if mixed_precision and cuda_parallel:
        eff_b0.to("cuda")
        eff_b0, optimizer_ft = amp.initialize(eff_b0, optimizer_ft, opt_level=opt_level)
        eff_b0 = DistributedDataParallel(eff_b0)

    elif mixed_precision and not cuda_parallel:
        eff_b0.to("cuda")
        eff_b0, optimizer_ft = amp.initialize(eff_b0, optimizer_ft, opt_level=opt_level)
    elif not mixed_precision and cuda_parallel:
        eff_b0.to("cuda")
        eff_b0 = nn.DataParallel(eff_b0)
    elif not mixed_precision and not cuda_parallel:
        eff_b0.to("cuda")

    callbacks = {}

    callbacks = train_model(
        eff_b0, criterion, optimizer_ft, data_loaders,
        mixed_precision, callbacks, num_epochs=n_epoch,
        epoch_scheduler=None, save_dir=None
    )


batch_size = 64
opt_level = "O1"
mixed_precision = True
cuda_parallel = True

test_apex(mixed_precision, cuda_parallel, batch_size, opt_level)
