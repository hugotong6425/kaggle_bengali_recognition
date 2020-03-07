import hydra
import os
import numpy as np
import wandb

import pickle
from pathlib import Path

import torch
import torch.optim as optim
from torch import nn
from torchvision import transforms
import torchcontrib
from sklearn.utils.class_weight import compute_class_weight
import hydra
from hydra.experimental import compose, initialize

from hw_grapheme.io.load_data import load_processed_data
from hw_grapheme.model_archs.se_resnext import se_resnext50
from hw_grapheme.models.train import train_model
from hw_grapheme.train_utils.create_dataloader import create_dataloaders_train
from hw_grapheme.train_utils.train_test_split import stratified_split_kfold
from omegaconf import DictConfig

# # Init

initialize(
    config_dir="configs", strict=True,
)

EXP_NAME = "regression"
MACHINE = "1080ti"

overrides = [f"exp_name={EXP_NAME}", f"machine={MACHINE}"]

cfg = compose("config.yaml", overrides=overrides)
print(cfg.pretty())

# # Assign Config

# +
if cfg.exp_name =='base': 
    raise ValueError('Give me a proper exp name')
print('EXP NAME', cfg.exp_name)
print('====='*20)
print(cfg.pretty())

if cfg.mix_precision:
    # 1080ti can't use mix precision & Window issue
    from apex import amp

DATA_PATH = Path(cfg.DATA_PATH)
random_seed = cfg.random_seed

# load processed data
pickle_paths = [
    #     DATA_PATH/"sample.pickle",
    DATA_PATH/"train_data_0.pickle",
    #     DATA_PATH/"train_data_1.pickle",
    #     DATA_PATH/"train_data_2.pickle",
    #     DATA_PATH/"train_data_3.pickle",
]

image_data, name_data, label_data = load_processed_data(
    pickle_paths, image_size=128)

# # +
# split train valid set
n_splits = cfg.n_splits
random_speed = cfg.random_seed

train_idx_list, test_idx_list = stratified_split_kfold(
    image_data, label_data, n_splits, random_seed
)

# # +
# default training setting
num_workers = cfg.num_workers
pin_memory = cfg.pin_memory
fold = list(range(n_splits))

# customize training setting
n_epoch = cfg.n_epoch
batch_size = cfg.batch_size
mixed_precision = cfg.mix_precision

model_arch = eval(cfg.model_arch)
model_parameter = cfg.model_parameter
# model_parameter = eval(cfg.model_parameter)
# -

# import image transforms config
rotate = cfg.data_transforms.rotate
scale = cfg.data_transforms.scale
p_affine = cfg.data_transforms.p_affine
data_transforms = {
    'train': transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomApply(
            [transforms.RandomAffine(degrees=rotate, scale=scale)],
            p=p_affine,
        ),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        # transforms.Normalize([0.0692], [0.2051]),
    ]),
    'val': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        # transforms.Normalize([0.0692], [0.2051])
    ]),
}

# +
swa = cfg.swa

optimizer = eval(cfg.optimizer)
print('OPTIM', optimizer, type(optimizer))
optimizer_parameter = cfg.optimizer_parameter
is_weighted_class_loss = cfg.is_weighted_class_loss

# create lr scheduler
epoch_scheduler_func = eval(cfg.epoch_scheduler_func)
epoch_scheduler_func_para = cfg.epoch_scheduler_func_para
error_plateau_scheduler_func = eval(cfg.error_plateau_scheduler_func)
error_plateau_scheduler_func_para = cfg.error_plateau_scheduler_func_para

# prob. of using ["mixup", "cutmix", "cross_entropy"] loss
train_loss_prob = cfg.train_loss_prob
mixup_alpha = cfg.mixup_alpha  # for mixup/cutmix only

# weighting of [root, vowel, consonant]
head_weights = cfg.head_weights

wandb_log = cfg.wandb_log

# save dir, set None to not save, need to manual create folders first
save_dir = cfg.save_dir
Path(save_dir).mkdir(parents=True, exist_ok=True)
# -

if is_weighted_class_loss:
    root_label = label_data[:, 0]
    vowel_label = label_data[:, 1]
    consonant_label = label_data[:, 2]

    class_weight = "balanced"

    root_cls_weight = compute_class_weight(
        class_weight, np.unique(root_label), root_label)
    vowel_cls_weight = compute_class_weight(
        class_weight, np.unique(vowel_label), vowel_label)
    consonant_cls_weight = compute_class_weight(
        class_weight, np.unique(consonant_label), consonant_label)

    class_weights = [
        torch.Tensor(root_cls_weight).cuda(),
        torch.Tensor(vowel_cls_weight).cuda(),
        torch.Tensor(consonant_cls_weight).cuda(),
    ]
else:
    class_weights = None

# # Training

for i, (train_idx, valid_idx) in enumerate(zip(train_idx_list, test_idx_list)):
    # skip unwanted fold
    if i not in [0]:
        continue

    print(f"Training fold {i}")

    # create model
    model = model_arch(**model_parameter)

    # create optimizer
    optimizer_ft = optimizer(model.parameters(), **optimizer_parameter)

    if swa:
        optimizer_ft = torchcontrib.optim.SWA(optimizer_ft)

    if mixed_precision:
        model.to("cuda")
        model, optimizer_ft = amp.initialize(
            model, optimizer_ft, opt_level="O1")
        model = nn.parallel.DataParallel(model)
    else:
        model.to("cuda")
        model = nn.DataParallel(model)

    # create data loader
    data_loaders = create_dataloaders_train(
        image_data, name_data, label_data, train_idx, valid_idx,
        data_transforms, batch_size, num_workers, pin_memory
    )

    # create epoch_scheduler
    if epoch_scheduler_func:
        epoch_scheduler = epoch_scheduler_func(
            optimizer_ft, **epoch_scheduler_func_para)
    else:
        epoch_scheduler = None

    # create error_plateaus_scheduler
    if error_plateau_scheduler_func:
        error_plateau_scheduler = error_plateau_scheduler_func(
            optimizer_ft, **error_plateau_scheduler_func_para)
    else:
        error_plateau_scheduler = None

    # callbacks = {}
    if save_dir:
        full_save_dir = os.path.join(save_dir, f"fold_{i}")
    else:
        full_save_dir = None

    wandb.init(name=cfg.exp_name, project=cfg.project,config=cfg)
    # Training
    train_input_args = {
        "model": model,
        "optimizer": optimizer_ft,
        "dataloaders": data_loaders,
        "mixed_precision": mixed_precision,
        "train_loss_prob": train_loss_prob,
        "class_weights": class_weights,
        "head_weights": head_weights,
        "mixup_alpha": mixup_alpha,
        "num_epochs": n_epoch,
        "epoch_scheduler": epoch_scheduler,
        "error_plateau_scheduler": error_plateau_scheduler,
        "save_dir": full_save_dir,
        "wandb_log": wandb_log,
        "swa": swa,
    }

    callbacks = train_model(**train_input_args)