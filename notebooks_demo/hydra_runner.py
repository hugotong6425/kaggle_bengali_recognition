import hydra
import os
import numpy as np
import wandb



import torch
import torch.optim as optim
from torch import nn
from torchvision import transforms
import torchcontrib
from sklearn.utils.class_weight import compute_class_weight

from hw_grapheme.io.load_data import load_processed_data
from hw_grapheme.model_archs.se_resnext import se_resnext50
from hw_grapheme.models.train import train_model
from hw_grapheme.train_utils.create_dataloader import create_dataloaders_train
from hw_grapheme.train_utils.train_test_split import stratified_split_kfold
from omegaconf import DictConfig
@hydra.main(config_path="configs/config.yaml")
def experiment_runner(cfg: DictConfig) -> None:
    print(cfg.pretty())
    print(cfg.keys())
    print(cfg['gpu_device'])

    import os
    print(os.getcwd())
    if cfg.mix_precision:
        # 1080ti can't use mix precision & Window issue
        from apex import amp

    

    # # load processed data
    # pickle_paths = [
    # "../data/processed/size_128/train_data_0.pickle",
    # #     "../data/processed/size_128/train_data_1.pickle",
    # #     "../data/processed/size_128/train_data_2.pickle",
    # #     "../data/processed/size_128/train_data_3.pickle",
    # ]

    # image_data, name_data, label_data = load_processed_data(
    # pickle_paths, image_size=128)

    # # split train valid set
    # n_splits = 5
    # random_seed = 2020

    # train_idx_list, test_idx_list = stratified_split_kfold(
    # image_data, label_data, n_splits, random_seed
    # )

    # # create data_transforms
    # data_transforms = {
    # 'train': transforms.Compose([
    #     transforms.ToPILImage(),
    #     # transforms.RandomAffine(degrees=10, scale=(1.0, 1.15)),
    #     transforms.Grayscale(num_output_channels=3),
    #     transforms.ToTensor(),
    #     # transforms.Normalize([0.0692], [0.2051]),
    #     # transforms.ToPILImage(),
    # ]),
    # 'val': transforms.Compose([
    #     transforms.ToPILImage(),
    #     transforms.Grayscale(num_output_channels=3),
    #     transforms.ToTensor(),
    #     # transforms.Normalize([0.0692], [0.2051])
    # ]),
    # }
if __name__ == "__main__":
    experiment_runner()
