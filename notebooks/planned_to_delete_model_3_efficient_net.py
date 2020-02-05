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

# +
import copy
import json
import os
import pickle
import time
import torch
import torchvision

import apex
from apex import amp

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim

from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import MemoryEfficientSwish

from tqdm import tqdm_notebook

from torchtools.optim import RangerLars

plt.ion()   # interactive mode


# -

class GraphemeDataset(Dataset):
    def __init__(self, image, label, transforms=None):
        self.image = image
        self.label = label
        self.transforms = transforms
        
    def __len__(self):
        return self.image.shape[0]
    
    def __getitem__(self, idx):
        data = self.image[idx]
        if self.transforms is not None:
            data = self.transforms(data)
        root, vowel, consonant = self.label[idx]
        return data, root, vowel, consonant


def load_data(pickle_paths):
    # load data from pickle
    image_data = []
    name_data = []
    label_data = []

    for pickle_path in pickle_paths:
        with open(pickle_path, "rb") as f:
            train_data = pickle.load(f)
            image_data.append(train_data[0])
            name_data.append(train_data[1])
            label_data.append(train_data[2])

    image_data = np.array(image_data)
    name_data = np.array(name_data)
    label_data = np.array(label_data)

    # print(image_data.shape, name_data.shape, label_data.shape)

    image_data = image_data.reshape(image_data.shape[0]*image_data.shape[1], 224, 224)
    name_data = name_data.reshape(-1)
    label_data = label_data.reshape(label_data.shape[0]*label_data.shape[1], 3)

    print(f"Load data done, shape: {image_data.shape}, {name_data.shape}, {label_data.shape}")
    return image_data, name_data, label_data


def train_test_split(num_train, valid_size, random_seed):
    # random train test split
    indices = list(range(num_train))

    np.random.seed(random_seed)
    np.random.shuffle(indices)

    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]
    
    return train_idx, valid_idx


# +
class Head(nn.Module):
    def __init__(self, in_feature, n_class, ps=0.5):
        super().__init__()
        self._fc = nn.Linear(in_feature, n_class)
        self._swish = MemoryEfficientSwish()
        
    def forward(self, x):
        x = self._fc(x)
        x = self._swish(x)
        return x

    
class Grapheme_network(nn.Module):
    def __init__(self, head, backbone_out_feature):
        super().__init__()
        
#         eff_b0_backbone = EfficientNet.from_pretrained('efficientnet-b0') 
#         eff_b0_backbone = nn.Sequential(*list(eff_b0_backbone.children())[:-2])
#         self.backbone = eff_b0_backbone
        
        eff_b0_backbone = EfficientNet.from_pretrained('efficientnet-b0') 
        self.backbone = nn.Sequential(
            eff_b0_backbone._conv_stem,
            eff_b0_backbone._bn0,
            eff_b0_backbone._blocks,
            eff_b0_backbone._conv_head,
            eff_b0_backbone._bn1,
            eff_b0_backbone._avg_pooling,
            eff_b0_backbone._dropout,
        )
        
        self.head_root = head(backbone_out_feature, 168)
        self.head_vowel = head(backbone_out_feature, 11)
        self.head_consonant = head(backbone_out_feature, 7)
        
    def forward(self, x):    
        x = self.backbone(x)
        
        x1 = self.head_root(x)
        x2 = self.head_vowel(x)
        x3 = self.head_consonant(x)
        
        return x1,x2,x3


# -

def train_phrase(model, optimizer, train_dataloader, criterion, num_train, mixed_precision):

    model.train()  # Set model to training mode

    running_loss = 0.0
    running_corrects = 0

    # Iterate over data.
    for image, root, vowel, consonant in tqdm_notebook(train_dataloader):
        image = image.to("cuda")
        root = root.long().to("cuda")

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        # track history if only in train
        with torch.set_grad_enabled(True):
            outputs = model(image)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, root)

        # backward + optimize
        if mixed_precision:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

        # statistics
        running_loss += loss.item() * image.size(0)
        running_corrects += torch.sum(preds == root.data)

    train_loss = running_loss / float(num_train)
    train_acc = running_corrects.double() / num_train
    
    return train_loss, train_acc


# +
def validate_phrase(model, valid_dataloader, criterion, num_val):
    # Each epoch has a training and validation phase
    model.eval()   # Set model to evaluate mode

    running_loss = 0.0
    running_corrects = 0

    # Iterate over data.
    for image, root, vowel, consonant in tqdm_notebook(valid_dataloader):
        image = image.to("cuda")
        root = root.long().to("cuda")

        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):
            outputs = model(image)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, root)

        # statistics
        running_loss += loss.item() * image.size(0)
        running_corrects += torch.sum(preds == root.data)

    val_loss = running_loss / float(num_val)
    val_acc = running_corrects.double() / num_val

    return val_loss, val_acc


def train_model(
    model, criterion, optimizer, dataloaders, 
    mixed_precision, callbacks, num_epochs=25, scheduler=None
):
    callbacks["train_loss_list"] = []
    callbacks["train_acc_list"] = []
    callbacks["val_loss_list"] = []
    callbacks["val_acc_list"] = []
    
    since = time.time()
    
    num_train = len(dataloaders["train"].dataset)
    num_val = len(dataloaders["val"].dataset)
    
    #high_acc_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    #low_loss_model_wts = copy.deepcopy(model.state_dict())
    lowest_loss = 999

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        train_loss, train_acc = train_phrase(
            model, optimizer, dataloaders["train"], criterion, num_train, mixed_precision
        )
        print("Train Loss: {:.4f} Acc: {:.4f}".format(train_loss, train_acc))
        
        val_loss, val_acc = validate_phrase(
            model, dataloaders["val"], criterion, num_val
        )
        print("Val Loss: {:.4f} Acc: {:.4f}".format(val_loss, val_acc))
        
        if scheduler:
            scheduler.step(val_loss)
        
        # deep copy the model
        if val_acc > best_acc:
            print(f"In epoch {epoch}, highest val accuracy increases from {best_acc} to {val_acc}.")
            best_acc = val_acc
            # high_acc_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), "./eff_0_high_acc.pth")

        # deep copy the model
        if val_loss < lowest_loss:
            print(f"In epoch {epoch}, lowest val loss decreases from {lowest_loss} to {val_loss}.")
            lowest_loss = val_loss
            # low_loss_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), "./eff_0_low_loss.pth")
        
        callbacks["train_loss_list"].append(train_loss)
        callbacks["train_acc_list"].append(train_acc)
        callbacks["val_loss_list"].append(val_loss)
        callbacks["val_acc_list"].append(val_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    
    return callbacks

    ## load best model weights
    #model.load_state_dict(best_model_wts)
    # return best_model_wts


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
    "../data/processed_data/size_224/train_data_1.pickle",
    "../data/processed_data/size_224/train_data_2.pickle",
    "../data/processed_data/size_224/train_data_3.pickle",
]

image_data, name_data, label_data = load_data(pickle_paths)

# +
# train test split
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

valid_size = 0.2
random_seed = 2020
num_train = len(label_data)

train_idx, valid_idx = train_test_split(
    num_train, valid_size, random_seed
)

train_dataset = GraphemeDataset(
    image_data[train_idx], label_data[train_idx], transforms=data_transforms["train"]
)
val_dataset = GraphemeDataset(
    image_data[valid_idx], label_data[valid_idx], transforms=data_transforms["val"]
)

dataset_sizes = {
    "train": len(train_dataset),
    "val": len(val_dataset),
}

batch_size = 64
num_workers = 2

train_loader = DataLoader(
    train_dataset, batch_size=batch_size,
    num_workers=num_workers, pin_memory=True,
)

val_loader = DataLoader(
    val_dataset, batch_size=batch_size,
    num_workers=num_workers, pin_memory=True,
)

data_loaders = {
    "train": train_loader,
    "val": val_loader,
}


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

# +
eff_b0 = EfficientNet.from_pretrained('efficientnet-b0', num_classes=168)
eff_b0.to("cuda")
eff_b0 = nn.DataParallel(eff_b0)

criterion = nn.CrossEntropyLoss()

# my_list = ['module._fc.weight', 'module._fc.bias']
# params = list(filter(lambda kv: kv[0] in my_list, eff_b0.named_parameters()))
# base_params = list(filter(lambda kv: kv[0] not in my_list, eff_b0.named_parameters()))

# params = [kv[1] for kv in params]
# base_params = [kv[1] for kv in base_params]

# optimizer_ft = optim.Adam(eff_b0.parameters())
optimizer_ft = RangerLars(eff_b0.parameters())


# mixed_precision = False
# if mixed_precision:
#     #eff_b0 = apex.parallel.DistributedDataParallel(eff_b0)
#     eff_b0.to("cuda")
# #     eff_b0 = torch.nn.parallel.DistributedDataParallel(
# #         eff_b0, device_ids=[0,1], output_device=0
# #     )
# #     #
#     # eff_b0, optimizer_ft = amp.initialize(eff_b0, optimizer_ft, opt_level="O1")
    
# else:
#     eff_b0 = nn.DataParallel(eff_b0)
#     # eff_b0.to("cuda")
    
    

# exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(
#     optimizer_ft, factor=0.5, patience=5,
# )

# +
callbacks = {}

callbacks = train_model(
    eff_b0, criterion, optimizer_ft, data_loaders,
    False, callbacks, num_epochs=100
)
# -

callbacks

import pandas as pd

plt.plot(callbacks["train_loss_list"][1:])
plt.plot(callbacks["val_loss_list"][1:])
# plt.plot(callbacks["train_loss_list"])
# plt.plot(callbacks["train_loss_list"])

pd.DataFrame(callbacks)

callbacks["train_acc_list"][0].cpu().numpy()



eff_b0_backbone = EfficientNet.from_pretrained('efficientnet-b0') 
eff_b0_backbone

# modelling
eff_b0_backbone = EfficientNet.from_pretrained('efficientnet-b0') 
eff_b0_backbone = nn.Sequential(*list(eff_b0_backbone.children())[:-2])
eff_b0_backbone

eff_b0_backbone(torch.ones(1,3, 64,64))

eff_b0_backbone = EfficientNet.from_pretrained('efficientnet-b0') 
backbone = nn.Sequential(
    eff_b0_backbone._conv_stem,
    eff_b0_backbone._bn0,
    eff_b0_backbone._blocks,
    eff_b0_backbone._conv_head,
    eff_b0_backbone._bn1,
    eff_b0_backbone._avg_pooling,
    eff_b0_backbone._dropout,
)

backbone(torch.ones(1,3, 64,64))



# +
backbone_out_feature = 1280

eff_b0 = Grapheme_network(Head, backbone_out_feature)
# -

eff_b0.train()

# +
sample = next(iter(data_loaders["train"]))


# -

eff_b0(sample[0])

eff_b0_backbone

eff_b0_backbone(sample[0])

# +

eff_b0(sample[0])
# -

eff_b0_backbone(sample[0])


