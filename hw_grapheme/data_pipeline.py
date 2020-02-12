import pickle

import numpy as np

from torch.utils.data import Dataset, DataLoader


def load_data(pickle_paths, image_size=224):
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

    image_data = image_data.reshape(image_data.shape[0]*image_data.shape[1], image_size, image_size)
    name_data = name_data.reshape(-1)
    label_data = label_data.reshape(label_data.shape[0]*label_data.shape[1], 3)

    print(f"Load data done, shape: {image_data.shape}, {name_data.shape}, {label_data.shape}")
    return image_data, name_data, label_data


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


def create_dataloaders(
    image_data, name_data, label_data, train_idx, valid_idx, 
    data_transforms, batch_size, num_workers, pin_memory, 
    create_train_dataloader=True
):
    if create_train_dataloader:
        print("Creating train dataloader...")
        train_dataset = GraphemeDataset(
            image_data[train_idx], label_data[train_idx], transforms=data_transforms["train"]
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size,
            num_workers=num_workers, pin_memory=pin_memory,
        )
    else:
        train_loader = None

    print("Creating test dataloader...")
    val_dataset = GraphemeDataset(
        image_data[valid_idx], label_data[valid_idx], transforms=data_transforms["val"]
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size*2,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    data_loaders = {
        "train": train_loader,
        "val": val_loader,
    }

    return data_loaders