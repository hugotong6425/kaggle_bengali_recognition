from torch.utils.data import Dataset, DataLoader


class GraphemeDataset_train(Dataset):
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


def create_dataloaders_train(
    image_data,
    name_data,
    label_data,
    train_idx,
    valid_idx,
    data_transforms,
    batch_size,
    num_workers,
    pin_memory,
):
    print("Creating train dataloader...")
    train_dataset = GraphemeDataset_train(
        image_data[train_idx],
        label_data[train_idx],
        transforms=data_transforms["train"],
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    print("Creating valid dataloader...")
    val_dataset = GraphemeDataset_train(
        image_data[valid_idx], label_data[valid_idx], transforms=data_transforms["val"]
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    data_loaders = {"train": train_loader, "val": val_loader}

    return data_loaders
