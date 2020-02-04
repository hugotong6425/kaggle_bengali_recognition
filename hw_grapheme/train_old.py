import time
import torch
import os

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm_notebook

import torch.nn.functional as F


class Loss_combine(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, input, target_root, target_vowel, target_consonant, reduction='mean'):
        root_pred, vowel_pred, consonant_pred = input
        root_pred = root_pred.float()
        vowel_pred = vowel_pred.float()
        consonant_pred = consonant_pred.float()
        
        root_target = root_target.long()
        vowel_target = vowel_target.long()
        consonant_target = consonant_target.long()
        return (
            0.7*F.cross_entropy(root_pred, root_target, reduction=reduction) + 
            0.1*F.cross_entropy(vowel_pred, vowel_target, reduction=reduction) + 
            0.2*F.cross_entropy(consonant_pred, consonant_target, reduction=reduction)
        )


def train_test_split(num_train, valid_size, random_seed):
    # random train test split
    indices = list(range(num_train))

    np.random.seed(random_seed)
    np.random.shuffle(indices)

    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]
    
    return train_idx, valid_idx


def generate_stratified_k_fold_index(image_data, label_data, n_splits, random_seed):
    str_label = []

    for root, vowel, consonant in label_data:
        if root < 10:
            root = "00" + str(root)
        elif root < 100:
            root = "0" + str(root)
        else:
            root = str(root)

        if vowel < 10:
            vowel = "0" + str(vowel)
        else:
            vowel = str(vowel)

        str_label.append(root + vowel + str(consonant))

    skf = StratifiedKFold(n_splits=n_splits, random_state=random_seed, shuffle=True)
    skf.get_n_splits(image_data, label_data)
    print(skf)
    
    train_idx_list = []
    test_idx_list = []
    
    for train_index, test_index in skf.split(image_data, str_label):
        train_idx_list.append(train_index)
        test_idx_list.append(test_index)

    
    return train_idx_list, test_idx_list


def train_phrase(
    model, optimizer, train_dataloader, criterion, 
    num_train, mixed_precision, batch_scheduler=None
):
    model.train()  # Set model to training mode

    running_loss = 0.0
    running_corrects = 0

    # Iterate over data.
    for image, root, vowel, consonant in tqdm_notebook(train_dataloader):
        image = image.to("cuda")
        
        # root output only
        # root = root.long().to("cuda")

        # all root vowel consonant outputs
        root = root.long().to("cuda")
        vowel = root.long().to("cuda")
        consonant = root.long().to("cuda")

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward with root output only
        # track history if only in train
        # with torch.set_grad_enabled(True):
        #     outputs = model(image)
        #     print("outputs: ", outputs.shape)
        #     _, preds = torch.max(outputs, 1)
        #     loss = criterion(outputs, root)

        # forward with all root vowel consonant outputs
        with torch.set_grad_enabled(True):
            root_pred, vowel_pred, consonant_pred = model(image)
            _, preds = torch.max(outputs, 1)
            _, preds = torch.max(outputs, 1)
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

        if batch_scheduler:
            batch_scheduler.step()

    train_loss = running_loss / float(num_train)
    train_acc = running_corrects.double() / num_train
    
    return train_loss, train_acc


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
    mixed_precision, callbacks, num_epochs=25, 
    epoch_scheduler=None, batch_scheduler=None,
    save_dir=None
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
            model, optimizer, dataloaders["train"], criterion, 
            num_train, mixed_precision, batch_scheduler=batch_scheduler
        )
        print("Train Loss: {:.4f} Acc: {:.4f}".format(train_loss, train_acc))
        
        val_loss, val_acc = validate_phrase(
            model, dataloaders["val"], criterion, num_val
        )
        print("Val Loss: {:.4f} Acc: {:.4f}".format(val_loss, val_acc))
        
        if epoch_scheduler:
            epoch_scheduler.step(val_loss)
        
        # deep copy the model
        if val_acc > best_acc:
            print(f"In epoch {epoch}, highest val accuracy increases from {best_acc} to {val_acc}.")
            best_acc = val_acc
            if save_dir:
                torch.save(model.state_dict(), os.path.join(save_dir, "eff_0_high_acc.pth"))

        # deep copy the model
        if val_loss < lowest_loss:
            print(f"In epoch {epoch}, lowest val loss decreases from {lowest_loss} to {val_loss}.")
            lowest_loss = val_loss
            if save_dir:
                torch.save(model.state_dict(), os.path.join(save_dir, "eff_0_low_loss.pth"))
        
        callbacks["train_loss_list"].append(train_loss)
        callbacks["train_acc_list"].append(train_acc)
        callbacks["val_loss_list"].append(val_loss)
        callbacks["val_acc_list"].append(val_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    if save_dir:
        pd.DataFrame(callbacks).to_csv(os.path.join(save_dir, "callbacks.csv"), index=False)
    
    return callbacks

    ## load best model weights
    #model.load_state_dict(best_model_wts)
    # return best_model_wts