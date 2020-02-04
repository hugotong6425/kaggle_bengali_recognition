import time
import torch
import os

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm_notebook


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

    # print("In train phrase")
    running_loss = 0.0
    root_corrects = 0
    vowel_corrects = 0
    consonant_corrects = 0

    # Iterate over data.
    for image, root, vowel, consonant in tqdm_notebook(train_dataloader):
        image = image.to("cuda")
        root = root.long().to("cuda")
        vowel = vowel.long().to("cuda")
        consonant = consonant.long().to("cuda")
        # print("image.shape: ", image.shape)
        # print("root.shape: ", root.shape)
        # print("vowel.shape: ", vowel.shape)
        # print("consonant.shape: ", consonant.shape)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward with all root vowel consonant outputs
        with torch.set_grad_enabled(True):
            root_logit, vowel_logit, consonant_logit = model(image)
            # print("root_logit.shape: ", root_logit.shape)
            # print("vowel_logit.shape: ", vowel_logit.shape)
            # print("consonant_logit.shape: ", consonant_logit.shape)
            loss = criterion((root_logit, vowel_logit, consonant_logit), root, vowel, consonant)

        # backward + optimize
        if mixed_precision:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        optimizer.step()

        # statistics
        running_loss += loss.item() * image.size(0)
        
        _, preds_root = torch.max(root_logit, 1)
        _, preds_vowel = torch.max(vowel_logit, 1)
        _, preds_consonant = torch.max(consonant_logit, 1)

        root_corrects += torch.sum(preds_root == root.data)
        vowel_corrects += torch.sum(preds_vowel == vowel.data)
        consonant_corrects += torch.sum(preds_consonant == consonant.data)

        if batch_scheduler:
            batch_scheduler.step()

    train_loss = running_loss / float(num_train)

    root_acc = root_corrects.double() / num_train
    vowel_acc = vowel_corrects.double() / num_train
    consonant_acc = consonant_corrects.double() / num_train
    
    return train_loss, root_acc, vowel_acc, consonant_acc


def validate_phrase(model, valid_dataloader, criterion, num_val):
    # Each epoch has a training and validation phase
    model.eval()   # Set model to evaluate mode

    running_loss = 0.0
    root_corrects = 0
    vowel_corrects = 0
    consonant_corrects = 0

    # Iterate over data.
    for image, root, vowel, consonant in tqdm_notebook(valid_dataloader):
        image = image.to("cuda")
        root = root.long().to("cuda")
        vowel = vowel.long().to("cuda")
        consonant = consonant.long().to("cuda")

        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):
            root_logit, vowel_logit, consonant_logit = model(image)
            loss = criterion((root_logit, vowel_logit, consonant_logit), root, vowel, consonant)

        running_loss += loss.item() * image.size(0)

        # statistics
        _, preds_root = torch.max(root_logit, 1)
        _, preds_vowel = torch.max(vowel_logit, 1)
        _, preds_consonant = torch.max(consonant_logit, 1)

        root_corrects += torch.sum(preds_root == root.data)
        vowel_corrects += torch.sum(preds_vowel == vowel.data)
        consonant_corrects += torch.sum(preds_consonant == consonant.data)

    val_loss = running_loss / float(num_val)

    root_acc = root_corrects.double() / num_val
    vowel_acc = vowel_corrects.double() / num_val
    consonant_acc = consonant_corrects.double() / num_val

    return val_loss, root_acc, vowel_acc, consonant_acc


def train_model(
    model, criterion, optimizer, dataloaders, 
    mixed_precision, callbacks, num_epochs=25, 
    epoch_scheduler=None, batch_scheduler=None,
    save_dir=None
):
    since = time.time()

    callbacks["train_loss_list"] = []
    callbacks["train_root_acc_list"] = []
    callbacks["train_vowel_acc_list"] = []
    callbacks["train_consonant_acc_list"] = []
    callbacks["train_combined_acc_list"] = []

    callbacks["val_loss_list"] = []
    callbacks["val_root_acc_list"] = []
    callbacks["val_vowel_acc_list"] = []
    callbacks["val_consonant_acc_list"] = []
    callbacks["val_combined_acc_list"] = []
    
    num_train = len(dataloaders["train"].dataset)
    num_val = len(dataloaders["val"].dataset)
    
    #high_acc_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    #low_loss_model_wts = copy.deepcopy(model.state_dict())
    lowest_loss = 999

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        train_loss, train_root_acc, train_vowel_acc, train_consonant_acc = train_phrase(
            model, optimizer, dataloaders["train"], criterion, 
            num_train, mixed_precision, batch_scheduler=batch_scheduler
        )
        combined_train_acc = train_root_acc * 0.5 + (train_vowel_acc + train_consonant_acc) * 0.25
        print(
            "Train Loss: {:.4f}, root_acc: {:.4f}, vowel_acc: {:.4f}, consonant_acc: {:.4f}, combined_acc: {:.4f}"
            .format(train_loss, train_root_acc, train_vowel_acc, train_consonant_acc, combined_train_acc)
        )
        
        val_loss, val_root_acc, val_vowel_acc, val_consonant_acc = validate_phrase(
            model, dataloaders["val"], criterion, num_val
        )
        combined_val_acc = val_root_acc * 0.5 + (val_vowel_acc + val_consonant_acc) * 0.25
        print(
            "Val Loss: {:.4f}, root_acc: {:.4f}, vowel_acc: {:.4f}, consonant_acc: {:.4f}, combined_acc: {:.4f}"
            .format(val_loss, val_root_acc, val_vowel_acc, val_consonant_acc, combined_val_acc)
        )
        
        if epoch_scheduler:
            epoch_scheduler.step(val_loss)
        
        # deep copy the model
        if combined_val_acc > best_acc:
            print(f"In epoch {epoch}, highest val accuracy increases from {best_acc} to {combined_val_acc}.")
            best_acc = combined_val_acc
            if save_dir:
                torch.save(model.state_dict(), os.path.join(save_dir, "eff_0_high_acc.pth"))

        # deep copy the model
        if val_loss < lowest_loss:
            print(f"In epoch {epoch}, lowest val loss decreases from {lowest_loss} to {val_loss}.")
            lowest_loss = val_loss
            if save_dir:
                torch.save(model.state_dict(), os.path.join(save_dir, "eff_0_low_loss.pth"))
        
        callbacks["train_loss_list"].append(train_loss)
        callbacks["train_root_acc_list"].append(train_root_acc.item())
        callbacks["train_vowel_acc_list"].append(train_vowel_acc.item())
        callbacks["train_consonant_acc_list"].append(train_consonant_acc.item())
        callbacks["train_combined_acc_list"].append(combined_train_acc.item())    
          
        callbacks["val_loss_list"].append(val_loss)
        callbacks["val_root_acc_list"].append(val_root_acc.item())
        callbacks["val_vowel_acc_list"].append(val_vowel_acc.item())
        callbacks["val_consonant_acc_list"].append(val_consonant_acc.item())
        callbacks["val_combined_acc_list"].append(combined_val_acc.item())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best Combnied Acc: {:4f}'.format(best_acc))

    if save_dir:
        pd.DataFrame(callbacks).to_csv(os.path.join(save_dir, "callbacks.csv"), index=False)
    
    return callbacks

    ## load best model weights
    #model.load_state_dict(best_model_wts)
    # return best_model_wts