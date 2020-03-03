import os
import time
import torch
import wandb

import numpy as np
import pandas as pd

from apex import amp
from tqdm import tqdm_notebook

from hw_grapheme.callbacks.CallbackRecorder import CallbackRecorder
from hw_grapheme.callbacks.ExportLogger import ExportLogger
from hw_grapheme.train_utils.loss_func import (
    cross_entropy_criterion,
    cutmix_criterion,
    mixup_criterion,
)


##### for mix up training
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix(data, targets1, targets2, targets3, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets1 = targets1[indices]
    shuffled_targets2 = targets2[indices]
    shuffled_targets3 = targets3[indices]

    lam = np.random.beta(alpha, alpha)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    data[:, :, bbx1:bbx2, bby1:bby2] = data[indices, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))

    targets = [
        targets1,
        shuffled_targets1,
        targets2,
        shuffled_targets2,
        targets3,
        shuffled_targets3,
        lam,
    ]
    return data, targets


def mixup(data, targets1, targets2, targets3, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets1 = targets1[indices]
    shuffled_targets2 = targets2[indices]
    shuffled_targets3 = targets3[indices]

    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)  # Remove duplicate case
    data = data * lam + shuffled_data * (1 - lam)
    targets = [
        targets1,
        shuffled_targets1,
        targets2,
        shuffled_targets2,
        targets3,
        shuffled_targets3,
        lam,
    ]

    return data, targets


def train_phrase(
    model,
    optimizer,
    train_dataloader,
    mixed_precision,
    train_loss_prob,
    head_loss_weight,
    mixup_alpha=0.4,
    batch_scheduler=None,
    wandb_log=True,
):
    recorder = CallbackRecorder()
    model.train()  # Set model to training mode

    # Iterate over data.
    for images, root, vowel, consonant in tqdm_notebook(train_dataloader):
        images = images.to("cuda")
        root = root.long().to("cuda")
        vowel = vowel.long().to("cuda")
        consonant = consonant.long().to("cuda")

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward with all root vowel consonant outputs
        # with torch.set_grad_enabled(True):
        train_method_choice_list = ["mixup", "cutmix", "cross_entropy"]
        train_method = np.random.choice(train_method_choice_list, 1, p=train_loss_prob)

        if train_method == "mixup":
            images, targets = mixup(images, root, vowel, consonant, mixup_alpha)
            root_logit, vowel_logit, consonant_logit = model(images)
            loss = mixup_criterion(
                root_logit,
                vowel_logit,
                consonant_logit,
                targets,
                weights=head_loss_weight,
            )
        elif train_method == "cutmix":
            images, targets = cutmix(images, root, vowel, consonant, mixup_alpha)
            root_logit, vowel_logit, consonant_logit = model(images)
            loss = cutmix_criterion(
                root_logit,
                vowel_logit,
                consonant_logit,
                targets,
                weights=head_loss_weight,
            )
        elif train_method == "cross_entropy":
            root_logit, vowel_logit, consonant_logit = model(images)
            targets = (root, vowel, consonant)
            loss = cross_entropy_criterion(
                root_logit,
                vowel_logit,
                consonant_logit,
                targets,
                weights=head_loss_weight,
            )

        # backward + optimize
        if mixed_precision:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        optimizer.step()

        recorder.update(
            loss,
            root_logit,
            vowel_logit,
            consonant_logit,
            root.data,
            vowel.data,
            consonant.data,
        )

        if batch_scheduler:
            batch_scheduler.step()

    recorder.evaluate()
    # root_true, root_predict = recorder.evaluate()
    # return root_true, root_predict

    if wandb_log:
        recorder.wandb_log(phrase="train")

    return recorder


def validate_phrase(model, valid_dataloader, wandb_log=True):
    recorder = CallbackRecorder()

    # Each epoch has a training and validation phase
    model.eval()  # Set model to evaluate mode

    running_loss = 0.0
    root_corrects = 0
    vowel_corrects = 0
    consonant_corrects = 0

    # Iterate over data.
    for images, root, vowel, consonant in tqdm_notebook(valid_dataloader):
        images = images.to("cuda")
        root = root.long().to("cuda")
        vowel = vowel.long().to("cuda")
        consonant = consonant.long().to("cuda")

        # forward
        # track history if only in train
        with torch.no_grad():
            root_logit, vowel_logit, consonant_logit = model(images)
            targets = (root, vowel, consonant)
            loss = cross_entropy_criterion(
                root_logit,
                vowel_logit,
                consonant_logit,
                targets,
                weights=[1 / 3, 1 / 3, 1 / 3],
            )

        recorder.update(
            loss,
            root_logit,
            vowel_logit,
            consonant_logit,
            root.data,
            vowel.data,
            consonant.data,
        )

    recorder.evaluate()

    if wandb_log:
        recorder.wandb_log(phrase="val")

    return recorder


def train_model(
    model,
    optimizer,
    dataloaders,
    mixed_precision,
    train_loss_prob,
    head_loss_weight,
    mixup_alpha=0.4,
    num_epochs=25,
    epoch_scheduler=None,
    batch_scheduler=None,
    save_dir=None,
    wandb_log=False,
):
    since = time.time()

    if wandb_log:
        wandb.init(project="my-project")

    export_logger = ExportLogger(save_dir)

    # need to co-change ExportLogger.update_from_callbackrecorder if want to
    # change list_of_field
    # prefer dont change
    list_of_field = [
        "train_loss",
        "train_root_acc",
        "train_vowel_acc",
        "train_consonant_acc",
        "train_combined_acc",
        "train_root_recall",
        "train_vowel_recall",
        "train_consonant_recall",
        "train_combined_recall",
        "val_loss",
        "val_root_acc",
        "val_vowel_acc",
        "val_consonant_acc",
        "val_combined_acc",
        "val_root_recall",
        "val_vowel_recall",
        "val_consonant_recall",
        "val_combined_recall",
    ]
    export_logger.define_field_to_record(list_of_field)

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        train_recorder = train_phrase(
            # return train_phrase(
            model,
            optimizer,
            dataloaders["train"],
            mixed_precision,
            train_loss_prob,
            head_loss_weight,
            mixup_alpha=mixup_alpha,
            batch_scheduler=batch_scheduler,
            wandb_log=wandb_log,
        )
        print("Finish training")
        train_recorder.print_statistics()
        print()

        valid_recorder = validate_phrase(model, dataloaders["val"], wandb_log=wandb_log)
        print("Finish validating")
        valid_recorder.print_statistics()
        print()

        # record training statistics into ExportLogger
        export_logger.update_from_callbackrecorder(train_recorder, valid_recorder)

        # check whether val_loss gets lower/val_combined_recall gets higher
        # also save the model.pth is required
        export_logger.export_models_and_csv(model, valid_recorder)

        print()

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    # print("Best Combnied Acc: {:4f}".format(best_acc))

    return export_logger.callbacks

    ## load best model weights
    # model.load_state_dict(best_model_wts)
    # return best_model_wts
