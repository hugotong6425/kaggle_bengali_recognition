import torch
import os
from pathlib import Path
import pandas as pd
import datetime

class ExportLogger:
    """ Export csv and model.pth"""

    def __init__(self, save_dir):
        """
        if save_dir is None, then don't save model.pth
        """
        self.highest_recall = 0.0
        self.lowest_loss = 999
        self.save_dir = save_dir
        self.save_dir = Path(self.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        now = datetime.datetime.now()
        now = now.strftime("%Y%m%d-%H%M%S")
        self.model_save_dir = self.save_dir/now
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def define_field_to_record(self, list_of_field):
        """ e.g. list_of_field = ["train_loss", "combined_recall"]
        must record val-loss and val_combined_recall as they are used to output model.pth

        initialize self.callbacks
        save self.list_of_field

        """
        self.callbacks = {}
        self.list_of_field = list_of_field
        for field in list_of_field:
            self.callbacks[field] = []

    def update(self, list_of_update):
        """
        list_of_update must have the same ordering as list_of_field defined.
        value in list_of_update should be np.array

        """
        for field, value in zip(self.list_of_field, list_of_update):
            self.callbacks[field].append(value)

    def update_from_callbackrecorder(self, train_recorder, val_recorder):
        """
        use self.update with input of train_recorder and val_recorder

        here assume list_of_field = [
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
        """
        # get train statistics
        train_loss = train_recorder.get_loss()
        train_root_acc, train_vowel_acc, train_consonant_acc, train_combined_acc = (
            train_recorder.get_accuracy()
        )
        train_root_recall, train_vowel_recall, train_consonant_recall, train_combined_recall = (
            train_recorder.get_recall()
        )

        # get val statistics
        val_loss = val_recorder.get_loss()
        val_root_acc, val_vowel_acc, val_consonant_acc, val_combined_acc = (
            val_recorder.get_accuracy()
        )
        val_root_recall, val_vowel_recall, val_consonant_recall, val_combined_recall = (
            val_recorder.get_recall()
        )

        list_of_update = [
            train_loss,
            train_root_acc,
            train_vowel_acc,
            train_consonant_acc,
            train_combined_acc,
            train_root_recall,
            train_vowel_recall,
            train_consonant_recall,
            train_combined_recall,
            val_loss,
            val_root_acc,
            val_vowel_acc,
            val_consonant_acc,
            val_combined_acc,
            val_root_recall,
            val_vowel_recall,
            val_consonant_recall,
            val_combined_recall,
        ]

        self.update(list_of_update)

    def export_models_and_csv(self, model, val_recorder):
        """
        model: pytorch model
        valid_recorder: CallbackRecorder
        """
        val_loss = val_recorder.get_loss()
        val_root_recall, val_vowel_recall, val_consonant_recall, val_combined_recall = (
            val_recorder.get_recall()
        )

        # export model with lowest val loss
        if val_loss < self.lowest_loss:
            print(f"Lowest val_loss decreases from {self.lowest_loss} to {val_loss}.")
            self.lowest_loss = val_loss
            if self.save_dir:
                torch.save(
                    model.state_dict(),
                    os.path.join(self.model_save_dir, "low_loss_model.pth"),
                )

        # export model with highest val recall
        if val_combined_recall > self.highest_recall:
            print(
                f"Highest val_combined_recall increases from {self.highest_recall} to {val_combined_recall}."
            )
            self.highest_recall = val_combined_recall
            if self.save_dir:
                torch.save(
                    model.state_dict(),
                    os.path.join(self.model_save_dir, "high_recall_model.pth"),
                )

        # export csv
        if self.model_save_dir:
            pd.DataFrame(self.callbacks).to_csv(
                os.path.join(self.save_dir, "callbacks.csv"), index=False
            )
