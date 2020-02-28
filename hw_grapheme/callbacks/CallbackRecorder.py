import torch
import wandb

import numpy as np

from sklearn.metrics import recall_score


class CallbackRecorder:
    def __init__(self):
        self.reset()

    def reset(self):
        self.running_loss = 0.0
        self.root_corrects = 0
        self.vowel_corrects = 0
        self.consonant_corrects = 0
        self.data_count = 0

        self.root_predict = []
        self.vowel_predict = []
        self.consonant_predict = []
        self.root_true = []
        self.vowel_true = []
        self.consonant_true = []

    def update(
        self,
        loss,
        root_logit,
        vowel_logit,
        consonant_logit,
        root_true,
        vowel_true,
        consonant_true,
    ):
        assert root_logit.shape[0] == vowel_logit.shape[0]

        # update loss
        self.running_loss += loss.item() * root_logit.shape[0]

        # update for calculating accuracy
        self.data_count += root_logit.shape[0]
        _, preds_root = torch.max(root_logit, 1)
        _, preds_vowel = torch.max(vowel_logit, 1)
        _, preds_consonant = torch.max(consonant_logit, 1)
        self.root_corrects += torch.sum(preds_root == root_true)
        self.vowel_corrects += torch.sum(preds_vowel == vowel_true)
        self.consonant_corrects += torch.sum(preds_consonant == consonant_true)

        # update for calculating macro recall
        self.root_predict.append(preds_root.cpu().numpy())
        self.vowel_predict.append(preds_vowel.cpu().numpy())
        self.consonant_predict.append(preds_consonant.cpu().numpy())
        self.root_true.append(root_true.cpu().numpy())
        self.vowel_true.append(vowel_true.cpu().numpy())
        self.consonant_true.append(consonant_true.cpu().numpy())

    def _unregular_array_flatten(self, nested_array):
        return [data_point for batch in nested_array for data_point in batch]

    def evaluate(self):
        self.loss = self.running_loss / float(self.data_count)

        # calculate accuracy
        root_acc = (self.root_corrects.double() / self.data_count).item()
        vowel_acc = (self.vowel_corrects.double() / self.data_count).item()
        consonant_acc = (self.consonant_corrects.double() / self.data_count).item()
        # print("root_acc: ", root_acc)
        # print("type(root_acc): ", type(root_acc))
        # print("vowel_acc: ", vowel_acc)
        # print("consonant_acc: ", consonant_acc)

        combined_acc = np.average(
            [root_acc, vowel_acc, consonant_acc], weights=[2, 1, 1]
        )

        # calculate recall
        self.root_predict = np.array(self.root_predict)
        self.vowel_predict = np.array(self.vowel_predict)
        self.consonant_predict = np.array(self.consonant_predict)
        self.root_true = np.array(self.root_true)
        self.vowel_true = np.array(self.vowel_true)
        self.consonant_true = np.array(self.consonant_true)

        self.root_predict = self._unregular_array_flatten(self.root_predict)
        self.vowel_predict = self._unregular_array_flatten(self.vowel_predict)
        self.consonant_predict = self._unregular_array_flatten(self.consonant_predict)
        self.root_true = self._unregular_array_flatten(self.root_true)
        self.vowel_true = self._unregular_array_flatten(self.vowel_true)
        self.consonant_true = self._unregular_array_flatten(self.consonant_true)

        # return self.root_true, self.root_predict

        root_recall = recall_score(self.root_true, self.root_predict, average="macro")
        vowel_recall = recall_score(
            self.vowel_true, self.vowel_predict, average="macro"
        )
        consonant_recall = recall_score(
            self.consonant_true, self.consonant_predict, average="macro"
        )

        # print("root_recall: ", root_recall)
        combined_recall = np.average(
            [root_recall, vowel_recall, consonant_recall], weights=[2, 1, 1]
        )

        self.accuracy_metrics = [root_acc, vowel_acc, consonant_acc, combined_acc]
        self.recall_metrics = [
            root_recall,
            vowel_recall,
            consonant_recall,
            combined_recall,
        ]

    def wandb_log(self, phrase):
        """
        phrase: ["train", "val"]
        """

        root_acc, vowel_acc, consonant_acc, combined_acc = self.accuracy_metrics
        root_recall, vowel_recall, consonant_recall, combined_recall = (
            self.recall_metrics
        )

        wandb.log(
            {
                f"{phrase} loss": self.loss,
                f"{phrase} root acc": root_acc,
                f"{phrase} vowel acc": vowel_acc,
                f"{phrase} consonant acc": consonant_acc,
                f"{phrase} combined acc": combined_acc,
                f"{phrase} root recall": root_recall,
                f"{phrase} vowel recall": vowel_recall,
                f"{phrase} consonant recall": consonant_recall,
                f"{phrase} combined recall": combined_recall,
            }
        )

    def get_loss(self):
        # all are float
        return self.loss

    def get_accuracy(self):
        # all are float
        return self.accuracy_metrics

    def get_recall(self):
        # all are float
        return self.recall_metrics

    def print_statistics(self):
        root_acc, vowel_acc, consonant_acc, combined_acc = self.accuracy_metrics
        root_recall, vowel_recall, consonant_recall, combined_recall = (
            self.recall_metrics
        )
        print(
            "Loss: {:.4f}, root_acc: {:.4f}, vowel_acc: {:.4f}, consonant_acc: {:.4f}, combined_acc: {:.4f}, root_recall: {:.4f}, vowel_recall: {:.4f}, consonant_recall: {:.4f}, combined_recall: {:.4f}".format(
                self.loss,
                root_acc,
                vowel_acc,
                consonant_acc,
                combined_acc,
                root_recall,
                vowel_recall,
                consonant_recall,
                combined_recall,
            )
        )
