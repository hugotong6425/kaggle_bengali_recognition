# ---
# jupyter:
#   jupytext:
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
import pickle
import random
import os
import torch
from tqdm import tqdm

import numpy as np
import torch.functional as F

from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models

from hw_grapheme.dl_utils.radam import RAdam

# +
sz = 128
bs = 64
nfolds = 4 #keep the same split as the initial dataset
fold = 0
SEED = 2019
TRAIN = '../input/grapheme-imgs-128x128/'
LABELS = '../input/bengaliai-cv19/train.csv'

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(SEED)


# -

class Loss_combine(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, input, target,reduction='mean'):
        x1,x2,x3 = input
        x1,x2,x3 = x1.float(),x2.float(),x3.float()
        y = target.long()
        return 0.7*F.cross_entropy(x1,y[:,0],reduction=reduction) + 0.1*F.cross_entropy(x2,y[:,1],reduction=reduction) + \
          0.2*F.cross_entropy(x3,y[:,2],reduction=reduction)


class GraphemeDataset(Dataset):
    def __init__(self, data_list, _type='train'):            
        self.data_list = data_list
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        data = self.data_list[idx]
        # name = data["name"]
        image = data["image"]
        grapheme_root = data["grapheme_root"]
        vowel_diacritic = data["vowel_diacritic"]
        consonant_diacritic = data["consonant_diacritic"]
        return image, vowel_diacritic, grapheme_root, consonant_diacritic


class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50,self).__init__()
        
        arch = models.resnet50(num_classes=1000, pretrained=True)
        arch = list(arch.children())
        w = arch[0].weight
        arch[0] = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=2, bias=False)
        arch[0].weight = nn.Parameter(torch.mean(w, dim=1, keepdim=True))
        self.backbone = nn.Sequential(*arch[:-1])
        
        # vowel_diacritic
        self.fc1 = nn.Linear(2048,11)
        # grapheme_root
        self.fc2 = nn.Linear(2048,168)
        # consonant_diacritic
        self.fc3 = nn.Linear(2048,7)
        
    def forward(self, x):
        x = self.backbone(x)
        x = x.view(-1, 2048)
        # print("x: ", x.shape)
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        x3 = self.fc3(x)
        
        return x1,x2,x3


# +
batch_size = 32

with open("../data/processed_data/train_data_0.pickle", "rb") as f:
    data_list = pickle.load(f)
    
train_dataset = GraphemeDataset(data_list)
train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
# -

len(train_dataset)

iter(train_loader).next()

# +
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = ResNet50()
model = nn.DataParallel(model)
model.to(device);

# -

optimizer = RAdam(model.parameters(), lr=1e-3)
#scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-4, max_lr=0.05)
criterion = nn.CrossEntropyLoss()


inputs, vowel, root, consonant = iter(train_loader).next()

inputs.shape

# +
epochs = 50 # original 50
model.train()
losses = []
accs = []
for epoch in range(epochs):
    running_loss = 0.0
    running_acc_vowel = 0.0
    running_acc_root = 0.0
    running_acc_consonant = 0.0
    
    for idx, (inputs, vowel, root, consonant) in tqdm(enumerate(train_loader),total=len(train_loader)):
        inputs = inputs.to(device)
        vowel = vowel.to(device)
        root = root.to(device)
        consonant = consonant.to(device)
        
        optimizer.zero_grad()
        outputs1, outputs2, outputs3 = model(inputs.unsqueeze(1).float())
        # print("outputs1: ", outputs1)
#         print("inputs: ", inputs.shape)
#         print("outputs1.shape: ", outputs1.shape)
#         print("vowel: ", vowel.shape)
        loss1 = criterion(outputs1, vowel.long())
        loss2 = criterion(outputs2, root.long())
        loss3 = criterion(outputs3, consonant.long())
        
        mini_batch_loss = loss1 + loss2 + loss3
        running_loss += mini_batch_loss * inputs.size(0)
        running_acc_vowel += (outputs1.argmax(1) == vowel.long()).sum()
        running_acc_root += (outputs2.argmax(1) == root.long()).sum()
        running_acc_consonant += (outputs3.argmax(1) == consonant.long()).sum()
        
        mini_batch_loss.backward()
        
        optimizer.step()
    #scheduler.step()
    losses.append(running_loss/len(train_dataset))
    print('running_acc_vowel : {:.2f}%'.format(running_acc_vowel.float()/len(train_dataset)))
    print('running_acc_root : {:.2f}%'.format(running_acc_root.float()/len(train_dataset)))
    print('running_acc_consonant : {:.2f}%'.format(running_acc_consonant.float()/len(train_dataset)))

    print('loss : {:.4f}'.format(running_loss/len(train_loader)))
    
torch.save(model.state_dict(), 'resnet34_50epochs_saved_weights.pth')
# -

model


