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

# %reload_ext autoreload
# %autoreload 2
# %matplotlib inline

# +
import torch
from torchvision import models
import random
import os
import numpy as np
import pandas as pd
from fastai.vision import ImageList, transform
from functools import partial

from hw_grapheme.densenet import Dnet_1ch

from hw_grapheme.dl_utils.radam import Over9000
from hw_grapheme.loss_func import Loss_combine
from hw_grapheme.callbacks import Metric_idx, Metric_tot, MixUpCallback

from hw_grapheme.dl_utils.csvlogger import CSVLogger
from fastai.basic_train import Learner

from torch import nn
from fastai.callbacks import SaveModelCallback



# import fastai
# import warnings

# import matplotlib.pyplot as plt
# import torch.nn.functional as F

# from sklearn.model_selection import KFold
# from fastai.callbacks import SaveModelCallback, TrackerCallback
# from fastai.callback import Callback
# from torch import nn, Tensor


# from hw_grapheme.dl_utils.radam import RAdam, Over9000
# from hw_grapheme.dl_utils.mish_activation import Mish, to_Mish



# warnings.filterwarnings("ignore")

# fastai.__version__

# from functools import partial
# 


# +
sz = 128
bs = 128
nfolds = 4 #keep the same split as the initial dataset
fold = 1
SEED = 2019
TRAIN = '../data/processed_data/train_images/'
LABELS = '../data/train.csv'

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

df = pd.read_csv(LABELS)
nunique = list(df.nunique())[1:-1]
print(nunique)
df.head()

range(fold*len(df)//nfolds,(fold+1)*len(df)//nfolds)

# +
stats = ([0.0692], [0.2051])
data = (ImageList.from_df(df, path='.', folder=TRAIN, suffix='.png', 
        cols='image_id', convert_mode='L')
        .split_by_idx(range(fold*len(df)//nfolds,(fold+1)*len(df)//nfolds))
        .label_from_df(cols=['grapheme_root','vowel_diacritic','consonant_diacritic'])
        .transform(transform.get_transforms(do_flip=False,max_warp=0.1), size=sz, padding_mode='zeros')
        .databunch(bs=bs)).normalize(stats)

# data.show_batch()
# -

data

# +
Metric_grapheme = partial(Metric_idx,0)
Metric_vowel = partial(Metric_idx,1)
Metric_consonant = partial(Metric_idx,2)

arch = models.densenet121
model = Dnet_1ch(arch, nunique)

learn = Learner(data, model, loss_func=Loss_combine(), opt_func=Over9000,
        metrics=[Metric_grapheme(),Metric_vowel(),Metric_consonant(),Metric_tot()])

learn.model = nn.DataParallel(learn.model) 


logger = CSVLogger(learn,f'log{fold}')
learn.clip_grad = 1.0
learn.split([model.head1])
learn.unfreeze()
# -

learn.summary()

learn.fit_one_cycle(100, max_lr=slice(0.2e-2,1e-2), wd=[1e-3,0.1e-1], pct_start=0.0, 
    div_factor=100, callbacks = [logger, SaveModelCallback(learn,monitor='metric_tot',
    mode='max',name=f'model_{fold}'),MixUpCallback(learn)])
#metrics: Metric_grapheme, Metric_vowel, Metric_consonant, Metric_tot (competition metric)


