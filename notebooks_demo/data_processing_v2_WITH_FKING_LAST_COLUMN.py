# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
# https://www.kaggle.com/iafoss/image-preprocessing-128x128

import cv2
import pickle

import pandas as pd
import numpy as np

from pathlib import Path
from tqdm import tqdm_notebook as tqdm

from sklearn import preprocessing


# +
SIZE = 224

PROCESS_DATA = Path("../data/processed/size_224_with_last_column")
RAW_DATA = Path("../data/raw")

PROCESS_DATA.mkdir(exist_ok=True, parents=True)
RAW_DATA.mkdir(exist_ok=True,parents=True)

# +
HEIGHT = 137
WIDTH = 236

def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax

def crop_resize(img0, size=SIZE, pad=16):
    #crop a box around pixels large than the threshold 
    #some images contain line at the sides
    ymin,ymax,xmin,xmax = bbox(img0[5:-5,5:-5] > 80)
    #cropping may cut too much, so we need to add it back
    xmin = xmin - 13 if (xmin > 13) else 0
    ymin = ymin - 10 if (ymin > 10) else 0
    xmax = xmax + 13 if (xmax < WIDTH - 13) else WIDTH
    ymax = ymax + 10 if (ymax < HEIGHT - 10) else HEIGHT
    img0 = img0[ymin:ymax,xmin:xmax]
    #remove lo intensity pixels as noise
    img0[img0 < 28] = 0
    lx, ly = xmax-xmin,ymax-ymin
    l = max(lx,ly) + pad
    #make sure that the aspect ratio is kept in rescaling
    img0 = np.pad(img0, [((l-ly)//2,), ((l-lx)//2,)], mode='constant')
    return cv2.resize(img0,(size,size), interpolation=cv2.INTER_AREA)

def proces_image(images):
    """
    images: (batch_size, 32332), np array
    
    """
    processed_image_list = []

    for idx in tqdm(range(len(images))):
        img0 = 255 - images[idx].reshape(HEIGHT, WIDTH).astype(np.uint8)
        #normalize each image by its max val
        # print(img0.max())
        img = crop_resize(img0)
        # print(img.max())
        img = (img*(255.0/img.max())).astype(np.uint8)
        # print(img.max())
        # print()
        processed_image_list.append(img)
    
    return np.array(processed_image_list)


# -

# get a full set of grapheme first
train_df = pd.read_csv("../data/raw/train.csv")
grapheme_full = train_df["grapheme"]
le = preprocessing.LabelEncoder()
le.fit(grapheme_full)

le.classes_, len(le.classes_)

# +
train_df = pd.read_csv("../data/raw/train.csv")

for i in range(4):
    print(f"==================== {i} ==================")
    fn = f"train_image_data_{i}.parquet"
    process_fn = f"train_data_{i}.pickle"
    df = pd.read_parquet(RAW_DATA/fn)
    
    if (PROCESS_DATA/process_fn).is_file():
        print(f'parquet {i} processed already, skipped')
        continue
    
    merged_df = df.merge(train_df, on="image_id")

    image_name = merged_df["image_id"]
    merged_df["grapheme_label"] = le.transform(merged_df["grapheme"])
    label = merged_df[["grapheme_root","vowel_diacritic","consonant_diacritic", "grapheme_label"]].astype(np.int16)
    image = merged_df.drop(["image_id", "grapheme_root","vowel_diacritic","consonant_diacritic", "grapheme", "grapheme_label"], axis=1).values

    image = proces_image(image)

    with open(PROCESS_DATA/process_fn, "wb") as f:
        pickle.dump((image, image_name, label.values), f, protocol=4)
# +
pickle_path = "../data/processed/size_224_with_last_column/train_data_0.pickle"

with open(pickle_path, "rb") as f:
    train_data = pickle.load(f)
# -


max(train_data[2][:, 3])





train_data[0][0].max()

train_data[0].shape

train_data[0].max()


