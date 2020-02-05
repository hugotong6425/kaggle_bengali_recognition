# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
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
# https://www.kaggle.com/iafoss/image-preprocessing-128x128
import cv2
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from sys import getsizeof, getrefcount
import psutil
import gc

# +
HEIGHT = 137
WIDTH = 236
SIZE = 224

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
    return cv2.resize(img0,(size,size))

def proces_image(images):
    """
    images: (batch_size, 32332), np array
    
    """
    image_max = images.max(axis=1).reshape(-1, 1).astype(np.uint8)
    images = (255 - images)/image_max*255
    images = images.reshape(-1, HEIGHT, WIDTH).astype(np.uint8)
    
    processed_image_list = []

    for idx in tqdm(range(len(images))):
        processed_image_list.append(crop_resize(images[idx]))

    del image_max
    del images
    print(gc.collect())
    
    return np.array(processed_image_list)


# +
from pathlib import Path
PROCESS_DATA = Path("../data/processed_data/size_224")
RAW_DATA = Path("../data")

PROCESS_DATA.mkdir(exist_ok=True)
RAW_DATA.mkdir(exist_ok=True)

# +
train_df = pd.read_csv("../data/train.csv")

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
    label = merged_df[["grapheme_root","vowel_diacritic","consonant_diacritic"]].astype(np.uint8)
    image = merged_df.drop(["image_id", "grapheme_root","vowel_diacritic","consonant_diacritic", "grapheme"], axis=1).values

    
    # full image processing
    image = image * 1

    print(getsizeof(image)/1024/1024/1024)
    print(psutil.virtual_memory()[4]/1024/1024/1024)

    image = proces_image(image)
    print(gc.collect())
    print(getsizeof(image)/1024/1024/1024)
    print(psutil.virtual_memory()[4]/1024/1024/1024)
    
    with open(PROCESS_DATA/process_fn, "wb") as f:
#         pickle.dump(image, f)
        pickle.dump((image, image_name, label.values), f)

# -

# # Step by step

# strange thing in memory
print(getsizeof(image)/1024/1024/1024)
print(psutil.virtual_memory()[4]/1024/1024/1024)
image = image * 1
print(getsizeof(image)/1024/1024/1024)
print(psutil.virtual_memory()[4]/1024/1024/1024)


# +
print(getsizeof(image)/1024/1024/1024)

image_max = image.max(axis=1).reshape(-1, 1).astype(np.uint8)
print(getsizeof(image_max))

image = (255 - image)/image_max*255
print(getsizeof(image)/1024/1024/1024)

image = image.reshape(-1, HEIGHT, WIDTH).astype(np.uint8)
print(getsizeof(image)/1024/1024/1024)


# +
processed_image_list = []

for idx in tqdm(range(len(image))):
    processed_image_list.append(crop_resize(image[idx]))
    

# result_list = []

# for idx in tqdm(range(len(image))):
#     root, vowel, consonant = label.values[idx]
#     result_list.append({
#         "image": crop_resize(image[idx]),
#         "name": image_name[idx],
#         "grapheme_root": root,
#         "vowel_diacritic": vowel,
#         "consonant_diacritic": consonant
#     })
# -

np.array(processed_image_list).shape

# size 224 X 224 
for i in range(5):
    plt.imshow(processed_image_list[i])
    plt.show()

a = (np.array(processed_image_list), image_name, label.values)

a[2]

with open("train_data_0.pickle", "wb") as f:
    pickle.dump(a, f)

with open("../data/processed_data/train_data_0.pickle", "rb") as f:
    data_list = pickle.load(f)
