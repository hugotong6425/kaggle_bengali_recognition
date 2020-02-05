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

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

image0_df = pd.read_parquet("../data/train_image_data_0.parquet") 
train_df = pd.read_csv("../data/train.csv")
class_map_df = pd.read_csv("../data/class_map.csv")

test_df = pd.read_csv("../data/test.csv")

# # class_map.csv

# 3 component_type: `grapheme_root`, `vowel_diacritic`, `consonant_diacritic`
#
# `grapheme_root`: 168 unique labels
#
# `vowel_diacritic`: 11 unique labels
#
# `consonant_diacritic`: 7 unique labels

class_map_df.head()

class_map_df["component_type"].unique()

class_map_df[class_map_df["component_type"]=="grapheme_root"]

class_map_df[class_map_df["component_type"]=="vowel_diacritic"]

class_map_df[class_map_df["component_type"]=="consonant_diacritic"]

# # train.csv

train_df

# # test.csv

test_df

# # free explore
#

image0_df

# +
image_15_id = list(train_df[train_df["grapheme_root"]==2]["image_id"])

f, ax = plt.subplots(4, 5, figsize=(16, 8))
ax = ax.flatten()

for i in range(20):
    # print(image_15_id[i])
    image_np = image0_df[image0_df["image_id"]==image_15_id[i]].values[0][1:].astype(np.uint8).reshape(137,236)
    ax[i].imshow(image_np, cmap='Greys')

# -

train_df[train_df["grapheme_root"]==2]

# +
cond = (train_df["grapheme_root"]==0) & (train_df["vowel_diacritic"]==0) & (train_df["consonant_diacritic"]==0)

train_df[cond]

image_15_id = list(train_df[cond]["image_id"])

f, ax = plt.subplots(4, 5, figsize=(16, 8))
ax = ax.flatten()

for i in range(20):
    # print(image_15_id[i])
    image_np = image0_df[image0_df["image_id"]==image_15_id[i]].values[0][1:].astype(np.uint8).reshape(137,236)
    ax[i].imshow(image_np, cmap='Greys')
# -

# # split train/test evenly

train_df

train_df["combine"] = (
    train_df["grapheme_root"].apply(lambda x: str(x)) + "!" + 
    train_df["vowel_diacritic"].apply(lambda x: str(x)) + "!" +
    train_df["consonant_diacritic"].apply(lambda x: str(x)) 
)

train_df["combine"].nunique()

combine_count = train_df.groupby("combine")["image_id"].count()

combine_count.sort_values()


