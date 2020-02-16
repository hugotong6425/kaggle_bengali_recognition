import pickle

import numpy as np


def load_raw_data(pickle_paths, image_size):
    pass

def load_processed_data(pickle_paths, image_size):
    # load data from pickle
    image_data = []
    name_data = []
    label_data = []

    for pickle_path in pickle_paths:
        with open(pickle_path, "rb") as f:
            train_data = pickle.load(f)
            image_data.append(train_data[0])
            name_data.append(train_data[1])
            label_data.append(train_data[2])

    image_data = np.array(image_data)
    name_data = np.array(name_data)
    label_data = np.array(label_data)

    # print(image_data.shape, name_data.shape, label_data.shape)

    image_data = image_data.reshape(image_data.shape[0]*image_data.shape[1], image_size, image_size)
    name_data = name_data.reshape(-1)
    label_data = label_data.reshape(label_data.shape[0]*label_data.shape[1], 3)

    print(f"Load data done, shape: {image_data.shape}, {name_data.shape}, {label_data.shape}")
    return image_data, name_data, label_data

