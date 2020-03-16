import numpy as np

from sklearn.model_selection import StratifiedKFold


def random_split(num_train, test_ratio, random_seed):
    # random train test split
    indices = list(range(num_train))

    np.random.seed(random_seed)
    np.random.shuffle(indices)

    split = int(np.floor(test_ratio * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    return train_idx, valid_idx


def stratified_split_kfold(image_data, label_data, n_splits, random_seed):
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


def stratified_split_kfold_with_last_column(image_data, label_data, n_splits, random_seed):
    str_label = []
    for root, vowel, consonant, grapheme in label_data:
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
