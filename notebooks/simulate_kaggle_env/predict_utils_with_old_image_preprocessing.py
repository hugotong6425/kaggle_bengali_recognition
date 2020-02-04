import cv2
import gc
import torch

import numpy as np
import pandas as pd

# from scipy.special import softmax
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from collections import OrderedDict


def load_model_weight(model, weight_path):
    # original saved file with DataParallel
    state_dict = torch.load(weight_path)
    
    # create new OrderedDict that does not contain `module.`
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[:7] == "module.":
            name = k[7:] # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict)

def softmax(array_2d):
    """
    array_2d (batch_size, dim)
    
    """
    arr_exp = np.exp(array_2d)
    return arr_exp/np.sum(arr_exp, axis=1, keepdims=True)


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
    img = img0[ymin:ymax,xmin:xmax]
    #remove lo intensity pixels as noise
    img[img < 28] = 0
    lx, ly = xmax-xmin,ymax-ymin
    l = max(lx,ly) + pad
    #make sure that the aspect ratio is kept in rescaling
    img = np.pad(img, [((l-ly)//2,), ((l-lx)//2,)], mode='constant')
    return cv2.resize(img,(size,size))

"""
class GraphemeDataset_test(Dataset):
    def __init__(self, images, transforms=None):
        self.images = images
        self.transforms = transforms
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        images = self.images[idx]
        
        image_max = images.max().reshape(-1).astype(np.uint8)
        images = (255 - images)/image_max*255
        images = images.astype(np.uint8)
        
        images = crop_resize(images)

        if self.transforms is not None:
            images = self.transforms(images)

        return images    
"""
class GraphemeDataset(Dataset):
    def __init__(self, fname):
        self.df = pd.read_parquet(fname)
        self.data = 255 - self.df.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH).astype(np.uint8)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        name = self.df.iloc[idx,0]
        #normalize each image by its max val
        img = (self.data[idx]*(255.0/self.data[idx].max())).astype(np.uint8)
        img = crop_resize(img)
        img = (img.astype(np.float32)/255.0 - stats[0])/stats[1]
        return img, name
    
def predit_full_dataset_in_mini_batch(torch_model, dataloader):
    """
    return probit of size (dataset length, 168), (dataset length, 11), (dataset length, 7)
    
    """
    torch_model.eval()
    
    probit_root = []
    probit_vowel = []
    probit_consonant = []

    # get probit (pred)
    with torch.no_grad():
        for image in dataloader:
            image = image.cuda()
            root_logit, vowel_logit, consonant_logit = torch_model(image)

            # single model pred in a mini-batch
            probit_root.append(softmax(root_logit.cpu().numpy()))
            probit_vowel.append(softmax(vowel_logit.cpu().numpy()))
            probit_consonant.append(softmax(consonant_logit.cpu().numpy()))

    # full dataset
    probit_root = np.vstack(probit_root)
    probit_vowel = np.vstack(probit_vowel)
    probit_consonant = np.vstack(probit_consonant)
    
    assert probit_root.shape[1] == 168
    assert probit_vowel.shape[1] == 11
    assert probit_consonant.shape[1] == 7
    
    return probit_root, probit_vowel, probit_consonant


def predit_ensemble(model_archs_weights, dataloader, file_name):
    """
    model_archs_weights: list of (model_arch_func, weights_file_path)

    """
    model_results_root = []
    model_results_vowel = []
    model_results_consonant = []
    
    row_id, target = [], []
    
    for i, (model_arch, model_weight) in enumerate(model_archs_weights):
        print(f"Loading model {i}...")
        
        # load model
        model = model_arch().to("cuda")
        
        # load weight into model (in-place)  
        load_model_weight(model, model_weight)
                
        # get prediction of the full dataset 
        probit_root, probit_vowel, probit_consonant = \
            predit_full_dataset_in_mini_batch(model, dataloader)
        
        del model
        print("delete model")
        print(gc.collect())
        print()
        
        model_results_root.append(probit_root)
        model_results_vowel.append(probit_vowel)
        model_results_consonant.append(probit_consonant)
        
        del probit_root
        del probit_vowel
        del probit_consonant
        print("delete subset probit")
        print(gc.collect())
        print()
        
        
    # get final pred = highest summed probit 
    probit_root_sum = model_results_root[0]
    for r in model_results_root[1:]:
        probit_root_sum += r
        
    probit_vowel_sum = model_results_vowel[0]
    for r in model_results_vowel[1:]:
        probit_vowel_sum += r
        
    probit_consonant_sum = model_results_consonant[0]
    for r in model_results_consonant[1:]:
        probit_consonant_sum += r
        
    del model_results_root
    del model_results_vowel
    del model_results_consonant
    print("delete probit")
    print(gc.collect())
    print()
    
    pred_ensemble_root = probit_root_sum.argmax(axis=1)
    pred_ensemble_vowel = probit_vowel_sum.argmax(axis=1)
    pred_ensemble_consonant = probit_consonant_sum.argmax(axis=1)
    
    # turn pred into df
    for idx, name in enumerate(file_name):
        row_id += [f'{name}_grapheme_root',f'{name}_vowel_diacritic',
                   f'{name}_consonant_diacritic']
        target += [pred_ensemble_root[idx],pred_ensemble_vowel[idx],pred_ensemble_consonant[idx]]
        
    return row_id, target
    
    
def process_data(test_data_path):
    """
    DO NOT DO NORMALIZATION STEP (will convert data to float)
    
    """
    df = pd.read_parquet(test_data_path)
    file_name = df.iloc[:, 0]
    images = df.iloc[:, 1:].values
    # image_max = images.max(axis=1).reshape(-1, 1).astype(np.uint8)
    # images = (255 - images)/image_max*255
    images = images.reshape(-1, HEIGHT, WIDTH).astype(np.uint8)
    
#     processed_image_list = []
#     print("processing images")
#     for idx in range(len(images)):
#         processed_image_list.append(crop_resize(images[idx]))
    # return file_name, np.array(processed_image_list)
    return file_name, images
    

def bengali_predict(test_data_paths, model_archs_weights, transforms, batch_size=128, n_workers=2):
    """
    test_data_paths: list of paths to test_image_data_x.parquet
    model_archs_weights: list of (model_arch_func, weights_file_path)
    
    """
    row_id, target = [], []        
    
    for test_data_path in test_data_paths:
        print("processing images")
        file_name, images = process_data(test_data_path)
        
        print(f"Doing data_path: {test_data_path}")
        dataset = GraphemeDataset_test(images, transforms)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=n_workers, shuffle=False)
        
        row_id_subset, target_subset = predit_ensemble(model_archs_weights, dataloader, file_name)

        row_id += row_id_subset
        target += target_subset
        
        del row_id_subset
        del target_subset
        print("delete row id taget subset")
        print(gc.collect())
        print()
        
        del file_name
        del images
        print("delete file name and images")
        print(gc.collect())
        print()
        
    return row_id, target

# def weagewa_predict_ensemble():
#     TEST = [
#     #      '../data/train_image_data_0.parquet',
#     #      '../data/train_image_data_2.parquet',
#         'test_image_data_0.parquet',
#         'test_image_data_1.parquet',
#         'test_image_data_2.parquet',
#         'test_image_data_3.parquet'
#     ]
#     model_names = ['model_0.pth', 'model_1.pth', 'model_2.pth', 'model_3.pth']
#     # model_dir = "/kaggle/input/bengalicv19trainedmodels"
#     # data_dir = "/kaggle/input/bengaliai-cv19"

#     bs = 128

#     row_id,target = [],[]

#     for fname in TEST:
#         # get dataloader
#         data_path = os.path.join(data_dir, fname)
#         ds = GraphemeDataset(data_path)
#         dl = DataLoader(ds, batch_size=bs, num_workers=nworkers, shuffle=False)

#         model_results_root = []
#         model_results_vowel = []
#         model_results_consonant = []

#         for i, model_name in enumerate(model_names):
#             # load model
#             print(f"Loading model {i}...")
#             model = Dnet_1ch(pre=False).cuda()
#             model_path = os.path.join(model_dir, model_name)
#             load_parallel_trained_model(model, model_path)

#             # get full dataset, multi-model pred
#             probit_root, probit_vowel, probit_consonant, file_name = predit_in_mini_batch(model, dl)
#             model_results_root.append(probit_root)
#             model_results_vowel.append(probit_vowel)
#             model_results_consonant.append(probit_consonant)

#         # get final pred = highest summed probit 
#         probit_root_sum = model_results_root[0].copy()
#         for r in model_results_root[1:]:
#             probit_root_sum += r

#         probit_vowel_sum = model_results_vowel[0].copy()
#         for r in model_results_vowel[1:]:
#             probit_vowel_sum += r

#         probit_consonant_sum = model_results_consonant[0].copy()
#         for r in model_results_consonant[1:]:
#             probit_consonant_sum += r

#         pred_ensemble_root = probit_root_sum.argmax(axis=1)
#         pred_ensemble_vowel = probit_vowel_sum.argmax(axis=1)
#         pred_ensemble_consonant = probit_consonant_sum.argmax(axis=1)

#         # turn pred into df
#         file_name = [x for batch_x in file_name for x in batch_x]
#         for idx, name in enumerate(file_name):
#             row_id += [f'{name}_grapheme_root',f'{name}_vowel_diacritic',
#                        f'{name}_consonant_diacritic']
#             target += [pred_ensemble_root[idx],pred_ensemble_vowel[idx],pred_ensemble_consonant[idx]]