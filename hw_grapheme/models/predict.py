import cv2
import torch

import numpy as np
import pandas as pd

# from scipy.special import softmax
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from hw_grapheme.utils import load_model_weight

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


class GraphemeDataset_test(Dataset):
    def __init__(self, fname, transforms=None):
        df = pd.read_parquet(fname)
        self.name = df.iloc[:, 0]
        self.images = df.iloc[:, 1:].values
        self.images = self.process_image(self.images)
        self.transforms = transforms

    def process_image(self, images):
        """
        images: (batch_size, 32332), np array

        """
        image_max = images.max(axis=1).reshape(-1, 1).astype(np.uint8)
        images = (255 - images)/image_max*255
        images = images.reshape(-1, HEIGHT, WIDTH).astype(np.uint8)
        return images
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        name = self.name[idx]
        images = crop_resize(self.images[idx])
        if self.transforms is not None:
            images = self.transforms(images)
        
        # TODO: normalize images
        # images = (images.astype(np.float32)/255.0 - stats[0])/stats[1]
        return images, name
    

def predit_full_dataset_in_mini_batch(torch_model, dataloader):
    """
    return probit of size (dataset length, 168), (dataset length, 11), (dataset length, 7)
    
    """
    torch_model.eval()
    
    probit_root = []
    probit_vowel = []
    probit_consonant = []
    file_name = []

    # get probit (pred)
    with torch.no_grad():
        for image, image_name in tqdm(dataloader):
#             x = x.unsqueeze(1).cuda()
            image = image.cuda()
            root_logit, vowel_logit, consonant_logit = torch_model(image)

            # single model pred in a mini-batch
            probit_root.append(softmax(root_logit.cpu().numpy()))
            probit_vowel.append(softmax(vowel_logit.cpu().numpy()))
            probit_consonant.append(softmax(consonant_logit.cpu().numpy()))
            file_name.append(image_name)

    # full dataset
    probit_root = np.vstack(probit_root)
    probit_vowel = np.vstack(probit_vowel)
    probit_consonant = np.vstack(probit_consonant)
    
    assert probit_root.shape[1] == 168
    assert probit_vowel.shape[1] == 11
    assert probit_consonant.shape[1] == 7
    
    return probit_root, probit_vowel, probit_consonant, file_name


def predit_ensemble(model_archs_weights, dataloader):
    """
    model_archs_weights: list of (model_arch_func, weights_file_path)

    """
    model_results_root = []
    model_results_vowel = []
    model_results_consonant = []
    
    row_id,target = [],[]
    
    for i, (model_arch, model_weight) in enumerate(model_archs_weights):
        print(f"Loading model {i}...")
        
        # load model
        model = model_arch()
        
        # load weight into model (in-place)  
        load_model_weight(model, model_weight)
        
        model.to("cuda")
        
        # get prediction of the full dataset 
        probit_root, probit_vowel, probit_consonant, file_name = \
            predit_full_dataset_in_mini_batch(model, dataloader)
        model_results_root.append(probit_root)
        model_results_vowel.append(probit_vowel)
        model_results_consonant.append(probit_consonant)
        
    # get final pred = highest summed probit 
    probit_root_sum = model_results_root[0].copy()
    for r in model_results_root[1:]:
        probit_root_sum += r
        
    probit_vowel_sum = model_results_vowel[0].copy()
    for r in model_results_vowel[1:]:
        probit_vowel_sum += r
        
    probit_consonant_sum = model_results_consonant[0].copy()
    for r in model_results_consonant[1:]:
        probit_consonant_sum += r
    
    pred_ensemble_root = probit_root_sum.argmax(axis=1)
    pred_ensemble_vowel = probit_vowel_sum.argmax(axis=1)
    pred_ensemble_consonant = probit_consonant_sum.argmax(axis=1)
    
    # turn pred into df
    file_name = [x for batch_x in file_name for x in batch_x]
    for idx, name in enumerate(file_name):
        row_id += [f'{name}_grapheme_root',f'{name}_vowel_diacritic',
                   f'{name}_consonant_diacritic']
        target += [pred_ensemble_root[idx],pred_ensemble_vowel[idx],pred_ensemble_consonant[idx]]
        
    return row_id, target


def bengali_predict(test_data_paths, model_archs_weights, transforms, batch_size=128, n_workers=2):
    """
    test_data_paths: list of paths to test_image_data_x.parquet
    model_archs_weights: list of (model_arch_func, weights_file_path)
    
    """
    row_id, target = [], []
    
    for test_data_path in test_data_paths:     
        dataset = GraphemeDataset_test(test_data_path, transforms)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=n_workers, shuffle=False)
        
        row_id_subset, target_subset = predit_ensemble(model_archs_weights, dataloader)
        row_id += row_id_subset
        target += target_subset
    return row_id, target
