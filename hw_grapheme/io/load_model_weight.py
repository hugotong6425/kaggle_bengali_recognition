import torch

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