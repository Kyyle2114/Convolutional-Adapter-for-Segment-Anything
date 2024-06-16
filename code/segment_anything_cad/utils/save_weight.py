from typing import Type, List

import torch
import torch.nn as nn 

def save_partial_weight(model: Type[nn.Module], 
                        save_path: str = 'Best_Decoder.pth') -> List:
    """
    Saves the weights of trainable parameters.

    Args:
        model (Type[nn.Module]): SAM
        save_path (str, optional): path to save. Defaults to 'Best_Decoder.pth'.

    Returns:
        List: trainable parameter name list
    """
    trainable_param = []
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            trainable_param.append(name)
    
    state_dict_trainable = {k: v for k, v in model.state_dict().items() if k in trainable_param}
    torch.save(state_dict_trainable, save_path)
    
    return trainable_param

def load_partial_weight(model: Type[nn.Module], 
                        load_path: str = 'Best_Decoder.pth',
                        dist: bool = True) -> Type[nn.Module]:
    """
    Loads the weights of trainable parameters.
    
    Args:
        model (Type[nn.Module]): SAM
        load_path (str, optional): path to load. Defaults to 'Best_Decoder.pth'.
        dist (bool, optional): If you trained the model on multi-GPU, pass True. Defaults to True.

    Returns:
        Type[nn.Module]: model with updated parameters
    """
    state_dict_trainable = torch.load(load_path)
    
    model_state_dict = model.state_dict()
    
    for k, v in state_dict_trainable.items():
        if dist:
            # module.mask_decoder.* -> mask_decoder.* 
            k = k[7:]
        model_state_dict[k] = v

    model.load_state_dict(model_state_dict)
    
    return model