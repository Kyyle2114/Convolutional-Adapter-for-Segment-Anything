import random
import numpy as np
import os
import torch

def seed_everything(seed = 21):
    """
    Set seed

    Args:
        seed (int, optional): random seed. Defaults to 21.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = True  