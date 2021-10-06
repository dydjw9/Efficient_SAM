import random
import torch
import numpy as np


def initialize(args, seed: int):
    random.seed(seed)
    np.random.seed(seed + args.local_rank)
    torch.manual_seed(seed+args.local_rank)
    torch.cuda.manual_seed_all(seed+args.local_rank)

    torch.backends.cudnn.enabled = True
    #set False to reduce randomiess
    torch.backends.cudnn.benchmark = True
    #set true to make training deterministic
    torch.backends.cudnn.deterministic = False
def initialize_bk(args, seed: int):
    random.seed(seed)
    np.random.seed(seed )
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    #set False to reduce randomiess
    torch.backends.cudnn.benchmark = True
    #set true to make training deterministic
    torch.backends.cudnn.deterministic = False
