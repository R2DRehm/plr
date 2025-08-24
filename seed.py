
import os
import random
import numpy as np
try:
    import torch
except Exception:
    torch = None

def set_seed(seed: int = 42, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            try:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass
    os.environ["PYTHONHASHSEED"] = str(seed)
