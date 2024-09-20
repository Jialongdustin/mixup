import torch
import numpy as np

picked_new_cls=torch.Tensor(np.random.choice(89 - 59, 128) + 59).long()
print(picked_new_cls)