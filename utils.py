import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import typing as ty
from torch import Tensor
import random
import numpy as np
from sklearn.metrics  import accuracy_score, mean_squared_error

def save_mode_with_json(model, config, args, count) -> None:

    if args.target == 1:
        output_path = "y1"
    else:
        output_path = "y2"

    with open(os.path.join(output_path + "_" + args.savepath, "info" + str(count) + ".json"), "w", encoding = "utf-8") as make_file:
        json.dump(config, make_file, ensure_ascii = False, indent = "\t")

    torch.save(model.module.state_dict(), os.path.join(output_path + "_" + args.savepath, "model" + str(count) + ".pt"))

def seed_everything(seed):     # set seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_rmse_score(y_pred : ty.List[np.ndarray], y_label : ty.List[np.ndarray]) -> float:
    return (mean_squared_error(y_pred, y_label) ** 0.5)

def get_loss(args) -> Tensor:
    
    """
    The loss function used varies depending on the type of task.
    Binaryclass using binary_crossentropy
    but, multicass using cross_entropy
    """
    
    if args.task == "binclass":
        return F.binary_cross_entropy_with_logits
    elif args.task == "multiclass":
        return F.cross_entropy
    else:
        return F.mse_loss

def get_optimizer(model, config : ty.Dict[str, str]) -> optim:
    
    """
    rtdl using default optim AdamW
    if you want change, see run yaml
    """

    if config["optim"] == "AdamW":
        return torch.optim.Adam(
            model.parameters(),
            lr = float(config["lr"]),
            weight_decay = float(config["weight_decay"]),
            eps = 1e-8
        )
    elif config["optim"] == "SGD":
        return torch.optim.SGD(
            model.parameters(),
            lr = float(config["lr"]),
            momentum=0.9,
            weight_decay = float(config["weight_decay"]),
        )
    else:
        pass