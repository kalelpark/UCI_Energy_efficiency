import enum
import math
import time
from copy import deepcopy
import warnings
import typing as ty
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch import Tensㅠor
from .fttransformer import Transformer

def load_model(args, config):
    if config["model"] == "ft-transformer":
        return Transformer(
                        d_numerical = 8,
                        categories = None,

                        # Model Architecture
                        n_layers = int(config["n_layers"]),
                        n_heads = int(config["n_heads"]),
                        d_token = int(config["d_token"]),
                        d_ffn_factor = float(config["d_ffn_factor"]),
                        attention_dropout = float(config["attention_dropout"]),
                        ffn_dropout = float(config["attention_dropout"]),
                        residual_dropout = float(config["residual_dropout"]),
                        activation = config["activation"],
                        prenormalization = True,
                        initialization = config["initialization"],
                        
                        # default_Setting
                        token_bias = True,
                        kv_compression = None if int(config["kv_compression"]) == 0 else int(config["kv_compression"]),
                        kv_compression_sharing= None if int(config["kv_compression"]) == 0 else float(config["kv_compression"]),
                        d_out = 1
        )
    else:
        pass