from fttransformer import Transformer
import torch
import numpy as np
import pandas as pd


X_data = torch.randn(10, 8)

model = Transformer(
                        d_numerical = 8,
                        categories = None,

                        # Model Architecture
                        n_layers = 3,
                        n_heads = 8,
                        d_token = 192,
                        d_ffn_factor = 1.3333333333333,
                        attention_dropout = 0.2,
                        ffn_dropout = 0.1,
                        residual_dropout = 0.0,
                        activation = "reglu",
                        prenormalization = True,
                        initialization = "kaiming",
                        
                        # default_Setting
                        token_bias = True,
                        kv_compression = None,
                        kv_compression_sharing= None,
                        d_out = 1
        )

output = model(x_num = X_data, x_cat = None)
print(output)
print(output.size())
# print(model)

