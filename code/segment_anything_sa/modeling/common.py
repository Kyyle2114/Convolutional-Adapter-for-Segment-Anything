# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
ADD
SAM Adapter
"""

import torch
import torch.nn as nn

from typing import Type

class Adapter(nn.Module):
    """
    Adapter for SAM, Image encoder(ViT)
    """
    def __init__(self,
                 in_dim: int,
                 hidden_dim: int,
                 act_layer: Type[nn.Module] = nn.GELU,
                 skip_connection: bool = True,
                 up_projection: Type[nn.Module] = None,
                 ) -> None:
        """
        Args:
            in_dim (int): Number of input channels.
            hidden_dim (int): Number of hidden channels.
            act_layer (nn.Module): Activation layer.
            skip_connection (bool): if True, skip connection will be applied. 
            up_projection (nn.Module): Shared linear layer for up-projection in Task Specific information Adapter.
        """
        super(Adapter, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.skip_connection = skip_connection
        
        self.down_projection = nn.Linear(in_dim, hidden_dim)
        self.act_layer = act_layer()
        # shared up projection layer
        self.up_projection = nn.Linear(hidden_dim, in_dim) if up_projection is None else up_projection
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # N: batch_size
        # e.g., Tensor (N, 64, 64, 768) -> (N, 4096, 768)
        x = x.view(-1, self.in_dim)
        shortcut = x
        
        x = self.act_layer(self.down_projection(x))
        x = self.up_projection(x) 
        
        # skip connection
        if self.skip_connection:
            x  = x + shortcut    
                
        # e.g., Tensor (N, 4096, 768) -> (N, 64, 64, 768) 
        x = x.view(-1, 64, 64, self.up_projection.out_features)
        
        return x

class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
