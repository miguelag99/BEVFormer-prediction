from typing import Any, Dict

import torch
import torch.nn as nn
from .positional_encoding import LearnedPositionalEncoding
from einops import rearrange

from .bevformer_block import BEVFormerBlock

class BEVFormerTransformer(nn.Module):

    def __init__(
        self,
        num_layers: int = 6,
        embed_dim: int = 256,
        num_cameras: int = 6,
        img_size: tuple[int, int] = (224, 480),
        grid_shape: tuple[int, int, int] = (100, 100, 8),
        grid_ranges: list[float] = [-50.0, -50.0, -10.0, 50.0, 50.0, 10.0],
        sa_num_heads: int = 8,
        sa_num_levels: int = 1,
        sa_num_points: int = 4,
        sa_dropout: float = 0.1,
        ca_num_points: int = 8,
        ca_num_levels: int = 1,
        ca_dropout: float = 0.1,
        ffn_num_fcs: int = 2,
        ffn_dropout: float = 0.1,    
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.num_cameras = num_cameras
        self.img_size = img_size
        self.grid_shape = grid_shape
        self.grid_ranges = grid_ranges
        self.sa_num_heads = sa_num_heads
        self.sa_num_levels = sa_num_levels
        self.sa_num_points = sa_num_points
        self.sa_dropout = sa_dropout
        self.ca_num_points = ca_num_points
        self.ca_num_levels = ca_num_levels
        self.ca_dropout = ca_dropout
        self.ffn_num_fcs = ffn_num_fcs
        self.ffn_dropout = ffn_dropout
        
        self.bev_pos_encoding = LearnedPositionalEncoding(
            embed_dim // 2,
            *grid_shape[:2],
        )
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                BEVFormerBlock(
                    embed_dim=embed_dim,
                    num_cameras=num_cameras,
                    img_size=self.img_size,
                    grid_shape=grid_shape,
                    grid_ranges=grid_ranges,
                    sa_num_heads=sa_num_heads,
                    sa_num_levels=sa_num_levels,
                    sa_num_points=sa_num_points,
                    sa_dropout=sa_dropout,
                    ca_num_points=ca_num_points,
                    ca_num_levels=ca_num_levels,
                    ca_dropout=ca_dropout,
                    ffn_num_fcs=ffn_num_fcs,
                    ffn_dropout=ffn_dropout,
                )
            )

    def forward(
        self,
        bev_q: torch.Tensor,
        camera_feats: torch.Tensor,
        img_metas: torch.Tensor,
    ) -> torch.Tensor:
        
        """"
        Forward pass for the BEVFormer transformer module.
        Args:
            bev_q (torch.Tensor): BEV query tensor of shape (B, H*W, C) or (B, C, H, W).
            camera_feats (torch.Tensor or List[torch.Tensor]): Camera feature tensors 
                with shape (B, N_cam, C, H, W).
            img_metas (torch.Tensor).
        Returns:
            torch.Tensor: Output tensor after passing through the transformer layers.
        """
        
        # Set-up features.
        H, W, _ = self.grid_shape
        B = bev_q.shape[0]
        
        if len(bev_q.shape) == 4:
            x = rearrange(bev_q, 'b c h w -> b (h w) c')    # Rearrange if neccesary
        else:
            x = bev_q
        
        # TODO: compat. with more multi-scale feats.
        if isinstance(camera_feats, torch.Tensor):
            camera_feats = [camera_feats]
        elif isinstance(camera_feats, list):
            camera_feats = camera_feats
        else:
            raise ValueError('camera_feats must be a tensor or a list of tensors.')
        
        # Positional encoding.
        ignore_mask = torch.zeros((B, H, W), device=bev_q.device)
        bev_pos = rearrange(self.bev_pos_encoding(ignore_mask), 'b c h w -> b (h w) c')

        # Forward pass.
        for layer in self.layers:
            x = layer(x, bev_pos, camera_feats, img_metas)

        return x