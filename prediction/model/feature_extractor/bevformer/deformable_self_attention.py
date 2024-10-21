from typing import Optional

import torch
import torch.nn as nn
from mmcv.ops import MultiScaleDeformableAttention


class DeformableSelfAttention(nn.Module):

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_levels: int = 1,
        num_points: int = 4,
        grid_shape: tuple[int, int] = (200, 200),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.H, self.W = grid_shape
        self.dropout = dropout

        self.attn = MultiScaleDeformableAttention(
            embed_dims=embed_dim,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
            batch_first=True,
            dropout=dropout,
        )

    def _get_reference_points(
        self,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:

        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, self.H - 0.5, self.H, dtype=torch.float, device=device),
            torch.linspace(0.5, self.W - 0.5, self.W, dtype=torch.float, device=device),
            indexing='ij'
        )
        ref_y = ref_y.reshape(-1)[None] / self.H
        ref_x = ref_x.reshape(-1)[None] / self.W
        reference_points = torch.stack((ref_x, ref_y), -1)
        reference_points = reference_points.repeat(batch_size, 1, 1).unsqueeze(2)  # noqa: (B, N, 1, 2)
        return reference_points

    def forward(
        self,
        x: torch.Tensor,
        x_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of the DeformableSelfAttention module.

        Args:
            x (torch.Tensor): Query tensor. BEV learned embeddings or 
                radar feature maps.
            x_pos (Optional[torch.Tensor], optional): Positional
                encoding. Defaults to None.

        Returns:
            torch.Tensor: Self-attention output.
        """
        B, _, _ = x.shape
        ref_points = self._get_reference_points(B, x.device)

        # Same as BEVFormer block, but without BEV history.
        spatial_shapes = torch.tensor(
            [[self.H, self.W]] * self.num_levels,
            device=x.device
        )
        level_start_index = torch.tensor([0], device=x.device)

        return self.attn(
            query=x,
            value=x,
            identity=None,
            query_pos=x_pos,
            reference_points=ref_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
        )