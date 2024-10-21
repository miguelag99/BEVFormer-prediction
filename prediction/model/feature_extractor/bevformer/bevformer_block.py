import torch
import torch.nn as nn
from einops import rearrange
from mmengine.registry import MODELS

from .deformable_self_attention import DeformableSelfAttention
from .spatial_cross_attention import SpatialCrossAttention
from .utils import get_reference_points, point_sampling


class BEVFormerBlock(nn.Module):

    def __init__(
        self,
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
        
        self.self_attn = DeformableSelfAttention(
            embed_dim=embed_dim,
            num_heads=sa_num_heads,
            num_levels=sa_num_levels,
            num_points=sa_num_points,
            grid_shape=grid_shape[:2],
            dropout=sa_dropout,
        )
        self.norm_layer_1 = nn.LayerNorm(embed_dim)

        self.cross_attn = SpatialCrossAttention(
            embed_dims=embed_dim,
            num_cams=num_cameras,
            pc_range=grid_ranges,
            dropout=ca_dropout,
            init_cfg=None,
            batch_first=True,
            deformable_attention=dict(
                type='MSDeformableAttention3D',
                embed_dims=embed_dim,
                num_points=ca_num_points,
                num_levels=ca_num_levels,
            )
        )
        self.norm_layer_2 = nn.LayerNorm(embed_dim)

        self.ffn = MODELS.build(
            dict(
                type='FFN',
                embed_dims=embed_dim,
                feedforward_channels=2 * embed_dim,
                num_fcs=ffn_num_fcs,
                ffn_drop=ffn_dropout,
                act_cfg=dict(type='ReLU', inplace=True),
            )
        )
        self.norm_layer_3 = nn.LayerNorm(embed_dim)

        self.level_embeddings = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(ca_num_levels, embed_dim)),
            requires_grad=True
        )
        self.camera_embeddings = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(num_cameras, embed_dim)),
            requires_grad=True
        )


    def forward(
        self,
        x: torch.Tensor,
        bev_pos: torch.Tensor,
        camera_feats: list[torch.Tensor],
        img_metas: torch.Tensor,
    ) -> torch.Tensor:

        # Stage 1. Deformable self-attention.
        residual = x
        x = self.norm_layer_1(x)
        x = self.self_attn(x, bev_pos) + residual
        
        # Stage 2. Spatial cross-attention.
        spatial_shapes, camera_feats_flatten = [], []
        # Camera features must be a list of tensors with shape (B, N, C, H, W).
        for lvl, camera_feat in enumerate(camera_feats):
            H, W = camera_feat.shape[-2:]

            # Flatten the features.
            camera_feat_flatten = rearrange(camera_feat, 'b n c h w -> n b (h w) c')
            camera_feat_flatten = camera_feat_flatten + \
                rearrange(self.camera_embeddings, 'l c -> l 1 1 c')
            camera_feat_flatten = camera_feat_flatten + \
                rearrange(self.level_embeddings[lvl:lvl+1, :], 'l c -> l 1 1 c')

            camera_feats_flatten.append(camera_feat_flatten)
            spatial_shapes.append((H, W))

        camera_feats_flatten = torch.cat(camera_feats_flatten, dim=2)
        camera_feats_flatten = rearrange(camera_feats_flatten, 'n b hw c -> n hw b c')
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long,
                                         device=x.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)),
                                       spatial_shapes.prod(1).cumsum(0)[:-1]))

        # Create reference points.
        ref_3d = get_reference_points(
            self.grid_shape[0],
            self.grid_shape[1],
            self.grid_shape[2],
            num_points_in_pillar=4,
            dim='3d',
            bs=x.shape[0],
            device=x.device,
            dtype=torch.float32,
        )
        reference_points_cam, bev_mask = point_sampling(ref_3d,
                                                        self.grid_ranges,
                                                        img_metas,
                                                        self.img_size)

        residual = x
        x = self.norm_layer_2(x)
                
        x = self.cross_attn(
            query=x,
            key=camera_feats_flatten,
            value=camera_feats_flatten,
            identity=None,
            query_pos=None,
            key_pos=None,
            reference_points=ref_3d,
            reference_points_cam=reference_points_cam,
            mask=None,
            attn_mask=None,
            spatial_shapes=spatial_shapes,
            bev_mask=bev_mask,
            level_start_index=level_start_index,
        ) + residual
        
        # Stage 3. Feed-forward network.
        residual = x
        x = self.norm_layer_3(x)
        x = self.ffn(x) + residual

        return x