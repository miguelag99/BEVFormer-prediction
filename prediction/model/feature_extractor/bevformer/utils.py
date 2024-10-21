import torch
import numpy as np

from mmengine.utils.dl_utils import TORCH_VERSION
from mmengine.utils import digit_version



def get_reference_points(H, W, Z=8, num_points_in_pillar=4, dim='3d', bs=1,
                         device='cuda', dtype=torch.float):
        """Get the reference points used in SCA and TSA.
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        # reference points in 3D space, used in spatial cross-attention (SCA)
        if dim == '3d':
            zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype,
                                device=device).view(-1, 1, 1).expand(
                                    num_points_in_pillar, H, W) / Z
            xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                                device=device).view(1, 1, W).expand(
                                    num_points_in_pillar, H, W) / W
            ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                                device=device).view(1, H, 1).expand(
                                    num_points_in_pillar, H, W) / H
            ref_3d = torch.stack((xs, ys, zs), -1)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
            return ref_3d

        # reference points on 2D bev plane, used in temporal self-attention (TSA).
        elif dim == '2d':
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=dtype, device=device),
                indexing='ij'
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_2d = torch.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d
        
        
def point_sampling(reference_points, pc_range,  img_metas, img_shape=(224,480)):
    # NOTE: close tf32 here.
    allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    lidar2img = reference_points.new_tensor(img_metas)  # (B, N, 4, 4)
    lidar2img = lidar2img.squeeze(0)
    reference_points = reference_points.clone()

    reference_points[..., 0:1] = reference_points[..., 0:1] * \
        (pc_range[3] - pc_range[0]) + pc_range[0]
    reference_points[..., 1:2] = reference_points[..., 1:2] * \
        (pc_range[4] - pc_range[1]) + pc_range[1]
    reference_points[..., 2:3] = reference_points[..., 2:3] * \
        (pc_range[5] - pc_range[2]) + pc_range[2]

    reference_points = torch.cat(
        (reference_points, torch.ones_like(reference_points[..., :1])), -1)
    
    reference_points = reference_points.permute(1, 0, 2, 3)
    D, B, num_query = reference_points.size()[:3]
    num_cam = lidar2img.size(1)

    reference_points = reference_points.view(
        D, B, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)

    lidar2img = lidar2img.view(
        1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, num_query, 1, 1)

    reference_points_cam = torch.matmul(lidar2img.to(torch.float32),
                                        reference_points.to(torch.float32)).squeeze(-1)
    eps = 1e-5

    bev_mask = (reference_points_cam[..., 2:3] > eps)
    reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
        reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps)

    reference_points_cam[..., 0] /= img_shape[1]
    reference_points_cam[..., 1] /= img_shape[0]

    bev_mask = (bev_mask & (reference_points_cam[..., 1:2] > 0.0)
                & (reference_points_cam[..., 1:2] < 1.0)
                & (reference_points_cam[..., 0:1] < 1.0)
                & (reference_points_cam[..., 0:1] > 0.0))
        
    if digit_version(TORCH_VERSION) >= digit_version('1.8'):
        bev_mask = torch.nan_to_num(bev_mask)
    else:
        bev_mask = bev_mask.new_tensor(
            np.nan_to_num(bev_mask.cpu().numpy()))

    reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)
    bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1)

    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32

    return reference_points_cam, bev_mask