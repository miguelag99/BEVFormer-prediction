import torch
import torch.nn as nn

from prediction.model.feature_extractor.bevformer import BEVFormerTransformer
from prediction.model.img_encoder import (EncoderEfficientNet, ResNetFPNEncoder,
                                          TimmFPNEncoder)
from prediction.utils.geometry import (VoxelsSumming,
                                       calculate_birds_eye_view_parameters,
                                       cumulative_warp_features)

class BEVFormerFeatureExtractor(nn.Module):
    def __init__(self,
                 x_bound = (-50.0, 50.0, 0.5),  # Forward
                 y_bound = (-50.0, 50.0, 0.5),  # Sides
                 z_bound = (-10.0, 10.0, 20.0),  # Height
                 d_bound = (2.0, 50.0, 1.0),
                 downsample: int = 8,
                 out_channels: int = 64,
                 receptive_field: int = 3,
                 use_depth_distribution: bool = True,
                 model_name: str = 'efficientnet-b0',
                 img_size = (224,480),
                 num_layers: int = 6,
                 embed_dim: int = 256,
                 num_cameras: int = 6,
                 grid_shape: tuple[int, int, int] = (100, 100, 8),
                 grid_ranges: list[float] = [-50.0, -50.0, -10.0, 50.0, 50.0, 10.0],
                 sa_num_heads: int = 8,
                 sa_num_levels: int = 1,
                 sa_num_points: int = 4,
                 sa_dropout: float = 0.1,
                 ca_num_points: int = 8,
                 ca_num_levels: int = 1,
                 ca_dropout: float = 0.1,
                 ca_im2col_step: int = 64,
                 ffn_num_fcs: int = 2,
                 ffn_dropout: float = 0.1,  
                 ):
        super().__init__()

        self.bounds = {
            'x': x_bound,
            'y': y_bound,
            'z': z_bound,
            'd': d_bound
        }

        bev_resolution, bev_start_position, bev_dimension = \
            calculate_birds_eye_view_parameters(x_bound, y_bound, z_bound)

        self.bev_resolution = nn.Parameter(bev_resolution, requires_grad=False)
        self.bev_start_position = nn.Parameter(bev_start_position, requires_grad=False)
        self.bev_dimension = nn.Parameter(bev_dimension, requires_grad=False)

        self.img_final_dim = img_size

        self.encoder_downsample = downsample
        self.encoder_out_channels = out_channels

        # temporal block
        self.receptive_field = receptive_field

        # Spatial extent in bird's-eye view, in meters
        self.spatial_extent = (x_bound[1], y_bound[1])
        self.bev_size = (self.bev_dimension[0].item(), self.bev_dimension[1].item())

        # Define the camera multi-sweep encoder. Depth distribution not needed.
        if 'efficientnet' in model_name.lower():
            self.encoder = EncoderEfficientNet(
                out_channels=out_channels,
                depth_distribution=False,
                downsample=downsample,
                model_name=model_name,
            )
        
        elif 'resnet' in model_name.lower():
            self.encoder = ResNetFPNEncoder(
                feature_extractor=model_name,
                out_channels=out_channels,
                downsample=downsample,
                depth_distribution=False,
            )
        
        elif 'mobilenet' or 'convnext' in model_name.lower():
            self.encoder = TimmFPNEncoder(
                feature_extractor=model_name,
                out_channels=out_channels,
                downsample=downsample,
                depth_distribution=False,
            )
        else:
            raise ValueError(f'Encoder model {model_name} not handled.')
        
        # Define the BEVFormer transformer
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.num_cameras = num_cameras
        self.grid_shape = grid_shape
        self.grid_ranges = grid_ranges
        self.sa_num_heads = sa_num_heads
        self.sa_num_levels = sa_num_levels
        self.sa_num_points = sa_num_points
        self.sa_dropout = sa_dropout
        self.ca_num_points = ca_num_points
        self.ca_num_levels = ca_num_levels
        self.ca_dropout = ca_dropout
        self.ca_im2col_step = ca_im2col_step
        self.ffn_num_fcs = ffn_num_fcs
        self.ffn_dropout = ffn_dropout
        
        self.bevformer = BEVFormerTransformer(
            num_layers = num_layers,
            embed_dim = embed_dim,
            num_cameras = num_cameras,
            img_size=img_size,
            grid_shape = grid_shape,
            grid_ranges = grid_ranges,
            sa_num_heads = sa_num_heads,
            sa_num_levels = sa_num_levels,
            sa_num_points = sa_num_points,
            sa_dropout = 0.1,
            ca_num_points = ca_num_points,
            ca_num_levels = ca_num_levels,
            ca_dropout = 0.1,
            ca_im2col_step=ca_im2col_step,
            ffn_num_fcs = ffn_num_fcs,
            ffn_dropout = 0.1,    
        )
        self.bev_embed = nn.Embedding(grid_shape[0]*grid_shape[1],embed_dim)


    def forward(self, image, future_egomotion, img_metas):
        '''
        Inputs:
            image: (b, sweeps, n_cam, channel, h, w)
            future_egomotion: (b, sweeps, 6)
            img_metas (dict): Image metadata with 3D to camera projection matrices
        '''
        # Only process features from the past and present (within receptive field)
        image = image[:, :self.receptive_field].contiguous()
        future_egomotion = future_egomotion[:, :self.receptive_field].contiguous()
        img_metas = img_metas[:, :self.receptive_field].contiguous()

        x = self.bev_features(image, img_metas) # (b, s, c, h, w)
        
        # Warp past features to the present's reference frame
        x = cumulative_warp_features(
            x.clone(), future_egomotion,
            mode='bilinear', spatial_extent=self.spatial_extent,
        )

        return x

    
    def bev_features(self, x, img_metas):
        '''
        Inputs:
            x: (b, sweeps, n_cam, channel, h, w)
            img_metas (dict): Image metadata with 3D to camera projection matrices
        '''
        
        b, s, n, c, h, w = x.shape
                
        x = x.view(b * s * n, *x.shape[3:])    
        x = self.encoder(x)       
        x = x.view(b*s, n, *x.shape[1:])
        
        bev_queries = self.bev_embed.weight.to(x.dtype)
        bev_queries = bev_queries.unsqueeze(0).expand(b*s,-1,-1) # Expand to Batch
               
        img_metas = img_metas.view(b * s, *img_metas.shape[2:])

        x = self.bevformer(bev_queries, x, img_metas)
        
        # Unpack sequence dim reshape to BEV map shape
        x = x.view(b,s,self.embed_dim,self.grid_shape[0],self.grid_shape[1])        

        return x
   


