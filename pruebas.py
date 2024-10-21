import torch
import time

import numpy as np

import pdb

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from prediction.model.predictor import BEVFormerPredictor
from prediction.config import get_config
from prediction.data.prepare_loader import prepare_dataloaders

from prediction.model.feature_extractor.bevformer.utils import get_reference_points

from einops import rearrange

from tqdm import tqdm

if __name__ == '__main__':
    
        
    cfg, hparams = get_config(cfg_dir='./prediction/configs/', 
                     cfg_file='bevformer_tiny_short')
    model = BEVFormerPredictor(cfg).to('cuda')# .feature_extractor.to('cuda')
    
    
    cfg.DATASET.VERSION = "v1.0-mini"
    cfg.N_WORKERS = 0

    _ , valloader = prepare_dataloaders(cfg, return_dataset=False,
                                        return_orig_images=True, return_pcl=False)
    
    receptive_field = 3
    
    data = next(iter(valloader))
    image = data['image'][:, :receptive_field].contiguous().to('cuda')
    full_image = data['orig_image'][:, :receptive_field].contiguous()
    intrinsics = data['intrinsics'][:, :receptive_field].contiguous().to('cuda')
    extrinsics = data['extrinsics'][:, :receptive_field].contiguous().to('cuda')
    future_egomotion = data['future_egomotion'][:, :receptive_field].contiguous().to('cuda')
    lidar_ego_2_img = data['lidar_ego_2_img'][:, :receptive_field].contiguous().to('cuda')
    
    B, S, N, _, _, _= image.shape
    
    # image = image[0,0,1]
    # extrinsics = extrinsics[0,0,1]
    # intrinsics = intrinsics[0,0,1]
    # lidar_to_sensor = lidar_ego_2_img[0,0,1]
    
    
    # (Pdb) reference_points.shape
    # torch.Size([6, 4, 40000, 3])
    # (Pdb) lidar2img.shape
    # torch.Size([6, 6, 4, 4])
            
    pdb.set_trace()

    out = model(data['image'].contiguous().to('cuda'),
                data['lidar_ego_2_img'].contiguous().to('cuda'),
                data['future_egomotion'].contiguous().to('cuda'))
    
    # Check for depth cannels information
    

    del model
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
