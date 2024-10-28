from types import SimpleNamespace

from prediction.configs.baseline import baseline_cfg

model_cfg = baseline_cfg

model_cfg.LIFT = SimpleNamespace(
        # Short BEV dimensions
        X_BOUND = [-15.0, 15.0, 0.15],  # Forward
        Y_BOUND = [-15.0, 15.0, 0.15],  # Sides
        Z_BOUND = [-10.0, 10.0, 20.0],  # Height
        D_BOUND = [2.0, 50.0, 1.0],
)

model_cfg.MODEL = SimpleNamespace(

        STCONV = SimpleNamespace(
            INPUT_EGOPOSE = True,
        ),    
        
        ENCODER = SimpleNamespace(
            DOWNSAMPLE = 8,
            NAME = 'efficientnet-b4',
            OUT_CHANNELS = 64,
            USE_DEPTH_DISTRIBUTION = True,
        ),
        
        FEATURE_EXTRACTOR = SimpleNamespace(
            NAME = 'bevformer',
            N_LAYERS = 2,
            EMBED_DIM = 64,
            SA_NUM_HEADS = 4,
            SA_NUM_LEVELS = 1,
            SA_NUM_POINTS = 4,
            CA_NUM_LEVELS = 1,
            CA_NUM_POINTS = 8,
            CA_IM2COL_STEP = 256,
            FFN_NUM_FCS = 2,
        ),
                
        # Tiny
        SEGFORMER = SimpleNamespace(
            N_ENCODER_BLOCKS = 5,
            DEPTHS = [2, 2, 2, 2, 2],
            SEQUENCE_REDUCTION_RATIOS = [8, 4, 2, 1, 1],
            HIDDEN_SIZES = [16, 24, 32, 48, 64], 
            PATCH_SIZES = [7, 3, 3, 3, 3],
            STRIDES = [2, 2, 2, 2, 2],
            NUM_ATTENTION_HEADS = [1, 2, 4, 8, 8],
            MLP_RATIOS = [4, 4, 4, 4, 4],
            HEAD_DIM_MULTIPLIER = 4,
            HEAD_KERNEL = 2,
            HEAD_STRIDE = 2,
        ),

        BN_MOMENTUM = 0.1,
)