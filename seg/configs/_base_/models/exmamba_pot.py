norm_cfg = dict(type='BN', requires_grad=True)  
data_preprocessor = dict(  # 数据预处理的配置项，通常包括图像的归一化和增强
    type='SegDataPreProcessor',  # 数据预处理的类型
    # mean = [85.80106741, 91.71422378, 84.93052268],
    # std = [35.85278217, 35.18219105, 36.55390113], # potsdam
    mean = [84.93105, 91.71646, 85.79928,], # potsdam
    std = [36.5273, 35.150696, 35.82283],
    # mean=[119.86990736,  81.21329333,  80.0872423],
    # std=[54.99957592, 39.34208628, 37.47628523], # 0.62
    size=(512, 512),
    bgr_to_rgb=False,  # 是否将图像从 BGR 转为 RGB
    pad_val=0,  # 图像的填充值
    seg_pad_val=255)  # 'gt_seg_map'的填充值

model = dict(
    type='EncoderDecoder',  # 分割器(segmentor)的名字
    data_preprocessor=data_preprocessor,
     backbone=dict(
        type='ViMM',
        in_dim=3,
        embed_dim=[64, 128, 256],
        num_experts=10,
        top_k=2,
        depths=[2,2,2],
        mlp_ratio=4.,
        ssd_expand=1.,
        state_dim=[49,25,9],
        router='topk',
        dropout=0.1,
        shared=True,
        drop_router=0
        # checkpoint='/home/jrf/mamba/src/checkpoint/vimm_small.pth'
    ),
    decode_head=dict(
        type='UPerHead',
        in_channels=[128, 256, 256],
        in_index=[1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=6,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=256,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=6,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'), )
tta_model = dict(
    type='SegTTAModel',
)