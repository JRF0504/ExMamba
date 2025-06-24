
# dataset settings
dataset_type = 'PotsdamDataset'
data_root = '../data/potsdam'
crop_size = (512, 512)
train_pipeline = [
    dict(
        type='RandomMosaic',
        prob=0.5,
        img_scale=(1024, 1024),
        pad_val=0,
        seg_pad_val=255
    ),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomRotate', prob=0.5, degree=10),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(512, 512), keep_ratio=True,interpolation='nearest'),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='PackSegInputs')
]
train_dataset = dict(
    type='MultiImageMixDataset',
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
        img_path='img_dir/train', seg_map_path='ann_dir/train'),
        pipeline=[dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations')]
            ),
    pipeline=train_pipeline
)
train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=train_dataset
    )
val_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val'),
        pipeline=test_pipeline))

tta_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='TestTimeAug',
        transforms=[
            [dict(type='Resize', scale_factor=1.0, keep_ratio=True, interpolation='nearest'),
             dict(type='Resize', scale_factor=0.75, keep_ratio=True, interpolation='nearest'),
             dict(type='Resize', scale_factor=1.25, keep_ratio=True, interpolation='nearest'),
             dict(type='Resize', scale_factor=1.5, keep_ratio=True, interpolation='nearest'),
             dict(type='Resize', scale_factor=0.5, keep_ratio=True, interpolation='nearest')],

            [dict(type='RandomFlip', prob=1.0, direction='horizontal'),
             dict(type='RandomFlip', prob=1.0, direction='vertical'),
             dict(type='RandomFlip', prob=0.0)],
            [dict(type='LoadAnnotations', reduce_zero_label=True)],
            [dict(type='PackSegInputs')]
        ]
    )
]

test_dataloader = val_dataloader

val_evaluator = dict(
    type='IoUMetric', iou_metrics=['mIoU'])  # 'mDice', 'mFscore'
test_evaluator = val_evaluator
