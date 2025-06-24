custom_imports = dict(imports=['mmseg.vimm', 'mmseg.efficientvim', 'mmseg.vimm_vis'], allow_failed_imports=False)
default_scope = 'mmseg'
# environment
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
vis_backends=[dict(type='LocalVisBackend'),
              dict(type='TensorboardVisBackend'),
              dict(type='WandbVisBackend')]
log_level = 'INFO'
log_processor = dict(by_epoch=False)
load_from = None  # 从文件中加载检查点(checkpoint)
resume = False  # 是否从已有的模型恢复