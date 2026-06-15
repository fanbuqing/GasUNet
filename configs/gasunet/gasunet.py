
_base_ = [
    '../_base_/datasets/gas1.py',
    '../_base_/default_runtime.py'
]

crop_size = (320, 256)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size)
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='GasUNet',
        in_channels=3,
        channels=64,
        num_branch_blocks=3,
        align_corners=False,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU', inplace=True),
        ),
    decode_head=dict(
        type='GasHead',
        in_channels=256,
        channels=256,
        num_classes=2,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU', inplace=True),
        align_corners=True,
        loss_decode=[
            dict(
                type='OhemCrossEntropy',
                thres=0.9,
                min_kept=131072,
                loss_weight=1.0),
            dict(type='BoundaryLoss', loss_weight=30.0)
        ]),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=(256, 256),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='GenerateEdge', edge_width=4),
    dict(type='PackSegInputs')
]
train_dataloader = dict(batch_size=16, dataset=dict(pipeline=train_pipeline))

iters = 160000
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)
# learning policy
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=0,
        power=0.9,
        begin=0,
        end=iters,
        by_epoch=False)
]
# training schedule for 120k
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=iters, val_interval=iters // 10)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', by_epoch=False, interval=iters // 10),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', draw=True, interval=1))

randomness = dict(seed=304)

#_base_ = [
#    '../_base_/datasets/gas1.py',
#    '../_base_/default_runtime.py'
#]
#
#crop_size = (320, 256)
#data_preprocessor = dict(
#    type='SegDataPreProcessor',
#    bgr_to_rgb=True,
#    pad_val=0,
#    seg_pad_val=255,
#    size=crop_size)
#norm_cfg = dict(type='BN', requires_grad=True)
#model = dict(
#    type='EncoderDecoder',
#    data_preprocessor=data_preprocessor,
#    backbone=dict(
#        type='GasUNet',
#        in_channels=3,
#        channels=64,
#        num_branch_blocks=3,
#        align_corners=False,
#        norm_cfg=norm_cfg,
#        act_cfg=dict(type='ReLU', inplace=True),
#        ),
#    decode_head=dict(
#        type='GasHead',
#        in_channels=256,
#        channels=256,
#        num_classes=2,
#        norm_cfg=norm_cfg,
#        act_cfg=dict(type='ReLU', inplace=True),
#        align_corners=True,
#       loss_decode=[
#           dict(
#               type='OhemCrossEntropy',
#               thres=0.9,
#               min_kept=131072,
#               loss_weight=1.0),
#           dict(type='BoundaryLoss', loss_weight=30.0)
#       ]
##         loss_decode=[dict(
##             type='FocalLoss',loss_name='loss_focal', loss_weight=1.0),dict(type='DiceLoss',loss_name='loss_dice', loss_weight=1.0)]
#        ),
#    train_cfg=dict(),
#    test_cfg=dict(mode='whole'))
#
#train_pipeline = [
#    dict(type='LoadImageFromFile'),
#    dict(type='LoadAnnotations'),
#    dict(
#        type='RandomResize',
#        scale=(320, 256),
#        ratio_range=(0.5, 2.0),
#        keep_ratio=True),
#    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
#    dict(type='RandomFlip', prob=0.5),
#    dict(type='PhotoMetricDistortion'),
#    dict(type='GenerateEdge', edge_width=4),
#    dict(type='PackSegInputs')
#]
#train_dataloader = dict(batch_size=16, dataset=dict(pipeline=train_pipeline))
#
#iters = 160000
## optimizer
#optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
#optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)
## learning policy
#param_scheduler = [
#    dict(
#        type='PolyLR',
#        eta_min=0,
#        power=0.9,
#        begin=0,
#        end=iters,
#        by_epoch=False)
#]
#
#vis_backends = [dict(type='LocalVisBackend')]
#visualizer = dict(
#    type='SegLocalVisualizer',
#    vis_backends=vis_backends,
#    name='visualizer'
#)
#
#train_cfg = dict(
#    type='IterBasedTrainLoop', max_iters=iters, val_interval=iters // 10)
#val_cfg = dict(type='ValLoop')
#test_cfg = dict(type='ValLoop')
#
##default_hooks = dict(
##    timer=dict(type='IterTimerHook'),
##    logger=dict(type='LoggerHook', interval=50),
##    param_scheduler=dict(type='ParamSchedulerHook'),
##    checkpoint=dict(type='CheckpointHook'),
##    sampler_seed=dict(type='DistSamplerSeedHook'),
##    visualization=dict(
##        type='SegVisualizationHook',
##        draw=True,        # 关键参数：必须为True
##        interval=1,       # 每个batch都可视化
##        show=False        # 保存到文件而不是显示
##    )
##)
#default_hooks = dict(
#    timer=dict(type='IterTimerHook'),
#    logger=dict(type='LoggerHook', interval=50),
#    param_scheduler=dict(type='ParamSchedulerHook'),
#    checkpoint=dict(
#        type='CheckpointHook',
#        save_best='mIoU',  # 基于验证集的mIoU保存最优模型
#        rule='greater',    # mIoU越大越好（若为损失则设为'less'）
#        max_keep_ckpts=5,  # 最多保留5个检查点（包括最优和最后）
#        interval=iters // 10,  # 每16000次迭代保存一次（与验证间隔一致）
#    ),
#    sampler_seed=dict(type='DistSamplerSeedHook'),
#    visualization=dict(
#        type='SegVisualizationHook',
#        draw=True,
#        interval=1,
#        show=False
#    )
#)
#randomness = dict(seed=304)
