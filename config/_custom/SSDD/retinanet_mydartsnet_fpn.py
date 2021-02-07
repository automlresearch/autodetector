_base_ = [
    '../../_base_/models/SSDD/retinanet_mydartsnet_fpn.py',
    '../../_base_/datasets/voc_ssdd.py',
#    '../../_base_/schedules/schedule_1x.py',
    '../../_base_/default_runtime.py'
]
cudnn_benchmark = True
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    neck=dict(
        relu_before_extra_convs=True,
        no_norm_on_lateral=True,
        norm_cfg=norm_cfg),
    bbox_head=dict(type='RetinaSepBNHead', num_ins=5, norm_cfg=norm_cfg))
# training and testing settings
train_cfg = dict(assigner=dict(neg_iou_thr=0.5))
# dataset settings
img_norm_cfg = dict(
    mean=[0.2, 0.2, 0.2], std=[0.2, 0.2, 0.2], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=(640, 640),
        ratio_range=(0.8, 1.2),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(640, 640)),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=(640, 640)),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 640),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=64),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=0,
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))
# optimizer
optimizer = dict(
    type='SGD',
    lr=0.01,
    momentum=0.9,
    weight_decay=0.0001,
    paramwise_cfg=dict(norm_decay_mult=0, bypass_duplicate=True))
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=0.1,
    step=[30, 40])
# runtime settings
total_epochs = 50
# data = dict(
#     samples_per_gpu=4,
#     workers_per_gpu=0)

#optimizer = dict(type='SGD', lr=0.007, momentum=0.9, weight_decay=0.0001)

work_dir = './work_dirs/SSDD/retinanet_mydartsnet_fpn_self_trained/1x/'