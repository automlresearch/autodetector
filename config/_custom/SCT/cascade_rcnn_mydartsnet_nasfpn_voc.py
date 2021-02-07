_base_ = [
    '../../_base_/models/SCT/cascade_rcnn_mydartsnet_fpn.py',
    '../../_base_/datasets/voc_sct_nasfpn.py',
    '../../_base_/schedules/schedule_3x.py', '../../_base_/default_runtime.py'
]

# model settings
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    neck=dict(type='NASFPN', stack_times=7, norm_cfg=norm_cfg),
    # bbox_head=dict(type='RetinaSepBNHead', num_ins=5, norm_cfg=norm_cfg)
    )
# training and testing settings
train_cfg = dict(assigner=dict(neg_iou_thr=0.5))
# dataset settings
img_norm_cfg = dict(
    mean=[0.581,0.44,0.136], std=[0.116,0.113,0.074], to_rgb=True)
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
            dict(type='Pad', size_divisor=128),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))

work_dir = './work_dirs/SCT/cascadercnn_mydartsnet_nasfpn_self_trained/3x/'
