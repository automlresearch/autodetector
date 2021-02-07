dataset_type = 'SROICocoDataset'
data_root = '/home/p/Documents/data/SonarROI/comprehensive/'
img_norm_cfg = dict(
    mean=[0.581,0.44,0.136], std=[0.116,0.113,0.074], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=3,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'trainval.json',
        img_prefix=data_root + 'Images/',
        # ann_file=[data_root + 'trainval.json','/home/p/Documents/data/SonarROI/negative/trainval.json']
        # img_prefix=[data_root + 'Images/','/home/p/Documents/data/SonarROI/negative/Images/']
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'test.json',
        img_prefix=data_root + 'Images/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test.json',
        img_prefix=data_root + 'Images/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')




# dataset_type = 'SROICocoDataset'
# data_root = '/home/p/Documents/data/SonarROI/sidelooking/'
# img_norm_cfg = dict(
#     mean=[0.581,0.44,0.136], std=[0.116,0.113,0.074], to_rgb=True)
# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
#     dict(type='RandomFlip', flip_ratio=0.5),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='Pad', size_divisor=32),
#     dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
# ]
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         type='MultiScaleFlipAug',
#         img_scale=(1333, 800),
#         flip=False,
#         transforms=[
#             dict(type='Resize', keep_ratio=True),
#             dict(type='RandomFlip'),
#             dict(type='Normalize', **img_norm_cfg),
#             dict(type='Pad', size_divisor=32),
#             dict(type='ImageToTensor', keys=['img']),
#             dict(type='Collect', keys=['img']),
#         ])
# ]
# data = dict(
#     samples_per_gpu=1,
#     workers_per_gpu=0,
#     train=dict(
#         type=dataset_type,
#         ann_file=data_root + 'trainval1.json',
#         img_prefix=data_root + 'Images/',
#         pipeline=train_pipeline),
#     val=dict(
#         type=dataset_type,
#         ann_file=data_root + 'test1.json',
#         img_prefix=data_root + 'Images/',
#         pipeline=test_pipeline),
#     test=dict(
#         type=dataset_type,
#         ann_file=data_root + 'test1.json',
#         img_prefix=data_root + 'Images/',
#         pipeline=test_pipeline))
# evaluation = dict(interval=1, metric='bbox')
