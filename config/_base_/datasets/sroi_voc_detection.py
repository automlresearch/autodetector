# dataset settings
dataset_type = 'SROIDataset_1class'
data_root = '/home/p/Documents/data/SonarROI/'
img_norm_cfg = dict(mean=[0.2, 0.2, 0.2], std=[0.2, 0.2, 0.2], to_rgb=True)
# img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[1, 1, 1], to_rgb=True)
# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(type='Resize', img_scale=(1000, 600), keep_ratio=True),
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
#         img_scale=(1000, 600),
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
    train=dict(
        type='RepeatDataset',
        times=3,
        dataset=dict(
            type=dataset_type,
            ann_file=[
                data_root + 'sidelooking/ImageSets/Main/trainval.txt',
            ],
            img_prefix=[data_root + 'sidelooking/', ],
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'sidelooking/ImageSets/Main/test.txt',
        img_prefix=data_root + 'sidelooking/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'sidelooking/ImageSets/Main/test.txt',
        img_prefix=data_root + 'sidelooking/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='mAP')



# # dataset settings
# dataset_type = 'SROIDataset_1class'
# data_root = '/home/p/Documents/data/SonarROI/'
# img_norm_cfg = dict(mean=[0.2, 0.2, 0.2], std=[0.2, 0.2, 0.2], to_rgb=True)
# # img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[1, 1, 1], to_rgb=True)
# # train_pipeline = [
# #     dict(type='LoadImageFromFile'),
# #     dict(type='LoadAnnotations', with_bbox=True),
# #     dict(type='Resize', img_scale=(1000, 600), keep_ratio=True),
# #     dict(type='RandomFlip', flip_ratio=0.5),
# #     dict(type='Normalize', **img_norm_cfg),
# #     dict(type='Pad', size_divisor=32),
# #     dict(type='DefaultFormatBundle'),
# #     dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
# # ]
# # test_pipeline = [
# #     dict(type='LoadImageFromFile'),
# #     dict(
# #         type='MultiScaleFlipAug',
# #         img_scale=(1000, 600),
# #         flip=False,
# #         transforms=[
# #             dict(type='Resize', keep_ratio=True),
# #             dict(type='RandomFlip'),
# #             dict(type='Normalize', **img_norm_cfg),
# #             dict(type='Pad', size_divisor=32),
# #             dict(type='ImageToTensor', keys=['img']),
# #             dict(type='Collect', keys=['img']),
# #         ])
# # ]
#
#
# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(
#         type='Resize',
#         img_scale=(640, 640),
#         ratio_range=(0.8, 1.2),
#         keep_ratio=True),
#     dict(type='RandomCrop', crop_size=(640, 640)),
#     dict(type='RandomFlip', flip_ratio=0.5),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='Pad', size=(640, 640)),
#     dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
# ]
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         type='MultiScaleFlipAug',
#         img_scale=(640, 640),
#         flip=False,
#         transforms=[
#             dict(type='Resize', keep_ratio=True),
#             dict(type='RandomFlip'),
#             dict(type='Normalize', **img_norm_cfg),
#             dict(type='Pad', size_divisor=128),
#             dict(type='ImageToTensor', keys=['img']),
#             dict(type='Collect', keys=['img']),
#         ])
# ]
#
# data = dict(
#     samples_per_gpu=2,
#     workers_per_gpu=2,
#     train=dict(
#         type='RepeatDataset',
#         times=3,
#         dataset=dict(
#             type=dataset_type,
#             ann_file=[
#                 data_root + 'sidelooking/ImageSets/Main/trainval.txt',
#                 data_root + 'negative/ImageSets/Main/trainval.txt'
#             ],
#             img_prefix=[data_root + 'sidelooking/', data_root + 'negative/'],
#             pipeline=train_pipeline)),
#     val=dict(
#         type=dataset_type,
#         ann_file=data_root + 'sidelooking/ImageSets/Main/test.txt',
#         img_prefix=data_root + 'sidelooking/',
#         pipeline=test_pipeline),
#     test=dict(
#         type=dataset_type,
#         ann_file=data_root + 'sidelooking/ImageSets/Main/test.txt',
#         img_prefix=data_root + 'sidelooking/',
#         pipeline=test_pipeline))
# evaluation = dict(interval=1, metric='mAP')
