_base_ = [
    '../../_base_/models/SSDD/faster_rcnn_r50_fpn.py', '../../_base_/datasets/voc_ssdd.py',#'../_base_/_custom_datasets/voc0712.py',
    '../../_base_/default_runtime.py'
]
# data
data = dict(samples_per_gpu=2, workers_per_gpu=2)
# optimizer
optimizer = dict(
    type='SGD',
    lr=0.0001,
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

work_dir = './work_dirs/SSDD/fasterrcnn_r50_fpn/50e/'

