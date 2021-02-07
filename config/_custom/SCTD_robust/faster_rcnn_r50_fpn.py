_base_ = [
    '../../_base_/models/SCTD/faster_rcnn_r50_fpn.py',
    '../../_base_/datasets/voc_sctd.py',
    '../../_base_/default_runtime.py'
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=0)
# optimizer
optimizer = dict(
    type='SGD',
    lr=0.0002,
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

work_dir = './work_dirs/SCTD_robust/faster_rcnn_r50_fpn/50e/lr00002'