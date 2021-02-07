_base_ = [
    '../../_base_/models/SCTD/faster_rcnn_r50_fpn.py',
    '../../_base_/datasets/voc_sctd_nasfpn.py',
    '../../_base_/schedules/schedule_3x.py', '../../_base_/default_runtime.py'
]
# model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101))
# model = dict(roi_head=dict(bbox_head=dict(num_classes=1)))
# optimizer
optimizer = dict(type='SGD', lr=0.0002, momentum=0.9, weight_decay=0.000001)
optimizer_config = dict(grad_clip=None)
data = dict(samples_per_gpu=2, workers_per_gpu=2)

work_dir = './work_dirs/SCTD/faster_rcnn_r50_fpn/3x'