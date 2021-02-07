_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py', '../_base_/datasets/voc_ssdd.py',#'../_base_/_custom_datasets/voc0712.py',
    '../_base_/default_runtime.py'
]
# model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101))
# model = dict(roi_head=dict(bbox_head=dict(num_classes=1)))
# optimizer
optimizer = dict(type='SGD', lr=0.0001, momentum=0.9, weight_decay=0.000001)
optimizer_config = dict(grad_clip=None)
data = dict(samples_per_gpu=2, workers_per_gpu=2)
# learning policy
# actual epoch = 3 * 3 = 9
lr_config = dict(policy='step', step=[3])
# runtime settings
total_epochs = 12  # actual epoch = 4 * 3 = 12
