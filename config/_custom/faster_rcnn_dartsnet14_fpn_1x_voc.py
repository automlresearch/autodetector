# _base_ = [
#     '../_base_/models/faster_rcnn_dartsnet_fpn_ssdd.py', '../_base_/datasets/voc_ssdd.py',
#     '../_base_/default_runtime.py'
# ]
_base_ = [
    '../_base_/models/faster_rcnn_dartsnet_fpn3_ssdd.py', '../_base_/datasets/voc_ssdd.py',
    '../_base_/default_runtime.py'
]

model = dict(roi_head=dict(bbox_head=dict(num_classes=1)))

data = dict(samples_per_gpu=2, workers_per_gpu=2)
# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.00005)
optimizer_config = dict(grad_clip=None)
# learning policy
# actual epoch = 3 * 3 = 9
lr_config = dict(policy='step', step=[3])
# work_dir = './work_dirs/faster_rcnn_x101_64x4d_fpn_1x'
# runtime settings
total_epochs = 12  # actual epoch = 4 * 3 = 12
work_dir = './work_dirs/faster_rcnn_dartsnet/0610_l14_best'
