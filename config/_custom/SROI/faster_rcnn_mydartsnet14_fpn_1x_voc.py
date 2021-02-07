_base_ = [comprehensive_data
    '../../_base_/models/SROI/faster_rcnn_mydartsnet_fpn_ssdd.py', '../../_base_/datasets/sroi_coco_detection.py',
    '../../_base_/default_runtime.py'
]
# _base_ = [
#     '../_base_/_custom_models/faster_rcnn_r50_fpn.py', '../_base_/datasets/voc_ssdd.py',
#     '../_base_/default_runtime.py'
# ]
# _base_ = [
#     '../_base_/_custom_models/faster_rcnn_darts_caffe_c4.py', '../_base_/datasets/voc_ssdd.py',
#     '../_base_/default_runtime.py'
# ]
data = dict(samples_per_gpu=1, workers_per_gpu=0)
# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.00005)
optimizer_config = dict(grad_clip=None)
# learning policy
# actual epoch = 3 * 3 = 9
lr_config = dict(policy='step', step=[3])
# work_dir = './work_dirs/faster_rcnn_x101_64x4d_fpn_1x'
# runtime settings
total_epochs = 24  # actual epoch = 4 * 3 = 12
# work_dir = './work_dirs/faster_rcnn_mydartsnet/0613/dartsnet_imagenet_pretrained'
work_dir = './work_dirs/SROI/fasterrcnn_mydartsnet_fpn'
