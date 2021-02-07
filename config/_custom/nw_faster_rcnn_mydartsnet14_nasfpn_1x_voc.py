_base_ = [
    '../_base_/models/faster_rcnn_mydartsnet_nasfpn_ssdd.py', '../_base_/datasets/voc_ssdd_nasfpn.py',
    '../_base_/default_runtime.py'
]
# _base_ = [
#     '../_base_/_custom_models/faster_rcnn_r50_fpn.py', '../_base_/datasets/voc_ssdd.py',
#     '../_base_/default_runtime.py'
# ]
# _base_ = [
#     '../_base_/_custom_models/faster_rcnn_darts_caffe_c4.py', '../_base_/datasets/voc_ssdd.py',
#     '../_base_/default_runtime.py'
# ]
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
# work_dir = './work_dirs/faster_rcnn_mydartsnet/0613/dartsnet_imagenet_pretrained'
work_dir = './work_dirs/faster_rcnn_mydartsnet/0613/mydartsnet_ssdd_nasfpn_c3_256'