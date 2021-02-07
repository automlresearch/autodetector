_base_ = [
    '../../_base_/models/SSDD_normal/cascade_rcnn_r50_fpn.py',
    '../../_base_/datasets/SSDD_normal/voc_ssdd.py',
    '../../_base_/schedules/schedule_1x.py', '../../_base_/default_runtime.py'
]

work_dir = './work_dirs/SSDD_normal/cascadercnn_r50_fpn/'
