_base_ = [
    '../../_base_/models/SSDD/cascade_rcnn_r50_fpn.py',
    '../../_base_/datasets/voc_ssdd.py',
    '../../_base_/schedules/schedule_3x.py', '../../_base_/default_runtime.py'
]

work_dir = './work_dirs/SSDD/cascade_rcnn_r50_fpn/'