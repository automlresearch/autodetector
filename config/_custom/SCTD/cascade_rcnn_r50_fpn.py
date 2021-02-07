_base_ = [
    '../../_base_/models/SCTD/cascade_rcnn_r50_fpn.py',
    '../../_base_/datasets/voc_sctd.py',
    '../../_base_/schedules/schedule_3x.py', '../../_base_/default_runtime.py'
]
work_dir = './work_dirs/SCTD_robust/cascade_rcnn_r50_nasfpn/3x'
