_base_ = [
    '../../_base_/models/SROI/retinanet_mydartsnet_fpn.py',
    '../../_base_/datasets/sroi_coco_detection.py',
    '../../_base_/schedules/schedule_1x.py', '../../_base_/default_runtime.py'
]
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)

work_dir = './work_dirs/SROI/retinanet_mydartsnet_fpn'