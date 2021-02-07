_base_ = [
    '../_base_/models/cascade_rcnn_mydartsnet_fpn.py',
    '../_base_/datasets/voc_ssdd.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

total_epochs = 12