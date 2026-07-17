"""RTMDet-S candidate using the measured RTMDet reference augmentation."""

_base_ = './rtmdet_tiny_mining6_40e.py'

pretrained = '{{$CAR5_RTMDET_S_CHECKPOINT:artifacts/models/rtmdet/rtmdet_s_coco_387a891e.pth}}'  # noqa: E501

model = dict(
    backbone=dict(
        deepen_factor=0.33,
        widen_factor=0.5,
        init_cfg=None),
    neck=dict(
        in_channels=[128, 256, 512],
        out_channels=128,
        num_csp_blocks=1),
    bbox_head=dict(
        in_channels=128,
        feat_channels=128,
        exp_on_reg=False,
        num_classes=6))

load_from = pretrained
