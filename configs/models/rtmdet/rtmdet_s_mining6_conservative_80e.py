"""Formal RTMDet-S baseline for the six-class CAR5 daytime dataset.

This configuration follows the reviewed technical route: aspect ratio is
preserved and Mosaic, MixUp, random crops, and horizontal flips are disabled.
"""

_base_ = '../../../vendor/mmdetection-3.2.0/configs/rtmdet/rtmdet_s_8xb32-300e_coco.py'

classes = ('Car', 'Truck', 'Bulldozer', 'Excavator', 'WaterTruck', 'Sign')
metainfo = dict(classes=classes)

data_root = '{{$CAR5_DATA_ROOT:../car5_20260611_day_02/car5_20260611_day_02/}}'
ann_root = '{{$CAR5_COCO_FOLD_ROOT:artifacts/release_v1_hashed/coco/fold_0/}}'
pretrained = '{{$CAR5_RTMDET_S_CHECKPOINT:artifacts/models/rtmdet/rtmdet_s_coco_387a891e.pth}}'  # noqa: E501

model = dict(
    backbone=dict(init_cfg=None),
    bbox_head=dict(num_classes=len(classes)),
    test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65)))

# Conservative daytime augmentation. Images are resized with keep_ratio=True
# and padded to 640x640, never geometrically stretched.
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(
        type='YOLOXHSVRandomAug',
        hue_delta=3,
        saturation_delta=15,
        value_delta=15),
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(type='PackDetInputs'),
]

train_dataloader = dict(
    batch_size=16,
    num_workers=3,
    persistent_workers=False,
    dataset=dict(
        data_root=data_root,
        ann_file=ann_root + 'train.json',
        data_prefix=dict(img=''),
        metainfo=metainfo,
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=16,
    num_workers=3,
    persistent_workers=False,
    dataset=dict(
        data_root=data_root,
        ann_file=ann_root + 'validation.json',
        data_prefix=dict(img=''),
        metainfo=metainfo))
test_dataloader = val_dataloader

val_evaluator = dict(ann_file=ann_root + 'validation.json', classwise=True)
test_evaluator = val_evaluator

optim_wrapper = dict(type='AmpOptimWrapper', loss_scale='dynamic')
auto_scale_lr = dict(enable=True, base_batch_size=16)

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=80,
    val_interval=5)
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-5,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='CosineAnnealingLR',
        eta_min=1e-4,
        begin=40,
        end=80,
        T_max=40,
        by_epoch=True,
        convert_to_iter_based=True),
]

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
    dict(
        type='EarlyStoppingHook',
        monitor='coco/bbox_mAP',
        rule='greater',
        min_delta=0.002,
        patience=5,
        strict=True),
]

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=20),
    checkpoint=dict(
        type='CheckpointHook', interval=5, max_keep_ckpts=3, save_best='coco/bbox_mAP'))

load_from = pretrained
resume = False
randomness = dict(seed=20260717, deterministic=False)
