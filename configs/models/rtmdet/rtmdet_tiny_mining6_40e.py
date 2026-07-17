"""RTMDet-Tiny fine-tuning config for the six-class CAR5 daytime set."""

_base_ = '../../../vendor/mmdetection-3.2.0/configs/rtmdet/rtmdet_tiny_8xb32-300e_coco.py'

classes = ('Car', 'Truck', 'Bulldozer', 'Excavator', 'WaterTruck', 'Sign')
metainfo = dict(classes=classes)

data_root = '{{$CAR5_DATA_ROOT:../car5_20260611_day_02/car5_20260611_day_02/}}'
ann_root = '{{$CAR5_COCO_FOLD_ROOT:artifacts/release_v1_hashed/coco/fold_0/}}'
pretrained = '{{$CAR5_RTMDET_TINY_CHECKPOINT:artifacts/models/rtmdet/rtmdet_tiny_coco_78e30dcc.pth}}'  # noqa: E501

# The full COCO checkpoint is loaded below, so an additional ImageNet backbone
# download is neither needed nor desirable in an offline production setup.
model = dict(
    backbone=dict(init_cfg=None),
    bbox_head=dict(num_classes=len(classes)),
    test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65)))

train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=False,
    dataset=dict(
        data_root=data_root,
        ann_file=ann_root + 'train.json',
        data_prefix=dict(img=''),
        metainfo=metainfo,
        # Reviewed empty images are valuable negative examples in this project.
        filter_cfg=dict(filter_empty_gt=False, min_size=32)))

val_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=False,
    dataset=dict(
        data_root=data_root,
        ann_file=ann_root + 'validation.json',
        data_prefix=dict(img=''),
        metainfo=metainfo))
test_dataloader = val_dataloader

val_evaluator = dict(ann_file=ann_root + 'validation.json')
test_evaluator = val_evaluator

# One 8 GB laptop GPU: AMP plus automatic LR scaling from RTMDet's reference
# batch size keeps this configuration both fast and memory-safe.
optim_wrapper = dict(type='AmpOptimWrapper', loss_scale='dynamic')
auto_scale_lr = dict(enable=True, base_batch_size=16)

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=40,
    val_interval=5,
    dynamic_intervals=[(35, 1)])
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
        begin=20,
        end=40,
        T_max=20,
        by_epoch=True,
        convert_to_iter_based=True),
]

train_pipeline_stage2 = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomResize',
        scale=(640, 640),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(640, 640)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(type='PackDetInputs'),
]
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
    dict(
        type='PipelineSwitchHook',
        switch_epoch=35,
        switch_pipeline=train_pipeline_stage2),
]

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=20),
    checkpoint=dict(
        type='CheckpointHook', interval=5, max_keep_ckpts=3, save_best='coco/bbox_mAP'))

load_from = pretrained
resume = False
randomness = dict(seed=20260717, deterministic=False)
