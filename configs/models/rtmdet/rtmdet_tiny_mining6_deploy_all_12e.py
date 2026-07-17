"""Deployment fine-tune using every reviewed CAR5 daytime camera view.

This checkpoint is for deployment after model selection. It must never be used
to report fold-0 generalization metrics because s00 is included in training.
"""

_base_ = './rtmdet_tiny_mining6_40e.py'

all_annotations = '{{$CAR5_COCO_ALL:artifacts/release_v1_hashed/coco/annotations_all.json}}'  # noqa: E501
v1_checkpoint = '{{$CAR5_V1_CHECKPOINT:artifacts/training/rtmdet_tiny_mining6_40e/best_coco_bbox_mAP_epoch_38.pth}}'  # noqa: E501

# Preserve the complete 16:9 camera frame. Removing Mosaic, MixUp, and random
# crops is important for large machines that touch an image edge.
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
    num_workers=4,
    persistent_workers=False,
    dataset=dict(
        ann_file=all_annotations,
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=train_pipeline))

# This all-data evaluation only checks fit/regression. It is not an independent
# generalization score; the immutable fold-0 checkpoint remains the benchmark.
val_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=False,
    dataset=dict(ann_file=all_annotations))
test_dataloader = val_dataloader
val_evaluator = dict(ann_file=all_annotations, classwise=True)
test_evaluator = val_evaluator

optim_wrapper = dict(optimizer=dict(lr=0.0004))
auto_scale_lr = dict(enable=True, base_batch_size=16)
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=2)
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=False,
        begin=0,
        end=100),
    dict(
        type='CosineAnnealingLR',
        eta_min=0.00004,
        begin=0,
        end=12,
        T_max=12,
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
]
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=20),
    checkpoint=dict(
        type='CheckpointHook', interval=2, max_keep_ckpts=3, save_best='coco/bbox_mAP'))

load_from = v1_checkpoint
resume = False
randomness = dict(seed=20260717, deterministic=False)
