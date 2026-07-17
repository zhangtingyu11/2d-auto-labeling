import os

_base_ = (
    "/opt/conda/envs/car5-2d/lib/python3.10/site-packages/"
    "mmdet/.mim/configs/rtmdet/rtmdet_s_8xb32-300e_coco.py"
)

classes = ("Car", "Truck", "Bulldozer", "Excavator", "WaterTruck", "Sign")
metainfo = dict(
    classes=classes,
    palette=[
        (59, 130, 246),
        (239, 68, 68),
        (245, 158, 11),
        (139, 92, 246),
        (6, 182, 212),
        (34, 197, 94),
    ],
)

data_root = os.environ.get("CAR5_TRAIN_DATA_ROOT", "/opt/car5-data/car5_v1/images/")
annotation_root = os.environ.get(
    "CAR5_COCO_ROOT",
    "/opt/car5-workspace/Docker/annotations/private/phase0/coco_s00",
)
train_ann_file = os.path.join(annotation_root, "instances_train.json")
val_ann_file = os.path.join(annotation_root, "instances_val.json")

model = dict(bbox_head=dict(num_classes=len(classes)))
load_from = (
    "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/"
    "rtmdet_s_8xb32-300e_coco/"
    "rtmdet_s_8xb32-300e_coco_20220905_161602-387a891e.pth"
)

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    dataset=dict(
        data_root=data_root,
        ann_file=train_ann_file,
        data_prefix=dict(img=""),
        metainfo=metainfo,
        filter_cfg=dict(filter_empty_gt=False, min_size=1),
    ),
)
val_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    dataset=dict(
        data_root=data_root,
        ann_file=val_ann_file,
        data_prefix=dict(img=""),
        metainfo=metainfo,
        test_mode=True,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(
    ann_file=val_ann_file,
    metric="bbox",
    classwise=True,
)
test_evaluator = val_evaluator

max_epochs = 40
stage2_num_epochs = 5
base_lr = 0.00025
train_cfg = dict(
    max_epochs=max_epochs,
    val_interval=2,
    dynamic_intervals=[(max_epochs - stage2_num_epochs, 1)],
)
optim_wrapper = dict(
    optimizer=dict(lr=base_lr),
)
param_scheduler = [
    dict(
        type="LinearLR",
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=200,
    ),
    dict(
        type="CosineAnnealingLR",
        eta_min=base_lr * 0.05,
        begin=0,
        end=max_epochs,
        T_max=max_epochs,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
]
custom_hooks = [
    dict(
        type="EMAHook",
        ema_type="ExpMomentumEMA",
        momentum=0.0002,
        update_buffers=True,
        priority=49,
    ),
    dict(
        type="PipelineSwitchHook",
        switch_epoch=max_epochs - stage2_num_epochs,
        switch_pipeline={{_base_.train_pipeline_stage2}},
    ),
]
default_hooks = dict(
    logger=dict(interval=20),
    checkpoint=dict(
        interval=1,
        max_keep_ckpts=3,
        save_best="coco/bbox_mAP_50",
        rule="greater",
    ),
)
auto_scale_lr = dict(enable=False, base_batch_size=16)
randomness = dict(seed=20260717, deterministic=False)
env_cfg = dict(cudnn_benchmark=True)

visualizer = dict(
    vis_backends=[dict(type="LocalVisBackend")],
    name="visualizer",
)
