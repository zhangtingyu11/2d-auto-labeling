_base_ = "./car5_rtmdet_s_pilot.py"

train_dataloader = dict(dataset=dict(indices=256))
val_dataloader = dict(dataset=dict(indices=120))
test_dataloader = val_dataloader

train_cfg = dict(max_epochs=1, val_interval=1, dynamic_intervals=None)
param_scheduler = [
    dict(
        type="LinearLR",
        start_factor=0.1,
        by_epoch=False,
        begin=0,
        end=10,
    )
]
custom_hooks = [
    dict(
        type="EMAHook",
        ema_type="ExpMomentumEMA",
        momentum=0.0002,
        update_buffers=True,
        priority=49,
    )
]
default_hooks = dict(
    logger=dict(interval=5),
    checkpoint=dict(interval=1, max_keep_ckpts=1),
)
