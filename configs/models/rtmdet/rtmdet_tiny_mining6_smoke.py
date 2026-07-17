"""One-epoch subset run used to validate the full 2D training path."""

_base_ = './rtmdet_tiny_mining6_40e.py'

train_dataloader = dict(
    batch_size=4,
    num_workers=1,
    dataset=dict(indices=120))
val_dataloader = dict(
    batch_size=4,
    num_workers=1,
    dataset=dict(indices=60))
test_dataloader = val_dataloader

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=1,
    val_interval=1,
    dynamic_intervals=None)
param_scheduler = []
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
]
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=5),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=1))

