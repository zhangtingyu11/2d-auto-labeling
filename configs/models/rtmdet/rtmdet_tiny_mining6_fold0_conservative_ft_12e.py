"""Honest fold-0 fine-tune with full-frame conservative augmentation."""

_base_ = './rtmdet_tiny_mining6_deploy_all_12e.py'

fold_root = '{{$CAR5_COCO_FOLD_ROOT:artifacts/release_v1_hashed/coco/fold_0/}}'

train_dataloader = dict(dataset=dict(ann_file=fold_root + 'train.json'))
val_dataloader = dict(dataset=dict(ann_file=fold_root + 'validation.json'))
test_dataloader = val_dataloader
val_evaluator = dict(ann_file=fold_root + 'validation.json', classwise=True)
test_evaluator = val_evaluator

# Keep the benchmark immutable: initialization is the old fold-0 V1 checkpoint
# and s00 remains absent from training throughout this experiment.
