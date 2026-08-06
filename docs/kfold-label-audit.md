# K 折标注审计

`run_rfdetr_kfold_label_audit.py` 按既有 `validation_fold` 做 5 折 OOF：每折用 4 折训练、1 折验证，最后输出模型指标和疑似标注错误候选。它只审计 v1 六类，不是产品自动验收流程。

## 一次准备

```bash
PYTHONPATH=src python tools/run_rfdetr_kfold_label_audit.py \
  --source 'core934=/path/to/core934/annotations_coco.json::/path/to/core934' \
  --source 'new151=/path/to/new151/instances_new151.coco.json::/path/to/new151' \
  --assignments /path/to/plan/assignments.csv \
  --workspace /path/to/kfold_workspace \
  --prepare-only
```

准备会生成 `workspace/plan/_READY.json`，重复运行可复用。图片使用硬链接，`workspace` 必须和原图位于同一文件系统；如果数据在 NAS，建议把产物写到本地 `--artifact-root`。

## 完整五折

训练镜像必须包含 RF-DETR 训练依赖（`rfdetr[train]`、PyTorch Lightning），通过 `--docker-image` 指定；预训练权重必须未见过这批数据。

```bash
PYTHONPATH=src python tools/run_rfdetr_kfold_label_audit.py \
  --source 'core934=/path/to/core934/annotations_coco.json::/path/to/core934' \
  --source 'new151=/path/to/new151/instances_new151.coco.json::/path/to/new151' \
  --assignments /path/to/plan/assignments.csv \
  --workspace /path/to/kfold_workspace \
  --artifact-root /path/to/local_artifacts \
  --docker-image <training-image> \
  --pretrain-weights /path/to/unseen_pretrain.pth
```

默认最多训练 80 epochs、early stopping patience 15；默认等待显存至少 22000 MiB 且利用率不超过 5% 的 GPU，五折串行执行。可重复 `--gpu 0 --gpu 1` 限定候选卡。成功阶段会写配置哈希和 `_SUCCESS.json`，配置一致且产物完整时可续跑；失败会写 `_FAILED.json` 并停止。

## 单折 smoke

先用一折确认镜像、GPU、数据和推理链路：

```bash
PYTHONPATH=src python tools/run_rfdetr_kfold_label_audit.py \
  --source 'core934=/path/to/core934/annotations_coco.json::/path/to/core934' \
  --source 'new151=/path/to/new151/instances_new151.coco.json::/path/to/new151' \
  --assignments /path/to/plan/assignments.csv \
  --workspace /path/to/kfold_workspace \
  --artifact-root /path/to/local_artifacts \
  --docker-image <training-image> \
  --pretrain-weights /path/to/unseen_pretrain.pth \
  --only-fold 0 --smoke
```

`--smoke` 强制 1 epoch，产物写入 `runs_smoke/fold_0/`；`--only-fold` 不生成五折汇总，也不会把 smoke checkpoint 当成完整训练结果。

## 过滤和审计口径

- 空图、含任意 GT 长边 `<70 px` 的帧，以及含 `Pedestrian`/`BoxTruck` 的帧会被剔除。
- 验证帧前后同数据源 ±3 秒内的所有相机帧不进该折训练集，避免时序泄漏。
- 内部框保持像素坐标；COCO/Label Studio 坐标只在适配边界转换。
- OOF 预测长边 `<70 px` 不导出；IoU ≥ 0.5、同类的一对一匹配用于 TP/FP/FN。
- 工作点默认 score `0.20`；mAP 保留 `0.001` 预测以维持完整置信度曲线。

## 主要输出

每折位于 `<artifact-root>/runs/fold_<n>/`：

- `oof_metrics_iou50.json`：TP/FP/FN、Precision、Recall、F1、mAP50、mAP50:95 及逐类结果；
- `annotation_error_candidates.csv`：疑似漏标、错类、定位偏差候选；
- `oof_predictions_70px.coco.json`、推理摘要、训练/推理日志和 checkpoint；
- `_SUCCESS.json` 或 `_FAILED.json`。

完整五折额外生成 `oof_combined/` 汇总。候选必须回到 Label Studio 人工确认，不能直接当作真实错误或自动验收结论。
