# 5 折训练集标注审计

该脚本用 4 折训练、1 折验证依次生成 5 份 OOF（out-of-fold）预测，供后续定位漏标、错标、类别混淆和明显偏框。

## 固定过滤口径

- GT 框尺寸按 COCO `xywh` 的原始浮点值 `max(width, height)` 计算，不做四舍五入。
- 一帧只要含有任意长边小于 70 px 的 GT 框，整帧从训练集和验证集剔除。
- 无 GT 框的空图整帧剔除。
- 训练类别严格使用 v1 六类；含 `BoxTruck` 或 `Pedestrian` 标注的帧整帧剔除，不能静默当作背景或类别别名。
- 验证推理中长边小于 70 px 的预测框不导出，因此不计 FP。
- 按既有 `validation_fold` 分折；验证帧前后 3 秒内、同数据源的所有相机帧都不进入该折训练集，避免同步或相邻帧泄漏。

## 准备并检查数据

```bash
PYTHONPATH=src python tools/run_rfdetr_kfold_label_audit.py \
  --source 'core934=/path/to/core934/annotations_coco.json::/path/to/core934' \
  --source 'new151=/path/to/new151/instances_new151.coco.json::/path/to/new151' \
  --assignments /path/to/plan/assignments.csv \
  --workspace /path/to/kfold_workspace \
  --prepare-only
```

数据集图片使用硬链接，避免重复占用空间，所以 `--workspace` 必须与原图位于同一文件系统。准备完成后会生成 `plan/_READY.json`，重复运行不会改写已准备的数据。

## 等待空闲 GPU 并串行训练

去掉 `--prepare-only` 即可。默认每 60 秒检查一次 GPU，连续 3 次满足“空闲显存至少 22000 MiB 且利用率不超过 5%”后开始一折；5 折严格串行执行。

```bash
PYTHONPATH=src nohup python tools/run_rfdetr_kfold_label_audit.py \
  --source 'core934=/path/to/core934/annotations_coco.json::/path/to/core934' \
  --source 'new151=/path/to/new151/instances_new151.coco.json::/path/to/new151' \
  --assignments /path/to/plan/assignments.csv \
  --workspace /path/to/kfold_workspace \
  > /path/to/kfold_workspace/runner.log 2>&1 &
```

可用重复的 `--gpu 0 --gpu 1` 限定候选卡。默认训练最多 80 epochs，early stopping patience 为 15；避免欠拟合模型制造大量伪标注错误。训练与推理分别写入带配置哈希的成功标记，再次运行会复用配置完全一致且产物完整的阶段；失败则生成 `_FAILED.json` 并停止。

默认从 RF-DETR 官方通用预训练权重开始，不能拿已经见过这批训练数据的现有业务 checkpoint 初始化，否则 OOF 结论会被数据泄漏污染。若有确认未见过这 1085 帧的通用权重，可显式传 `--pretrain-weights /path/to/weights.pth`。

每折输出包括：

- `train.log` 与 `inference.log`
- 最优训练 checkpoint
- `oof_predictions_70px.coco.json`
- `oof_inference_summary.json`
- `oof_metrics_iou50.json`（TP/FP/FN、Precision、Recall、F1、mAP50、mAP50:95，含逐类别结果）
- `annotation_error_candidates.csv`（疑似漏标、错标、类别混淆候选）
- `_SUCCESS.json` 或 `_FAILED.json`
- `oof_combined/metrics.json` 与 `oof_combined/annotation_error_candidates.csv`（五折整体汇总）

其中 TP 的判定是同一图像、同一类别、IoU ≥ 0.5，并采用最大总 IoU 的一对一分配；未匹配预测计 FP，未匹配 GT 计 FN。高 IoU 但类别不同的框会额外标成 `class_mismatch`，便于人工检查。

推理默认以 0.001 导出候选，保证 mAP 的置信度排序曲线完整；mAP 由 `pycocotools.COCOeval` 按标准 bbox 参数计算。TP/FP/FN、Precision、Recall、F1 默认使用 0.20 的审计工作点。可用 `--operating-score-threshold` 调整工作点，两类指标不会互相污染。

当前临时数据审计没有接入相机 host-vehicle ignore polygon，指标文件会明确记录 `ignore_regions_applied: false`，不能把这份 OOF 结果当作产品自动验收结果。

## 单折 smoke 验证

在完整五折前，用同一份已准备数据只跑一折 smoke：

```bash
PYTHONPATH=src python tools/run_rfdetr_kfold_label_audit.py \
  --source 'core934=/path/to/core934/annotations_coco.json::/path/to/core934' \
  --source 'new151=/path/to/new151/instances_new151.coco.json::/path/to/new151' \
  --assignments /path/to/plan/assignments.csv \
  --workspace /path/to/kfold_workspace \
  --artifact-root /path/to/local_artifacts \
  --only-fold 0 \
  --smoke
```

smoke 模式强制训练 1 epoch，产物写入 `<artifact-root>/runs_smoke/fold_0/`，不会被完整训练误当作可续跑 checkpoint；单折模式也不会生成五折合并指标。数据可继续放在 NAS `workspace`，checkpoint 和指标建议写入支持普通文件时间戳操作的本地文件系统。
