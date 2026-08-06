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

## 参数说明

所有参数也可以通过下面的命令查看：

```bash
PYTHONPATH=src python tools/run_rfdetr_kfold_label_audit.py --help
```

| 参数 | 必填/默认值 | 含义 |
| --- | --- | --- |
| `--source NAME=COCO_JSON::IMAGE_ROOT` | 必填，可重复 | 一个输入数据包。`NAME` 是 assignments 中使用的包名；`COCO_JSON` 是标注文件；`IMAGE_ROOT` 是 COCO `file_name` 的根目录。多个包分别传一次。 |
| `--assignments PATH` | 必填 | 分折 CSV。每行必须包含 `package`、`source_image_id`、`source_dataset`、`camera`、`timestamp`、`validation_fold`；折号从 0 开始。 |
| `--workspace PATH` | 必填 | 保存准备后的各折数据、`plan/_READY.json` 和运行锁。图片通过硬链接准备，因此必须与原图位于同一文件系统。 |
| `--artifact-root PATH` | 默认等于 `workspace` | 保存 checkpoint、日志、预测和指标。可放在本地磁盘，即使 `workspace` 位于 NAS。 |
| `--folds N` | `5` | 总折数；assignments 中的 `validation_fold` 必须在 `0..N-1`。 |
| `--only-fold N` | 默认全部折 | 只运行指定的零基折号，不生成五折合并报告。常与 `--smoke` 一起使用。 |
| `--minimum-long-side-px N` | `70` | GT 中只要有一个框的原始浮点长边 `< N`，整帧剔除；推理时长边 `< N` 的预测框也不导出。单位为像素。 |
| `--purge-seconds SEC` | `3.0` | 对每个验证帧，剔除同数据源时间差绝对值不超过该值的训练候选，且跨相机执行。单位为秒；`0` 仍会剔除同时间戳帧。 |
| `--prepare-only` | 默认关闭 | 只过滤、分折并验证数据，然后退出；不等待 GPU、不启动 Docker、不训练。 |
| `--docker-image IMAGE` | `car5-rfdetr:v4.4-70px-eval` | 训练、推理和评估所用的 RF-DETR Docker 镜像名或 digest。镜像必须包含训练依赖及仓库训练入口。 |
| `--pretrain-weights PATH` | 默认使用镜像/模型默认初始化 | 指定初始化权重。该权重不得训练或验证过本次被审计图片，否则会造成 OOF 泄漏。 |
| `--model NAME` | `medium` | 传给 RF-DETR 训练入口的模型规格。必须是所用镜像支持的名称。 |
| `--taxonomy NAME` | `car5-v1-six-class` | 写入训练配置和产物溯源的类别体系版本；本审计实际数据固定使用 v1 六个活跃类别。 |
| `--epochs N` | `80` | 每折最大训练 epoch 数；可能因 early stopping 提前结束。`--smoke` 会将它强制改为 1。 |
| `--smoke` | 默认关闭 | 启用单 epoch 冒烟训练，并把产物隔离到 `runs_smoke/`，避免被正式训练续跑逻辑误用。 |
| `--early-stopping-patience N` | `15` | 验证指标连续多少个 epoch 没有改善后停止训练。 |
| `--batch-size N` | `4` | 每个设备步骤输入的图像数；也用于批量推理。显存不足时优先调小。 |
| `--grad-accum-steps N` | `4` | 累积多少个设备步骤后执行一次优化器更新；默认有效批量约为 `4 × 4 = 16` 张。 |
| `--num-workers N` | `0` | 训练 DataLoader 的 worker 数。`0` 表示在主进程加载，稳定但可能较慢。 |
| `--prediction-threshold SCORE` | `0.001` | 推理导出预测的最低置信度，用于保留完整排序曲线计算 COCO mAP；取值 `0..1`。它不直接决定 TP/FP/FN。 |
| `--operating-score-threshold SCORE` | `0.20` | 计算 TP、FP、FN、Precision、Recall、F1 和错误候选时使用的置信度工作点；取值 `0..1`。它不改变 mAP 曲线。 |
| `--gpu INDEX` | 默认所有可见 GPU，可重复 | 限制候选物理 GPU，例如 `--gpu 0 --gpu 1`。五折仍串行，只会从候选卡中选择一张空闲卡。 |
| `--minimum-free-memory-mib N` | `22000` | GPU 被视为可用所需的最小空闲显存，单位 MiB。 |
| `--maximum-gpu-utilization PCT` | `5` | GPU 被视为空闲所允许的最大核心利用率，单位百分比。 |
| `--idle-confirmations N` | `3` | 同一 GPU 必须连续满足空闲条件多少次才会加锁并启动该折。 |
| `--poll-seconds N` | `60` | GPU 空闲状态轮询间隔，单位秒；脚本允许 `1..60`。 |

`--prediction-threshold` 和 `--operating-score-threshold` 刻意分开：前者服务于需要完整置信度排序的 mAP，后者服务于一个明确工作点下的 TP/FP/FN。调高工作点通常会减少 FP、增加 FN，但不应改变同一份预测集合计算出的 mAP。

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
