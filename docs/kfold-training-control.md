# Label Studio 一键 K 折训练入口

该入口把 K 折训练控制页挂载到 Label Studio 同一地址下：

```text
Label Studio 项目页 -> K折训练 -> /kfold-training/
```

网页只填写训练版本名。点击“一键读取、训练并更新”后，固定流水线依次执行：

```text
Label Studio API 快照
  -> 人工标注过滤与五折继承
  -> RF-DETR 五折训练和 OOF 推理
  -> 训练期间标签变更检查
  -> 幂等更新锁定审计提示
```

## 自动读取与过滤

worker 使用服务端环境变量中的 Label Studio API Token 获取项目原生 JSON 导出，
Token 不返回浏览器，也不写入任务元数据或命令行。转换器执行以下强制规则：

- 每个任务必须存在已提交且未取消的 annotation；未提交任务不会被当成空图；
- 只保留 `type=rectanglelabels`、`from_name=label` 且非只读的人工框；
- 删除 `audit_hint`、`prediction_reference`、`review_issue` 等非人工结果；
- 删除 `candidate_summary`、`review_tag` 和 `kfold_*` 等审计字段；
- 只接受固定八个训练类别，“可能漏标”等提示不能成为训练类别；
- 按完整源图片路径继承上一轮 `validation_fold`，不同相机不会因文件同名而混淆；
- 明确提交的零框任务会记录为空图，随后由 K 折准备阶段按既定规则排除。

转换报告记录任务数、人工框数、删除的提示数、空图数以及每折图片数。原始 API
快照、过滤后的 COCO 和 assignments 都保存在该任务目录，用于复现和审计。

## 训练和自动回写

网页用户不能设置命令、镜像或文件路径。后台固定：

- Docker 镜像 `car5-rfdetr:v4.4-70px-train-user`；
- batch size 32；
- gradient accumulation 1；
- 最多 80 epochs，early stopping patience 15；
- 每次使用新的 workspace 和 artifact 目录；
- 不使用上一轮任何折的 checkpoint 作为共享预训练权重；
- 同时最多运行一个五折任务。

五折成功后，worker 再次从 Label Studio 读取并转换人工标注。两次规范化训练输入的
SHA-256 不一致时，任务进入 `stale`，保留模型产物但不回写，避免用旧提示覆盖训练
期间的新人工修改。

输入未变化时，worker 保存回写前的完整项目 JSON，然后调用幂等同步器：

- 只 PATCH 现有 task、annotation、draft 和项目配置；
- `created_tasks=0`、`created_annotations=0`；
- 先删除历史审计提示，再按确定性 ID 写入本轮锁定提示；
- 人工 `label` 框保持可编辑且不会被删除；
- 重试不会累计重复提示。

网页服务只写入受限任务队列，不挂载 Docker socket。一个不监听网络的本机 worker
读取队列并执行固定训练与同步命令。浏览器用户不需要 Linux、sudo 或 Docker 权限。

## 服务启动

控制服务不需要 Label Studio Token：

```bash
CAR5_KFOLD_TRAINING_PASSWORD='<secret>' PYTHONPATH=src \
python3 tools/serve_kfold_training_control.py \
  --assignment-template /path/to/previous/assignments.csv \
  --image-root /path/to/dataset/root \
  --workspace-root /path/to/new/workspaces \
  --artifact-root /path/to/new/artifacts \
  --repository-root /path/to/2d-auto-labeling \
  --project-id 3 \
  --gpu 3 --gpu 4
```

本机 worker 通过环境变量持有 API Token：

```bash
CAR5_LABEL_STUDIO_API_TOKEN='<token>' PYTHONPATH=src \
python3 tools/run_kfold_training_worker.py \
  --jobs-root /path/to/artifacts/control_jobs \
  --assignment-template /path/to/previous/assignments.csv \
  --image-root /path/to/dataset/root \
  --workspace-root /path/to/new/workspaces \
  --artifact-root /path/to/new/artifacts \
  --repository-root /path/to/2d-auto-labeling \
  --label-studio-url http://127.0.0.1:8090 \
  --label-studio-project-id 3 \
  --gpu 3 --gpu 4
```

worker 不监听任何端口，同时只能运行一个 worker 和一个五折任务。`queued`、
`exporting`、`preparing`、`interrupted` 和 `syncing` 状态可在 worker 重启后安全恢复；
已有训练哨兵和回写备份会被复用。

控制服务监听 `8091`。`deploy/kfold-training-control/nginx.conf` 在同机对外入口中把
`/kfold-training/` 转发到控制服务，其余请求转发到 Label Studio，并在项目页注入
“K折训练”入口。

当前部署使用 HTTP 内网访问，Basic 凭据在传输层不加密。若访问范围超出可信内网，
必须配置 HTTPS。回滚时停止训练控制容器并恢复原有纯 TCP 代理即可；Label Studio
数据库和项目配置不需要修改。
